// ---------------------------------------------------------------------------
// band_sram -- row-banded, double-buffered activation scratchpad with a
//              3x3 im2col window extractor.
//
// ---------------------------------------------------------------------------
// THE PROBLEM THIS SOLVES (ACCELERATOR_PLAN.md section 6)
//
// Holding whole feature maps on chip needs the largest live pair:
//     conv2 output  64 x 40 x 101 = 258,560 B
//     conv1 output  32 x 40 x 101 = 129,280 B
//                                   --------
//                                   387,840 B = 379 KiB at INT8
//
// 379 KiB of SRAM on an always-on part is not acceptable: SRAM would dominate
// both area and leakage, and leakage is what drains the battery between wake
// words. Copying the reference project here would hurt most -- their 34x34
// scratchpad holds one tile of a single-channel image, which does not
// generalise to multi-channel feature maps.
//
// THE FIX: a 3x3 convolution only ever needs THREE input rows live at once.
// Band the feature map by rows and stream the bands. Sized for the worst case
// across the three heavy layers, which come out almost perfectly balanced:
//
//     conv2  32 ch x 3 rows x 101 = 9,696 B
//     conv3  64 ch x 3 rows x  50 = 9,600 B
//     conv4 128 ch x 3 rows x  25 = 9,600 B
//
// That balance is a good sign the banding is the right shape for this model.
// DEPTH below is bytes per ROW bank, so the whole input band is ROWS*DEPTH and
// double buffering makes it 2*ROWS*DEPTH = 19,392 B. Against 379 KiB that is a
// 20x reduction on the input side, and it is what makes the M3 pooling fusion
// natural: a 2x2 max-pool consumes two output rows, which is exactly what the
// output band already holds.
//
// ---------------------------------------------------------------------------
// WHY A SLIDING WINDOW AND NOT 27 READ PORTS
//
// The array wants an im2col vector every cycle. Read naively that is
// WIN_CH x ROWS x 3 = 27 byte reads per cycle. But consecutive output columns
// overlap in two of their three taps, so the window is kept in registers and
// SHIFTED: each cycle drops the oldest tap column and loads ONE new column,
// which is WIN_CH x ROWS = 9 reads per cycle. Three times less read
// bandwidth for WIN_CH*ROWS*3 = 27 bytes of register.
//
// WIN_CH = 3 because a K = 16 tile of flat im2col taps spans at most
// ceil((16 + 8) / 9) = 3 channels. The array's K-vector is then a 16-byte
// slice of this 27-byte window, selected outside this module.
//
// PADDING: every convolution in the model is 3x3 with padding 1, so output
// column x reads input columns x-1, x and x+1. Columns outside [0, W) read as
// zero, which this module generates rather than requiring the filler to pad
// the buffer.
//
// NOTE ON MAPPING: the storage is modelled as register arrays with async reads,
// which is right for simulation and for a small FPGA mapping. For SAED14nm the
// three row banks become SRAM macros with SYNCHRONOUS reads, which adds one
// cycle to the load path. The window shift absorbs that without changing the
// external timing -- prime one cycle earlier. That swap is an M3 synthesis
// task, not a functional change.
// ---------------------------------------------------------------------------

`default_nettype none

module band_sram #(
    parameter int DEPTH  = 3232,  // bytes per row bank: max channels*width
    parameter int ROWS   = 3,     // rows live at once; 3 for a 3x3 kernel
    parameter int WIN_CH = 3      // channels held live in the window
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- Fill port: writes the bank the array is NOT reading -------------
    input  wire                       wr_en,
    input  wire                       wr_bank,
    input  wire [$clog2(ROWS)-1:0]    wr_row,
    input  wire [$clog2(DEPTH)-1:0]   wr_addr,   // channel*width + column
    input  wire [7:0]                 wr_data,

    // ---- Sweep control ---------------------------------------------------
    input  wire                       rd_bank,
    input  wire                       sweep_start,  // pulse: restart at x=0
    input  wire [$clog2(DEPTH)-1:0]   rd_base,      // first channel's offset
    input  wire [$clog2(DEPTH)-1:0]   ch_stride,    // = row width W
    input  wire [15:0]                row_width,    // W, for edge padding

    // ---- Window output: [((ch*ROWS + row)*3 + tap)*8 +: 8] ----------------
    output reg                        win_vld,
    output reg  [WIN_CH*ROWS*3*8-1:0] win,
    output reg  [15:0]                win_x         // column this window is for
);

    localparam int AW = $clog2(DEPTH);

    // ---- Storage: 2 banks x ROWS row-banks x DEPTH bytes ------------------
    reg [7:0] mem [0:1][0:ROWS-1][0:DEPTH-1];

    integer bi, ri, di;
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_bank][wr_row][wr_addr] <= wr_data;
    end

    // ---- Sweep counter ----------------------------------------------------
    // xc is the column being LOADED into the newest tap this cycle. The window
    // becomes valid once two columns are in, because output x needs columns
    // x-1, x and x+1 and x=0 takes its left tap from padding.
    reg [16:0] xc;
    reg [1:0]  prime;

    wire loading = (xc < {1'b0, row_width}) || (prime != 2'd0) || win_vld;

    // Per-channel base address for this cycle's column.
    wire [AW-1:0] col_addr [0:WIN_CH-1];
    genvar gc;
    generate
        for (gc = 0; gc < WIN_CH; gc = gc + 1) begin : g_addr
            assign col_addr[gc] = rd_base + (gc * ch_stride) + xc[AW-1:0];
        end
    endgenerate

    // Newest tap column, zero outside the row.
    wire in_range = (xc < {1'b0, row_width});

    integer c, r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win     <= {(WIN_CH*ROWS*3*8){1'b0}};
            win_vld <= 1'b0;
            win_x   <= 16'd0;
            xc      <= 17'd0;
            prime   <= 2'd0;
        end else if (sweep_start) begin
            // Clear the window so x=0's left tap is padding, then start
            // loading at column 0.
            win     <= {(WIN_CH*ROWS*3*8){1'b0}};
            win_vld <= 1'b0;
            win_x   <= 16'd0;
            xc      <= 17'd0;
            prime   <= 2'd2;
        end else begin
            // Shift the window left by one tap and load the new column.
            for (c = 0; c < WIN_CH; c = c + 1) begin
                for (r = 0; r < ROWS; r = r + 1) begin
                    // tap0 <= tap1, tap1 <= tap2, tap2 <= new column
                    win[((c*ROWS + r)*3 + 0)*8 +: 8] <=
                        win[((c*ROWS + r)*3 + 1)*8 +: 8];
                    win[((c*ROWS + r)*3 + 1)*8 +: 8] <=
                        win[((c*ROWS + r)*3 + 2)*8 +: 8];
                    win[((c*ROWS + r)*3 + 2)*8 +: 8] <=
                        in_range ? mem[rd_bank][r][col_addr[c]] : 8'h00;
                end
            end

            if (prime != 2'd0) begin
                prime <= prime - 2'd1;
                if (prime == 2'd1) begin
                    // Two columns are in; the window now covers output x=0.
                    win_vld <= 1'b1;
                    win_x   <= 16'd0;
                end
            end else if (win_vld) begin
                if (win_x + 16'd1 >= row_width) begin
                    win_vld <= 1'b0;          // swept the whole row
                end else begin
                    win_x <= win_x + 16'd1;
                end
            end

            if (loading)
                xc <= xc + 17'd1;
        end
    end

endmodule

`default_nettype wire
