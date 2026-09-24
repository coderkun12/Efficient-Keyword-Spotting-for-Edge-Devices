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
// MAPPING: the storage is nine flat arrays with SYNCHRONOUS reads -- one per
// (channel tap, row), with the bank folded into the address MSB. That shape
// maps directly onto FPGA block RAM and onto compiled SRAM macros, and the
// reasoning behind it is at the declaration below.
//
// The synchronous read costs one cycle on the load path, which the window
// shift absorbs by priming for three cycles instead of two. External timing
// is unchanged: win_vld, win and win_x still arrive together, which is why
// layer_top needed no modification when this changed.
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

    // ---- Storage lives below, after col_addr ------------------------------
    // The arrays are declared next to the read that uses them, because the
    // read address has to exist first. See "Storage + synchronous read".

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
            // The sum is 32 bits wide because gc*ch_stride promotes to
            // integer width, and it is narrowed to AW deliberately.
            //
            // The bound that makes that safe: the window reads channels
            // ch_base .. ch_base+2, so the highest address any tap forms is
            // (ch_base+3)*width - 1, and ch_base+3 never exceeds the channel
            // count -- so it stays below channels*width, which is what DEPTH
            // is sized to. An EXPLICIT cast says so. Left implicit the tool
            // warns, and one more routine truncation warning is exactly where
            // a real overflow would hide later.
            assign col_addr[gc] = AW'(rd_base + (gc * ch_stride) + xc[AW-1:0]);
        end
    endgenerate

    // Newest tap column, zero outside the row.
    wire in_range = (xc < {1'b0, row_width});

    // ---- Storage + synchronous read ---------------------------------------
    //
    // WHY NINE FLAT ARRAYS AND NOT ONE MULTI-DIMENSIONAL ONE
    // Each channel tap reads a different address every cycle -- rd_base +
    // c*ch_stride + xc for c = 0..WIN_CH-1. Those are ch_stride apart, so
    // they cannot be fetched as one wide word, and no FPGA block RAM has
    // three read ports. The array must therefore be replicated per tap.
    //
    // The shape of the replication decides whether it works at all. Writing
    // it as one array indexed [tap][bank][row][addr] is the obvious form and
    // is useless: RAM inference matches a FLAT array under a SINGLE address,
    // and a 4-D array with three variable indices does not match. Quartus
    // silently builds it out of logic instead -- 465 Kb of it, three times
    // every register on this device -- and synthesis grinds for tens of
    // minutes before failing to fit.
    //
    // So each (tap, row) pair gets its own flat array, with the bank folded
    // into the address MSB. Nine arrays, each one write port with an enable
    // and one read port with a registered output: textbook simple dual port,
    // which is exactly what M9K implements and what an SRAM macro compiles
    // to. Addressing by concatenation rather than rd_bank*DEPTH keeps an
    // adder out of the address path, at the cost of rounding each array up
    // to a power of two.
    //
    // Cost: 9 x 2^(AW+1) bytes = 590 Kb, about 9% of the EP4CGX150's M9K.
    reg [7:0] mem_q [0:WIN_CH-1][0:ROWS-1];

    genvar gm, gr;
    generate
        for (gm = 0; gm < WIN_CH; gm = gm + 1) begin : g_tap
            for (gr = 0; gr < ROWS; gr = gr + 1) begin : g_row
                reg [7:0] m [0:(2<<AW)-1];

                // Deliberately no reset on the output register: an async
                // clear on a block RAM's output stops the tool packing it
                // into the memory. The priming sequence below guarantees
                // real data is present before anything reads it.
                always @(posedge clk) begin
                    if (wr_en && (wr_row == gr[$clog2(ROWS)-1:0]))
                        m[{wr_bank, wr_addr}] <= wr_data;
                    mem_q[gm][gr] <= m[{rd_bank, col_addr[gm]}];
                end
            end
        end
    endgenerate

    // in_range travels with the data it qualifies, so it needs the same
    // one-cycle delay. It is a scalar in logic, not part of the RAM, so a
    // reset here costs nothing and keeps it out of X.
    reg in_range_q;

    integer c, r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win        <= {(WIN_CH*ROWS*3*8){1'b0}};
            win_vld    <= 1'b0;
            win_x      <= 16'd0;
            xc         <= 17'd0;
            prime      <= 2'd0;
            in_range_q <= 1'b0;
        end else if (sweep_start) begin
            // Clear the window so x=0's left tap is padding, then start
            // loading at column 0.
            //
            // PRIME IS 3, NOT 2. Two columns must be in the window before
            // x=0 is valid, and with a registered read each column now takes
            // an extra cycle to arrive. The third count covers that.
            win        <= {(WIN_CH*ROWS*3*8){1'b0}};
            win_vld    <= 1'b0;
            win_x      <= 16'd0;
            xc         <= 17'd0;
            prime      <= 2'd3;
            in_range_q <= 1'b0;
        end else begin
            in_range_q <= in_range;
            // Shift the window left by one tap and load the new column.
            //
            // Held still on the FIRST priming cycle: the read register has
            // not produced its first column yet, so it still holds whatever
            // the previous sweep left behind. Shifting that in would push
            // stale bytes toward tap0 -- which is precisely the position x=0
            // reads as left padding, so the corruption would land on the
            // first output column and nowhere else.
            if (prime != 2'd3) begin
                for (c = 0; c < WIN_CH; c = c + 1) begin
                    for (r = 0; r < ROWS; r = r + 1) begin
                        // tap0 <= tap1, tap1 <= tap2, tap2 <= new column
                        win[((c*ROWS + r)*3 + 0)*8 +: 8] <=
                            win[((c*ROWS + r)*3 + 1)*8 +: 8];
                        win[((c*ROWS + r)*3 + 1)*8 +: 8] <=
                            win[((c*ROWS + r)*3 + 2)*8 +: 8];
                        win[((c*ROWS + r)*3 + 2)*8 +: 8] <=
                            in_range_q ? mem_q[c][r] : 8'h00;
                    end
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
