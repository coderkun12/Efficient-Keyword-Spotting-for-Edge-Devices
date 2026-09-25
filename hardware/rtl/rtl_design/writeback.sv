// ---------------------------------------------------------------------------
// writeback -- fused requantise + BatchNorm + ReLU + 2x2 max-pool.
//
// ---------------------------------------------------------------------------
// WHY THIS MODULE IS THE POINT OF THE WHOLE PROJECT
//
// Profiling on two independent machines (Profiling/op_breakdown.txt) found:
//
//     convolution   55.4%  of host runtime,  99.3% of the MACs
//     max-pool      38.1%  of host runtime,  ~0%   of the MACs
//     BatchNorm      2.9%
//     ReLU           1.4%
//
// Max-pool is a third of the runtime and essentially none of the arithmetic.
// It is pure data movement, and a MAC-count analysis -- the standard way
// accelerator targets get picked, and what the reference ECE 410/510 design
// did -- misses it completely. An accelerator that takes only convolution is
// capped by Amdahl at 2.24x no matter how large the array. Taking convolution,
// pooling, BatchNorm and ReLU together raises the ceiling to 44.3x.
//
// This module is what makes that difference, and it is cheap because the
// operations are already sitting in the array's output path.
//
// ---------------------------------------------------------------------------
// BATCHNORM IS FREE
//
// BatchNorm at inference is an affine map, y = gamma*(x-mu)/sigma + beta, with
// all four constants known once training ends. Folded into the convolution it
// becomes a per-output-channel scale and offset -- exactly the shape of the
// requantisation that INT8 inference needs anyway:
//
//     out = clamp( ((acc + bias) * mult + round) >>> shift )
//
// So BatchNorm costs no hardware at all beyond the requantiser we already
// needed. It is folded into `bias` and `mult` by the host at compile time.
//
// ---------------------------------------------------------------------------
// ORDERING: REQUANTISE BEFORE POOLING
//
// Requantise, ReLU and max are all monotonically increasing, so they commute:
//     max(relu(rq(a)), relu(rq(b))) == relu(rq(max(a, b)))
// Either order is numerically identical. Pooling first would let one
// requantiser serve four outputs, but the pooling line buffer would then hold
// INT32 instead of INT8 and grow 4x. Requantising first keeps the line buffer
// at MAXW*M bytes, and the requantisers are then fully utilised rather than
// idle 75% of the time.
//
// The area trade is worth revisiting if synthesis says the M requantiser lanes
// dominate: pooling first plus a 4:1 time-multiplexed requantiser is the other
// corner, costing ~2.4 KB more SRAM and saving ~75% of the multiplier area.
//
// ---------------------------------------------------------------------------
// POOLING GEOMETRY
//
// PyTorch's MaxPool2d(2) floors: a 101-wide row gives 50 outputs and the last
// column is dropped. Same for rows. That matches the model exactly:
//     conv2  40x101 -> 20x50     conv3  20x50 -> 10x25     conv4  10x25 -> 5x12
// Getting this wrong shifts every downstream feature by one pixel, which shows
// up as a quiet accuracy loss rather than an obvious failure.
// ---------------------------------------------------------------------------

`default_nettype none

module writeback #(
    parameter int M      = 16,   // channels handled in parallel
    parameter int ACC_W  = 32,
    parameter int MULT_W = 16,   // requantisation multiplier width
    parameter int MAXW   = 64    // max pooled columns per row (>= ceil(W/2))
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- Per-channel configuration (host writes once per tile) -----------
    input  wire                     cfg_we,
    input  wire [$clog2(M)-1:0]     cfg_ch,
    input  wire signed [ACC_W-1:0]  cfg_bias,
    input  wire [MULT_W-1:0]        cfg_mult,
    input  wire [5:0]               cfg_shift,

    // ---- Layer configuration --------------------------------------------
    input  wire                     cfg_relu_en,
    input  wire                     cfg_pool_en,
    input  wire [15:0]              cfg_row_width,   // pre-pool columns per row

    input  wire                     start,           // reset the x/y counters

    // ---- Accumulator input, one vector per cycle -------------------------
    input  wire                     acc_vld,
    input  wire [M*ACC_W-1:0]       acc_vec,

    // ---- INT8 output -----------------------------------------------------
    output reg                      out_vld,
    output reg  [M*8-1:0]           out_vec
);

    localparam int PROD_W = ACC_W + 1 + MULT_W;
    localparam int PW     = (MAXW > 1) ? $clog2(MAXW) : 1;

    // ---- Configuration storage -------------------------------------------
    reg signed [ACC_W-1:0] bias_r  [0:M-1];
    reg [MULT_W-1:0]       mult_r  [0:M-1];
    reg [5:0]              shift_r [0:M-1];

    // The loop counter is declared INSIDE the loop, not at module scope.
    //
    // As a module-scope `integer ci` this inferred a latch, and the tool was
    // right to say so: the only loop over ci sits in the reset branch, so on
    // the cfg_we path nothing assigns it and it holds its previous value.
    // That is the definition of state, and Quartus built a register for a
    // variable that is meant to vanish at elaboration.
    //
    // Note the contrast with m1 below, which is also module-scope but loops
    // in BOTH branches, so it is never live across one -- which is exactly
    // why that one drew no warning. The rule is not "module-scope integers
    // are bad", it is "a variable live across a branch is state".
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int ci = 0; ci < M; ci = ci + 1) begin
                bias_r[ci]  <= {ACC_W{1'b0}};
                mult_r[ci]  <= {MULT_W{1'b0}};
                shift_r[ci] <= 6'd0;
            end
        end else if (cfg_we) begin
            bias_r[cfg_ch]  <= cfg_bias;
            mult_r[cfg_ch]  <= cfg_mult;
            shift_r[cfg_ch] <= cfg_shift;
        end
    end

    // ---- Stage 1: bias add and multiply ----------------------------------
    // Every operand is made explicitly signed before use. A concatenation in
    // Verilog is ALWAYS unsigned, so folding the sign extension into a concat
    // here would silently zero-extend a negative bias and flip the sign of the
    // result -- which then saturates to +127 instead of 0 after ReLU.
    reg                      s1_vld;
    reg signed [PROD_W-1:0]  s1_prod [0:M-1];
    reg [5:0]                s1_shift [0:M-1];

    integer m1;
    reg signed [ACC_W:0]      biased;
    reg signed [MULT_W:0]     mult_s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_vld <= 1'b0;
            for (m1 = 0; m1 < M; m1 = m1 + 1) begin
                s1_prod[m1]  <= {PROD_W{1'b0}};
                s1_shift[m1] <= 6'd0;
            end
        end else begin
            s1_vld <= acc_vld;
            for (m1 = 0; m1 < M; m1 = m1 + 1) begin
                biased       = $signed(acc_vec[m1*ACC_W +: ACC_W]) + bias_r[m1];
                mult_s       = $signed({1'b0, mult_r[m1]});
                s1_prod[m1]  <= biased * mult_s;
                s1_shift[m1] <= shift_r[m1];
            end
        end
    end

    // ---- Stage 2: round, shift, clamp (ReLU folded into the clamp) -------
    reg           s2_vld;
    reg [M*8-1:0] s2_vec;

    integer m2;
    reg signed [PROD_W-1:0] rnd, shifted;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_vld <= 1'b0;
            s2_vec <= {(M*8){1'b0}};
        end else begin
            s2_vld <= s1_vld;
            for (m2 = 0; m2 < M; m2 = m2 + 1) begin
                // Round half away from zero-ish: add half an LSB then shift.
                rnd = (s1_shift[m2] == 6'd0)
                        ? {PROD_W{1'b0}}
                        : ({{(PROD_W-1){1'b0}}, 1'b1} <<< (s1_shift[m2] - 6'd1));
                shifted = (s1_prod[m2] + rnd) >>> s1_shift[m2];

                if (cfg_relu_en && shifted < 0)
                    s2_vec[m2*8 +: 8] <= 8'sd0;
                else if (shifted > $signed(127))
                    s2_vec[m2*8 +: 8] <= 8'sd127;
                else if (shifted < $signed(-128))
                    // 8'sh80, not -8'sd128. The latter does not fit: 8-bit
                    // signed spans -128..127, so 8'sd128 overflows to -128
                    // and negating THAT overflows again back to -128. It
                    // reaches the right value by two wrongs cancelling, and
                    // the tool flags it as a constant overflow. 8'sh80 is
                    // the bit pattern for -128 stated directly.
                    s2_vec[m2*8 +: 8] <= 8'sh80;
                else
                    s2_vec[m2*8 +: 8] <= shifted[7:0];
            end
        end
    end

    // ---- Position tracking, pipelined alongside the datapath -------------
    // x and y are the PRE-pool coordinates of the vector entering stage 1.
    reg [15:0] x_cnt, y_cnt;
    reg [15:0] s1_x, s1_y, s2_x, s2_y;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_cnt <= 16'd0;
            y_cnt <= 16'd0;
        end else if (start) begin
            x_cnt <= 16'd0;
            y_cnt <= 16'd0;
        end else if (acc_vld) begin
            if (x_cnt + 16'd1 >= cfg_row_width) begin
                x_cnt <= 16'd0;
                y_cnt <= y_cnt + 16'd1;
            end else begin
                x_cnt <= x_cnt + 16'd1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_x <= 16'd0; s1_y <= 16'd0;
            s2_x <= 16'd0; s2_y <= 16'd0;
        end else begin
            s1_x <= x_cnt; s1_y <= y_cnt;
            s2_x <= s1_x;  s2_y <= s1_y;
        end
    end

    // ---- Stage 3: horizontal max over column pairs -----------------------
    // Emits on odd x, dropping a trailing odd column the way floor() does.
    reg [M*8-1:0] hcarry;
    reg           h_vld;
    reg [M*8-1:0] h_vec;
    reg [15:0]    h_y;
    reg [15:0]    h_x;      // pooled column index, = s2_x >> 1

    function automatic signed [7:0] smax(input signed [7:0] a, input signed [7:0] b);
        smax = (a > b) ? a : b;
    endfunction

    integer m3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hcarry <= {(M*8){1'b0}};
            h_vld  <= 1'b0;
            h_vec  <= {(M*8){1'b0}};
            h_y    <= 16'd0;
            h_x    <= 16'd0;
        end else begin
            h_vld <= 1'b0;
            if (s2_vld) begin
                if (!cfg_pool_en) begin
                    h_vec <= s2_vec;
                    h_vld <= 1'b1;
                    h_y   <= s2_y;
                    h_x   <= s2_x;
                end else if (s2_x[0] == 1'b0) begin
                    hcarry <= s2_vec;                 // even column: hold
                end else begin
                    for (m3 = 0; m3 < M; m3 = m3 + 1)
                        h_vec[m3*8 +: 8] <= smax($signed(hcarry[m3*8 +: 8]),
                                                 $signed(s2_vec[m3*8 +: 8]));
                    h_vld <= 1'b1;                    // odd column: emit
                    h_y   <= s2_y;
                    h_x   <= {1'b0, s2_x[15:1]};      // pooled column = x >> 1
                end
            end
        end
    end

    // ---- Stage 4: vertical max through a line buffer ---------------------
    // Even source rows are stored, odd source rows emit max(stored, current).
    // The pooled column index rides the pipeline as h_x, so no row-boundary
    // detection is needed. A trailing odd row is simply stored and never
    // emitted, and a trailing even column never reaches this stage -- which is
    // exactly what floor() does, and what MaxPool2d(2) does.
    reg [M*8-1:0] linebuf [0:MAXW-1];

    integer m4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_vld <= 1'b0;
            out_vec <= {(M*8){1'b0}};
        end else begin
            out_vld <= 1'b0;
            if (h_vld) begin
                if (!cfg_pool_en) begin
                    out_vec <= h_vec;
                    out_vld <= 1'b1;
                end else if (h_y[0] == 1'b0) begin
                    linebuf[h_x[PW-1:0]] <= h_vec;         // even row: store
                end else begin
                    for (m4 = 0; m4 < M; m4 = m4 + 1)
                        out_vec[m4*8 +: 8] <= smax(
                            $signed(linebuf[h_x[PW-1:0]][m4*8 +: 8]),
                            $signed(h_vec[m4*8 +: 8]));
                    out_vld <= 1'b1;                       // odd row: emit
                end
            end
        end
    end

endmodule

`default_nettype wire
