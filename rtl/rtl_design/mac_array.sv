// ---------------------------------------------------------------------------
// mac_array -- K x M weight-stationary INT8 systolic array.
//
// Computes, for every cycle n that an aligned activation vector is presented:
//
//     Y[m][n] = sum over k of  W[m][k] * X[k][n]        for m = 0 .. M-1
//
// A streaming matrix-vector product: one input vector per cycle in, one output
// vector per cycle out, at fixed latency. That is the im2col GEMM inner loop
// for convolution, with M = output channels, K = input channels x 9, and
// N = output pixels.
//
// WHY THIS MAPPING (and not the reference project's):
// The ECE 410/510 anemia accelerator broadcast one 3x3 kernel across 1024
// SPATIAL tiles, which only works for single-channel convolution. Every
// convolution in KeywordSpottingCNN is multi-channel (32->64, 64->128,
// 128->128) and needs accumulation across input channels, which that structure
// has no path for -- their own report flags it in section 8.5. Tiling over
// output channel x input channel puts the reduction on the vertical psum
// chain. conv2, conv3 and conv4, 99.3% of all MACs, then tile 16x16 exactly.
//
// ---------------------------------------------------------------------------
// SYSTOLIC SKEW -- the part that has to be right
//
// A value hops one COLUMN per cycle but one ROW per PIPE cycles. Activation
// X[k][n] reaches PE(k,m) at cycle (inject_k + m); a partial sum started at
// PE(0,m) reaches row k at (start + PIPE*k). Those coincide for every k only
// if row k is injected PIPE*k cycles late. So:
//
//   * INPUT SKEW    : row k delayed by PIPE*k cycles       (8 bit)
//   * SWITCH SKEW   : row k's weight commit delayed by PIPE*k (1 bit)
//   * OUTPUT DESKEW : column m delayed by (M-1-m) cycles   (ACC_W)
//
//   LATENCY = PIPE*K + M - 1
//
// The switch skew matters as much as the data skew. Diagonals in flight when a
// new weight tile is committed must finish with the OLD weights, and they
// occupy row k at a time proportional to PIPE*k -- so the commit has to travel
// down the array at exactly the speed of the data it must not overtake.
//
// With all three in place the external interface is fully aligned: present a
// K-vector, get an M-vector. The skew is an internal detail, which is what
// makes this block easy to drive from AXI4-Stream and to check against Python.
//
// Skew register cost at K = M = 16, PIPE = 1:
//   input   8 * K(K-1)/2      =   960 flops
//   switch  1 * K(K-1)/2      =   120 flops
//   output  ACC_W * M(M-1)/2  = 3,840 flops
// The output deskew dominates. Draining columns sequentially instead would
// trade those 3,840 flops for M extra cycles per burst, which is nothing
// against bursts of 250 to 4,040 vectors -- the first thing to revisit if
// the tool reports the array as register-bound.
// ---------------------------------------------------------------------------

`default_nettype none

module mac_array #(
    parameter int K     = 16,   // reduction depth  (rows)    = in_channels * 9
    parameter int M     = 16,   // output channels  (columns)
    parameter int ACC_W = 32,
    parameter int PIPE  = 1     // psum pipeline stages per row (1 or 2)
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // Weight load: one INT8 per column, shifted down one row per enabled
    // cycle, into each PE's SHADOW register. Drive column m with W[m][K-1]
    // first and W[m][0] last; after K shifts PE(k,m) holds W[m][k].
    input  wire                      w_shift_en,
    input  wire [M*8-1:0]            w_top,      // column m at [m*8 +: 8]

    // Commit shadow -> active for the whole array. Internally skewed per row.
    input  wire                      w_switch,

    // Activation input: aligned K-vector, element k at [k*8 +: 8]
    input  wire                      a_vld,
    input  wire [K*8-1:0]            a_vec,

    // Result output: aligned M-vector, element m at [m*ACC_W +: ACC_W]
    output wire                      r_vld,
    output wire [M*ACC_W-1:0]        r_vec
);

    localparam int LATENCY = PIPE * K + M - 1;

    genvar k, m;

    // -----------------------------------------------------------------------
    // Input skew: row k delayed by PIPE*k cycles.
    // -----------------------------------------------------------------------
    wire signed [7:0] a_skewed [0:K-1];

    generate
        for (k = 0; k < K; k = k + 1) begin : g_iskew
            localparam int DEPTH = PIPE * k;
            if (DEPTH == 0) begin : g_passthrough
                assign a_skewed[0] = a_vec[0 +: 8];
            end else begin : g_delay
                reg signed [7:0] sr [0:DEPTH-1];
                integer i;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (i = 0; i < DEPTH; i = i + 1)
                            sr[i] <= 8'sd0;
                    end else begin
                        sr[0] <= a_vec[k*8 +: 8];
                        for (i = 1; i < DEPTH; i = i + 1)
                            sr[i] <= sr[i-1];
                    end
                end
                assign a_skewed[k] = sr[DEPTH-1];
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Weight-commit skew: same PIPE*k delay, so the commit travels down the
    // array at exactly the speed of the data it must not overtake.
    // -----------------------------------------------------------------------
    wire w_switch_row [0:K-1];

    generate
        for (k = 0; k < K; k = k + 1) begin : g_wskew
            localparam int DEPTH = PIPE * k;
            if (DEPTH == 0) begin : g_passthrough
                assign w_switch_row[0] = w_switch;
            end else begin : g_delay
                reg [DEPTH-1:0] sr;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n)      sr <= {DEPTH{1'b0}};
                    else if (DEPTH == 1) sr <= w_switch;
                    else             sr <= {sr[DEPTH-2:0], w_switch};
                end
                assign w_switch_row[k] = sr[DEPTH-1];
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // PE grid.
    //   a_h[k][m]  activation entering PE(k,m) from the left
    //   p_v[k][m]  partial sum entering PE(k,m) from above
    //   w_v[k][m]  weight entering PE(k,m) from above (shadow chain)
    // -----------------------------------------------------------------------
    wire signed [7:0]       a_h [0:K-1][0:M];
    wire signed [ACC_W-1:0] p_v [0:K][0:M-1];
    wire signed [7:0]       w_v [0:K][0:M-1];

    generate
        for (k = 0; k < K; k = k + 1) begin : g_row_in
            assign a_h[k][0] = a_skewed[k];
        end
        for (m = 0; m < M; m = m + 1) begin : g_col_in
            assign p_v[0][m] = {ACC_W{1'b0}};       // each diagonal starts at 0
            assign w_v[0][m] = w_top[m*8 +: 8];
        end

        for (k = 0; k < K; k = k + 1) begin : g_k
            for (m = 0; m < M; m = m + 1) begin : g_m
                pe_int8 #(.ACC_W(ACC_W), .PIPE(PIPE)) u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .w_shift_en (w_shift_en),
                    .w_in       (w_v[k][m]),
                    .w_out      (w_v[k+1][m]),
                    .w_switch   (w_switch_row[k]),
                    .a_in       (a_h[k][m]),
                    .a_out      (a_h[k][m+1]),
                    .psum_in    (p_v[k][m]),
                    .psum_out   (p_v[k+1][m])
                );
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Output deskew: column m's result emerges (M-1-m) cycles early.
    // -----------------------------------------------------------------------
    generate
        for (m = 0; m < M; m = m + 1) begin : g_odeskew
            localparam int DEPTH = M - 1 - m;
            if (DEPTH == 0) begin : g_passthrough
                assign r_vec[m*ACC_W +: ACC_W] = p_v[K][m];
            end else begin : g_delay
                reg signed [ACC_W-1:0] sr [0:DEPTH-1];
                integer i;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (i = 0; i < DEPTH; i = i + 1)
                            sr[i] <= {ACC_W{1'b0}};
                    end else begin
                        sr[0] <= p_v[K][m];
                        for (i = 1; i < DEPTH; i = i + 1)
                            sr[i] <= sr[i-1];
                    end
                end
                assign r_vec[m*ACC_W +: ACC_W] = sr[DEPTH-1];
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Valid tracking. The datapath never stalls and has fixed latency, so a
    // plain shift register is exact -- no need to carry valid through every PE.
    // -----------------------------------------------------------------------
    reg [LATENCY-1:0] vld_pipe;
    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_pipe <= {LATENCY{1'b0}};
        end else begin
            vld_pipe[0] <= a_vld;
            for (j = 1; j < LATENCY; j = j + 1)
                vld_pipe[j] <= vld_pipe[j-1];
        end
    end

    assign r_vld = vld_pipe[LATENCY-1];

endmodule

`default_nettype wire
