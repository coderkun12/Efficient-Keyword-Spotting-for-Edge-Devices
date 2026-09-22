// ---------------------------------------------------------------------------
// pe_int8 -- one processing element of the weight-stationary INT8 array.
//
// An INT8 x INT8 multiply added into an INT32 partial sum, with registered
// outputs and deliberately NO per-PE state machine. The reference BF16 design
// (ECE 410/510 anemia accelerator) ran a 4-state FSM per tile and spent 2 of
// every 4 cycles in LOAD and DONE_ST, so it achieved only 50% pipeline
// utilisation. This array never stalls: one MAC per PE per cycle.
//
// DATAFLOW (weight-stationary, TPU-style):
//   * w_active is loaded once per weight tile and then held.
//   * Activations flow LEFT -> RIGHT   (a_in  -> a_out), always 1 cycle/hop.
//   * Partial sums flow TOP  -> BOTTOM (psum_in -> psum_out), PIPE cycles/hop.
//
// ---------------------------------------------------------------------------
// M3 ADDITION 1 -- PIPE: pipeline depth of the psum path
//
//   PIPE = 1  multiply and accumulate share one cycle (the M2 design).
//             ACCELERATOR_PLAN.md section 5 estimates 1.0-1.5 ns for this path
//             on SAED14nm against a 2.0 ns budget at 500 MHz.
//   PIPE = 2  multiply registers first, then accumulate. Halves the
//             combinational path at the cost of one extra cycle per ROW.
//
// This is the knob synthesis decides. It is a parameter rather than an edit
// because the array's skew depends on it: a value hops one column per cycle
// but one row per PIPE cycles, so mac_array.sv skews row k by PIPE*k. Get that
// coupling wrong and the array silently computes the wrong dot product, which
// is exactly why test_array checks latency == PIPE*K + M - 1 explicitly.
//
// ---------------------------------------------------------------------------
// M3 ADDITION 2 -- shadow weight register (double buffering)
//
// The shift chain now loads w_shadow, which is NOT the weight being multiplied.
// w_active takes the shadow's value only when w_switch pulses. That lets the
// next tile's weights shift in while the current tile is still streaming,
// removing the K-cycle load bubble between tiles.
//
// The switch MUST be skewed to match the data wavefront. Diagonal n occupies
// row k at a time proportional to PIPE*k, so if every row switched at once,
// diagonals still in flight would be finished with the wrong tile's weights.
// mac_array.sv delays w_switch by PIPE*k for row k, the same skew it applies
// to activations. rtl/sim/tile_schedule.py measures what this recovers:
// 93.9% -> 99.5% array utilisation, 1.550 ms -> 1.462 ms per inference.
// ---------------------------------------------------------------------------

`default_nettype none

module pe_int8 #(
    parameter int ACC_W = 32,
    parameter int PIPE  = 1     // 1 = fused multiply-add, 2 = registered product
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // Weight shift chain (vertical) -- loads the SHADOW register
    input  wire                      w_shift_en,
    input  wire signed [7:0]         w_in,
    output wire signed [7:0]         w_out,

    // Commit shadow -> active. Skewed per row by the array.
    input  wire                      w_switch,

    // Activation, flows left -> right, one cycle per hop
    input  wire signed [7:0]         a_in,
    output reg  signed [7:0]         a_out,

    // Partial sum, flows top -> bottom, PIPE cycles per hop
    input  wire signed [ACC_W-1:0]   psum_in,
    output wire signed [ACC_W-1:0]   psum_out
);

    reg signed [7:0] w_shadow;
    reg signed [7:0] w_active;

    // The chain passes the shadow value down, so a whole column loads in K
    // cycles without disturbing the weights currently in use.
    assign w_out = w_shadow;

    // Full-width product: 8b x 8b signed needs 16 bits, sign-extended into the
    // accumulator. INT8 operands with an INT32 accumulator cannot overflow for
    // any K we tile (worst case conv4: 1152 taps x 127 x 128 = 1.9e7).
    wire signed [15:0]      product   = w_active * a_in;
    wire signed [ACC_W-1:0] product_x = {{(ACC_W-16){product[15]}}, product};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_shadow <= 8'sd0;
            w_active <= 8'sd0;
            a_out    <= 8'sd0;
        end else begin
            if (w_shift_en) w_shadow <= w_in;
            if (w_switch)   w_active <= w_shadow;
            a_out <= a_in;
        end
    end

    generate
        if (PIPE == 1) begin : g_pipe1
            // One stage: multiply and add in the same cycle.
            reg signed [ACC_W-1:0] psum_q;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) psum_q <= {ACC_W{1'b0}};
                else        psum_q <= psum_in + product_x;
            end
            assign psum_out = psum_q;
        end else begin : g_pipe2
            // Two stages: register the product, then accumulate. The incoming
            // partial sum must be delayed to match, or it would be added to a
            // product from the wrong cycle.
            reg signed [ACC_W-1:0] prod_q;
            reg signed [ACC_W-1:0] psum_d;
            reg signed [ACC_W-1:0] psum_q;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    prod_q <= {ACC_W{1'b0}};
                    psum_d <= {ACC_W{1'b0}};
                    psum_q <= {ACC_W{1'b0}};
                end else begin
                    prod_q <= product_x;
                    psum_d <= psum_in;
                    psum_q <= psum_d + prod_q;
                end
            end
            assign psum_out = psum_q;
        end
    endgenerate

endmodule

`default_nettype wire
