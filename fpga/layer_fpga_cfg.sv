// ---------------------------------------------------------------------------
// layer_fpga_cfg -- layer_top wired to the parameters the FPGA build uses.
//
// WHY THIS EXISTS RATHER THAN set_parameter IN THE .qsf
// Passing top-level parameters through the project file is one more thing
// that can silently not happen. If MAXW stays at its 128 default, rowacc is
// 128 x 448 flops behind two 128:1 muxes, synthesis grinds for tens of
// minutes, and the failure looks identical to a memory that would not infer.
// Two candidate causes, one symptom, no way to tell them apart without
// another compile.
//
// A wrapper removes the question. The parameters are in the source, the
// elaborated design is whatever this file says, and no tool setting can
// disagree with it.
//
// THE PARAMETERS, AND WHY EACH ONE
//   MAXW       32    rowacc is read ASYNCHRONOUSLY at two independent
//                    indices -- oc to accumulate, flush_x to flush. No block
//                    RAM offers that, so it stays in flops and MAXW sizes it
//                    directly. conv4 is W=25, so 32 covers it at 32 x 448 =
//                    14,336 flops. The 128 default would be 57,344 plus two
//                    128:1 muxes 448 bits wide.
//   MAX_KTILES 72    conv4 has 1152 taps = 72 k-tiles, the deepest reduction
//                    in the model. Holding fewer would force a mid-row refill
//                    that has no flow control.
//   BAND_DEPTH 3232  max channels x width across conv2/3/4.
//
// conv4 is the deliberate target. rtl/sim/layer_cycles.py identifies it as
// the weak layer -- W=25 is below 2K=32, so its weight load cannot fully hide
// inside the sweep -- which makes it the layer worth measuring on hardware
// rather than the one most likely to look good.
// ---------------------------------------------------------------------------

`default_nettype none

module layer_fpga_cfg (
    input  wire         clk,
    input  wire         rst_n,

    // ---- Activation band fill --------------------------------------------
    input  wire         bnd_wr_en,
    input  wire         bnd_wr_bank,
    input  wire [1:0]   bnd_wr_row,
    input  wire [11:0]  bnd_wr_addr,      // $clog2(3232) = 12
    input  wire [7:0]   bnd_wr_data,

    // ---- Weight memory fill ----------------------------------------------
    input  wire         wm_wr_en,
    input  wire [14:0]  wm_wr_addr,       // $clog2(72*16*16) = 15
    input  wire [7:0]   wm_wr_data,

    // ---- Requantisation config (BatchNorm folded in by the host) ---------
    input  wire         cfg_we,
    input  wire [3:0]   cfg_ch,           // $clog2(16) = 4
    input  wire signed [27:0] cfg_bias,   // ACC_W = 28
    input  wire [15:0]  cfg_mult,         // MULT_W = 16
    input  wire [5:0]   cfg_shift,

    // ---- Layer config -----------------------------------------------------
    input  wire         cfg_relu_en,
    input  wire         cfg_pool_en,
    input  wire [15:0]  cfg_width,
    input  wire [7:0]   cfg_ktiles,
    input  wire         cfg_rd_bank,
    input  wire [15:0]  cfg_ch_stride,

    // ---- Control ----------------------------------------------------------
    input  wire         layer_start,
    input  wire         row_go,
    output wire         busy,

    // ---- INT8 output ------------------------------------------------------
    output wire         out_vld,
    output wire [127:0] out_vec           // M*8 = 128
);

    layer_top #(
        .K          (16),
        .M          (16),
        .ACC_W      (28),
        .MULT_W     (16),
        .PIPE       (1),
        .MAXW       (32),
        .MAX_KTILES (72),
        .BAND_DEPTH (3232),
        .WIN_CH     (3)
    ) u_layer (
        .clk           (clk),
        .rst_n         (rst_n),
        .bnd_wr_en     (bnd_wr_en),
        .bnd_wr_bank   (bnd_wr_bank),
        .bnd_wr_row    (bnd_wr_row),
        .bnd_wr_addr   (bnd_wr_addr),
        .bnd_wr_data   (bnd_wr_data),
        .wm_wr_en      (wm_wr_en),
        .wm_wr_addr    (wm_wr_addr),
        .wm_wr_data    (wm_wr_data),
        .cfg_we        (cfg_we),
        .cfg_ch        (cfg_ch),
        .cfg_bias      (cfg_bias),
        .cfg_mult      (cfg_mult),
        .cfg_shift     (cfg_shift),
        .cfg_relu_en   (cfg_relu_en),
        .cfg_pool_en   (cfg_pool_en),
        .cfg_width     (cfg_width),
        .cfg_ktiles    (cfg_ktiles),
        .cfg_rd_bank   (cfg_rd_bank),
        .cfg_ch_stride (cfg_ch_stride),
        .layer_start   (layer_start),
        .row_go        (row_go),
        .busy          (busy),
        .out_vld       (out_vld),
        .out_vec       (out_vec)
    );

endmodule

`default_nettype wire
