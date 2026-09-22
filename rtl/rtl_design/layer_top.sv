// ---------------------------------------------------------------------------
// layer_top -- the integrated accelerator: scratchpad, array, accumulation,
//              and fused write-back under one sequencer.
//
//   band_sram   3-row activation band, emits a 3x3 im2col window per column
//   mac_array   K x M weight-stationary INT8 systolic array
//   row accum   partial sums across k-tiles, because K is usually > 16
//   writeback   requantise (BatchNorm folded in) + ReLU + 2x2 max-pool
//
// ---------------------------------------------------------------------------
// WHY A ROW ACCUMULATOR EXISTS
//
// The array reduces over only K = 16 taps at a time, but a real layer needs
// far more: conv2 has 288 taps, conv3 576, conv4 1152. Those are 18, 36 and 72
// k-tiles, and every k-tile contributes a partial sum to the SAME output pixel.
// Something has to hold those partials while the tiles are walked.
//
// Holding them for a whole feature map would need N x M x 32 bits, which for
// conv2 is 4040 x 16 x 4 = 258 KB -- the same trap band_sram exists to avoid.
// Accumulating over ONE OUTPUT ROW instead needs W x M x 4 = 6,464 B, and the
// k-tile loop sits inside the row loop:
//
//     for each output row y:
//         for each k-tile:
//             load that tile's weights (overlapped with the previous sweep)
//             sweep x = 0 .. W-1, accumulating into rowacc[x]
//         stream rowacc into the write-back path
//
// Reordering the loops this way costs nothing: the total sweep count is
// unchanged at tiles x N, which is what rtl/sim/tile_schedule.py already
// models. It only changes which partial sums have to be resident.
//
// ---------------------------------------------------------------------------
// FLAT TAP INDEXING
//
// im2col tap t maps to (channel, row, col) as c = t/9, r = (t%9)/3, s = t%3.
// A k-tile covers taps [16*kt, 16*kt+16), which straddles channel boundaries
// because 9 does not divide 16. The tile therefore spans at most
// ceil((16+8)/9) = 3 channels, which is exactly band_sram's WIN_CH, and the
// array's K-vector is a 16-byte slice of the 27-byte window at a byte offset
// of 16*kt - 9*floor(16*kt/9). That offset is never more than 8, so the slice
// always fits.
// ---------------------------------------------------------------------------

`default_nettype none

module layer_top #(
    parameter int K          = 16,
    parameter int M          = 16,
    // 26 bits is the HARD bound for the array: the deepest reduction is
    // conv4's 1152 taps, and 1152 x 127 x 127 = 18,580,608 needs 26 bits
    // signed. 28 leaves 4x headroom for the per-channel bias that writeback
    // adds through the same width. The 4 bits saved against a reflexive 32
    // remove 12.5% of every accumulator register in the design -- the PE
    // partial sums, the output deskew and the row accumulator -- and shorten
    // the PE's carry chain, which is free timing margin if 500 MHz is tight.
    parameter int ACC_W      = 28,
    parameter int MULT_W     = 16,
    parameter int PIPE       = 1,
    parameter int MAXW       = 128,   // max pre-pool output columns per row
    // Sized for the WORST layer, not the first one. conv4 has 1152 taps =
    // 72 k-tiles, and all of them are swept once per output row, so holding
    // fewer would force the host to refill mid-row -- for which there is no
    // flow control, meaning the sequencer would sweep stale weights. The cost
    // is 72 x K x M = 18,432 B, loaded once per m-tile and reused across all
    // H rows of that tile.
    parameter int MAX_KTILES = 72,    // k-tiles of weights held on chip
    parameter int BAND_DEPTH = 3232,  // bytes per band row bank
    parameter int WIN_CH     = 3
) (
    input  wire                        clk,
    input  wire                        rst_n,

    // ---- Activation band fill (writes the bank the array is not reading) --
    input  wire                        bnd_wr_en,
    input  wire                        bnd_wr_bank,
    input  wire [1:0]                  bnd_wr_row,
    input  wire [$clog2(BAND_DEPTH)-1:0] bnd_wr_addr,
    input  wire [7:0]                  bnd_wr_data,

    // ---- Weight memory fill: index kt*K*M + m*K + k ----------------------
    input  wire                        wm_wr_en,
    input  wire [$clog2(MAX_KTILES*K*M)-1:0] wm_wr_addr,
    input  wire [7:0]                  wm_wr_data,

    // ---- Requantisation configuration (BatchNorm folded in by the host) --
    input  wire                        cfg_we,
    input  wire [$clog2(M)-1:0]        cfg_ch,
    input  wire signed [ACC_W-1:0]     cfg_bias,
    input  wire [MULT_W-1:0]           cfg_mult,
    input  wire [5:0]                  cfg_shift,

    // ---- Layer configuration ---------------------------------------------
    input  wire                        cfg_relu_en,
    input  wire                        cfg_pool_en,
    input  wire [15:0]                 cfg_width,      // output columns W
    input  wire [7:0]                  cfg_ktiles,     // k-tiles per output
    input  wire                        cfg_rd_bank,    // band bank to read
    input  wire [15:0]                 cfg_ch_stride,  // input row width

    // ---- Control ----------------------------------------------------------
    input  wire                        layer_start,    // reset write-back x/y
    input  wire                        row_go,         // process one output row
    output wire                        busy,

    // ---- INT8 output ------------------------------------------------------
    output wire                        out_vld,
    output wire [M*8-1:0]              out_vec
);

    localparam int LATENCY = PIPE * K + M - 1;
    localparam int WMEM_AW = $clog2(MAX_KTILES*K*M);
    localparam int XW      = $clog2(MAXW);

    // -----------------------------------------------------------------------
    // Weight memory, stored ONE WORD PER (k-tile, tap) rather than per byte.
    //
    // A shift cycle needs W[m][k] for all M columns at once. Storing bytes
    // would make that M independent reads of an 18,432-entry array -- sixteen
    // 18,432:1 muxes, about fifteen levels of logic each, plus sixteen address
    // computations containing a multiply. That mux tree, not the systolic
    // array, would be the critical path, so synthesis would characterise the
    // wrong thing entirely. It also cannot map to an SRAM macro, and 18 KB of
    // weights has to become one.
    //
    // One M*8-bit word per (tile, tap) turns it into a single 1,152:1 read at
    // one address. The host still writes bytes; the lane is decoded here.
    // -----------------------------------------------------------------------
    localparam int WWORDS = MAX_KTILES * K;
    localparam int WW_AW  = $clog2(WWORDS);

    reg [M*8-1:0] wmem [0:WWORDS-1];

    // Host byte address is kt*K*M + m*K + k; split it into word and lane.
    wire [31:0] wm_kt   = wm_wr_addr / (K*M);
    wire [31:0] wm_rem  = wm_wr_addr % (K*M);
    wire [31:0] wm_lane = wm_rem / K;
    wire [31:0] wm_k    = wm_rem % K;
    wire [WW_AW-1:0] wm_word = WW_AW'(wm_kt * K + wm_k);

    always @(posedge clk) begin
        if (wm_wr_en)
            wmem[wm_word][wm_lane[$clog2(M)-1:0]*8 +: 8] <= wm_wr_data;
    end

    // -----------------------------------------------------------------------
    // Sequencer
    // -----------------------------------------------------------------------
    localparam [2:0] S_IDLE  = 3'd0,
                     S_WWAIT = 3'd1,   // waiting for a shadow tile to be ready
                     S_SWEEP = 3'd2,
                     S_DRAIN = 3'd3,
                     S_FLUSH = 3'd4;

    reg [2:0]  state;
    reg [7:0]  ckt;                // k-tile currently COMPUTING (active weights)
    reg [15:0] drain_cnt;
    reg [15:0] flush_x;

    assign busy = (state != S_IDLE);

    // Byte offset of the COMPUTING tile's first tap within the 3-channel
    // window, and the channel group that window must cover.
    wire [15:0] tap_base  = 16'(ckt) * 16'(K);
    wire [15:0] ch_base   = tap_base / 16'd9;
    wire [15:0] tap_off   = tap_base - ch_base * 16'd9;

    // -----------------------------------------------------------------------
    // band_sram
    // -----------------------------------------------------------------------
    reg  sweep_start;
    wire win_vld;
    wire [WIN_CH*3*3*8-1:0] win;
    wire [15:0] win_x;

    band_sram #(.DEPTH(BAND_DEPTH), .ROWS(3), .WIN_CH(WIN_CH)) u_band (
        .clk         (clk),
        .rst_n       (rst_n),
        .wr_en       (bnd_wr_en),
        .wr_bank     (bnd_wr_bank),
        .wr_row      (bnd_wr_row),
        .wr_addr     (bnd_wr_addr),
        .wr_data     (bnd_wr_data),
        .rd_bank     (cfg_rd_bank),
        .sweep_start (sweep_start),
        .rd_base     ($clog2(BAND_DEPTH)'(ch_base * cfg_ch_stride)),
        .ch_stride   ($clog2(BAND_DEPTH)'(cfg_ch_stride)),
        .row_width   (cfg_width),
        .win_vld     (win_vld),
        .win         (win),
        .win_x       (win_x)
    );

    // The array's K-vector is a K-byte slice of the window at tap_off.
    wire [WIN_CH*3*3*8-1:0] win_shifted = win >> (tap_off * 16'd8);
    wire [K*8-1:0]          a_vec       = win_shifted[K*8-1:0];

    // -----------------------------------------------------------------------
    // mac_array
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // Weight shift engine.
    //
    // This runs independently of the compute FSM, which is the whole point:
    // the shift chain writes each PE's SHADOW register, so the next k-tile can
    // load while the current one is still sweeping. Serialising them instead
    // costs K cycles per k-tile, and with 72 k-tiles per output row in conv4
    // that dominates -- rtl/sim/layer_cycles.py measures the difference as
    // 47.5% array utilisation against 92%.
    // -----------------------------------------------------------------------
    reg [7:0]           wkt;        // k-tile being shifted into the shadows
    reg [$clog2(K)-1:0] wcnt;
    reg                 shifting;
    reg                 w_ready;    // shadows hold tile wkt, awaiting commit
    reg                 start_shift; // FSM request pulse
    reg                 want_shift; // request latched until it can run
    reg [7:0]           next_kt;
    reg [$clog2(K+1)-1:0] hold;     // cycles the shadows must stay untouched

    wire w_shift_en = shifting;
    reg  w_switch;

    // Shift cycle i presents W[m][K-1-i] for every m at once, so PE(k,m) ends
    // up holding W[m][k] after K shifts. One read at one address.
    //
    // This must sit AFTER wkt and wcnt are declared. Icarus elaborates the
    // whole module before resolving names and so tolerates a forward
    // reference here; Genus's parser does not, and rejects it outright with
    // "Reference to undeclared variable". Declaration order is not optional.
    wire [M*8-1:0] w_top = wmem[WW_AW'({8'd0, wkt} * K + (K-1-wcnt))];

    // -----------------------------------------------------------------------
    // The commit is SKEWED: row k copies shadow -> active at commit_cycle + k,
    // so the shadows have to hold still for K cycles after a commit is issued.
    // Starting the next tile's shift immediately overwrites them, and rows
    // below 0 then commit whatever the shift left behind -- an array holding
    // neither tile. That is exactly how this first failed: k-tile 0 produced
    // garbage while k-tile 1, whose shift had finished, was perfect.
    //
    // So a requested shift waits out `hold`. The next tile still loads inside
    // the current sweep whenever W >= 2K, which covers conv2 (W=101) and conv3
    // (W=50). conv4 (W=25) stalls a few cycles per tile, which the cycle model
    // in rtl/sim/layer_cycles.py accounts for.
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wkt        <= 8'd0;
            wcnt       <= {$clog2(K){1'b0}};
            shifting   <= 1'b0;
            w_ready    <= 1'b0;
            want_shift <= 1'b0;
            hold       <= {$clog2(K+1){1'b0}};
        end else begin
            if (w_switch)
                hold <= $clog2(K+1)'(K);
            else if (hold != 0)
                hold <= hold - 1'b1;

            if (start_shift)
                want_shift <= 1'b1;

            if (shifting) begin
                if (wcnt == $clog2(K)'(K-1)) begin
                    shifting <= 1'b0;
                    w_ready  <= 1'b1;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
            end else if (want_shift && (hold == 0) && !w_switch) begin
                wkt        <= next_kt;
                wcnt       <= {$clog2(K){1'b0}};
                shifting   <= 1'b1;
                w_ready    <= 1'b0;
                want_shift <= 1'b0;
            end

            if (w_switch)
                w_ready <= 1'b0;
        end
    end

    wire r_vld;
    wire [M*ACC_W-1:0] r_vec;

    mac_array #(.K(K), .M(M), .ACC_W(ACC_W), .PIPE(PIPE)) u_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .w_shift_en (w_shift_en),
        .w_top      (w_top),
        .w_switch   (w_switch),
        .a_vld      (win_vld && (state == S_SWEEP)),
        .a_vec      (a_vec),
        .r_vld      (r_vld),
        .r_vec      (r_vec)
    );

    // -----------------------------------------------------------------------
    // Row accumulator: partial sums across k-tiles for one output row.
    // -----------------------------------------------------------------------
    reg [M*ACC_W-1:0] rowacc [0:MAXW-1];
    reg [XW-1:0]      oc;        // output column of the arriving result
    reg [7:0]         rkt;       // k-tile the arriving result belongs to

    // With sweeps running back to back there is no drain between k-tiles, so
    // results arrive continuously and "which tile is this?" can no longer be a
    // single flag set by the FSM. It is derived from the result stream itself:
    // every cfg_width results completes one tile.
    wire first_kt = (rkt == 8'd0);

    reg [M*ACC_W-1:0] acc_sum;
    integer ma;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            oc  <= {XW{1'b0}};
            rkt <= 8'd0;
        end else if ((state == S_IDLE) && row_go) begin
            oc  <= {XW{1'b0}};
            rkt <= 8'd0;
        end else if (r_vld) begin
            for (ma = 0; ma < M; ma = ma + 1) begin
                acc_sum[ma*ACC_W +: ACC_W] =
                    first_kt ? r_vec[ma*ACC_W +: ACC_W]
                             : ($signed(rowacc[oc][ma*ACC_W +: ACC_W])
                                + $signed(r_vec[ma*ACC_W +: ACC_W]));
            end
            rowacc[oc] <= acc_sum;
            if ({{(16-XW){1'b0}}, oc} + 16'd1 >= cfg_width) begin
                oc  <= {XW{1'b0}};
                rkt <= rkt + 8'd1;
            end else begin
                oc <= oc + 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // writeback: requantise + BatchNorm + ReLU + 2x2 max-pool
    // -----------------------------------------------------------------------
    // The line buffer indexes POOLED columns, so it needs half the width.
    writeback #(.M(M), .ACC_W(ACC_W), .MULT_W(MULT_W),
                .MAXW((MAXW+1)/2)) u_wb (
        .clk           (clk),
        .rst_n         (rst_n),
        .cfg_we        (cfg_we),
        .cfg_ch        (cfg_ch),
        .cfg_bias      (cfg_bias),
        .cfg_mult      (cfg_mult),
        .cfg_shift     (cfg_shift),
        .cfg_relu_en   (cfg_relu_en),
        .cfg_pool_en   (cfg_pool_en),
        .cfg_row_width (cfg_width),
        .start         (layer_start),
        .acc_vld       (state == S_FLUSH),
        .acc_vec       (rowacc[flush_x[XW-1:0]]),
        .out_vld       (out_vld),
        .out_vec       (out_vec)
    );

    // -----------------------------------------------------------------------
    // Sequencer FSM
    //
    // Sweeps run BACK TO BACK. The commit that swaps k-tile kt-1's weights for
    // kt's is skewed down the array at exactly the speed of the data, so the
    // last diagonals of kt-1 finish with kt-1's weights while kt's first data
    // follows one cycle behind. That is what makes the inter-tile drain
    // unnecessary; only the final tile of a row drains.
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            ckt         <= 8'd0;
            drain_cnt   <= 16'd0;
            flush_x     <= 16'd0;
            w_switch    <= 1'b0;
            sweep_start <= 1'b0;
            start_shift <= 1'b0;
            next_kt     <= 8'd0;
        end else begin
            w_switch    <= 1'b0;
            sweep_start <= 1'b0;
            start_shift <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (row_go) begin
                        ckt         <= 8'd0;
                        next_kt     <= 8'd0;
                        start_shift <= 1'b1;      // shift tile 0
                        state       <= S_WWAIT;
                    end
                end

                S_WWAIT: begin
                    // The shadows are loading. Commit and launch as soon as
                    // they are ready. Only reachable for tile 0, or if a row is
                    // narrower than K cycles so the shift could not hide.
                    if (w_ready) begin
                        w_switch    <= 1'b1;
                        sweep_start <= 1'b1;
                        if ({8'd0, ckt} + 16'd1 < {8'd0, cfg_ktiles}) begin
                            next_kt     <= ckt + 8'd1;
                            start_shift <= 1'b1;   // overlap the next tile
                        end
                        state <= S_SWEEP;
                    end
                end

                S_SWEEP: begin
                    if (win_vld && (win_x + 16'd1 >= cfg_width)) begin
                        if ({8'd0, ckt} + 16'd1 >= {8'd0, cfg_ktiles}) begin
                            drain_cnt <= 16'd0;
                            state     <= S_DRAIN;
                        end else if (w_ready) begin
                            // Next tile is already in the shadows: swap and go.
                            ckt         <= ckt + 8'd1;
                            w_switch    <= 1'b1;
                            sweep_start <= 1'b1;
                            if ({8'd0, ckt} + 16'd2 < {8'd0, cfg_ktiles}) begin
                                next_kt     <= ckt + 8'd2;
                                start_shift <= 1'b1;
                            end
                        end else begin
                            ckt   <= ckt + 8'd1;
                            state <= S_WWAIT;
                        end
                    end
                end

                S_DRAIN: begin
                    if (drain_cnt >= 16'(LATENCY) + 16'd2) begin
                        flush_x <= 16'd0;
                        state   <= S_FLUSH;
                    end else begin
                        drain_cnt <= drain_cnt + 16'd1;
                    end
                end

                S_FLUSH: begin
                    if (flush_x + 16'd1 >= cfg_width)
                        state <= S_IDLE;
                    else
                        flush_x <= flush_x + 16'd1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
