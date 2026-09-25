// ---------------------------------------------------------------------------
// de2i150_layer_top -- stage C4: a full fused conv layer, self-checking, on
//                      the DE2i-150.
//
// WHAT THIS RUNS
// conv4 m-tile 0 of the keyword-spotting model: 128 -> 16 output channels,
// W=25, H=10, all 72 k-tiles accumulated on the array, then BatchNorm folded
// into the requantiser, ReLU, and 2x2 max-pool fused into the write-back.
// 60 pooled INT8 output vectors, each checked against a golden value.
//
// That is the complete accelerator datapath -- scratchpad, systolic array,
// k-tile accumulation, fused write-back -- not a slice of it. Stage A proved
// the array alone; this proves the layer.
//
// WHERE THE GOLDEN DATA COMES FROM
// fpga/gen_layer_vectors.py, which computes it with conv_layer() and
// fused_writeback() out of rtl/tb/ref_model.py -- the same two functions the
// eight passing cocotb tests check layer_top against. There is no second
// model here to disagree with the first.
//
// THE DRIVE SEQUENCE is lifted from test_layer_top.py's Layer class, so the
// hardware and the simulation exercise layer_top identically:
//
//   LOADW   18,432 weight bytes at kt*K*M + m*K + k
//   LOADC   16 requant words: bias, multiplier, shift
//   CFG     width/k-tiles/stride/ReLU/pool, then a layer_start pulse
//   per row FILL 9,600 band bytes (3 rows x 128 ch x 25 col), row_go, wait busy
//   DONE    hold the verdict
//
// ROM NOTE THAT COST AN AFTERNOON
// Every ROM here is declared at MODULE scope. A reg array inside a generate
// block does not map to block RAM on this device and silently becomes
// registers instead -- 412 Kb of them, for these vectors. See byte_ram.sv.
//
// LEDS
//   LEDG[7]      heartbeat, ~1.5 Hz -- check this first
//   LEDG[0]      running
//   LEDG[1]      done
//   LEDG[2]      PASS
//   LEDG[3]      FAIL
//   LEDG[4]      vectors are non-zero (the ROMs really loaded)
//   LEDG[5]      checker overrun -- results arrived faster than they could be
//                compared. Not a wrong answer, but an unchecked one.
//   LEDR[7:0]    mismatch count, saturating
//   LEDR[15:8]   output vectors received (expect 60 = 8'b00111100)
// ---------------------------------------------------------------------------

`default_nettype none

module de2i150_layer_top #(
    parameter int M        = 16,
    parameter int K        = 16,
    parameter int ACC_W    = 28,
    parameter int MULT_W   = 16,
    parameter int CHANNELS = 128,
    parameter int NROWS    = 10,     // output rows, = layer height
    parameter int WIDTH    = 25,     // output columns
    parameter int KTILES   = 72,
    parameter int NGOLD    = 60,     // (NROWS/2) * (WIDTH/2)
    parameter     WMIF     = "layer_w.txt",
    parameter     AMIF     = "layer_a.txt",
    parameter     CMIF     = "layer_c.txt",
    parameter     GMIF     = "layer_g.txt"
) (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    output reg  [8:0]  LEDG,
    output reg  [17:0] LEDR
);

    localparam int WBYTES = KTILES * K * M;        // 18,432
    localparam int ABYTES = CHANNELS * NROWS * WIDTH; // 32,000
    localparam int CW     = ACC_W + MULT_W + 6;    // 50
    localparam int WAW    = $clog2(WBYTES);
    localparam int AAW    = $clog2(ABYTES);
    localparam int GAW    = $clog2(NGOLD);

    // -----------------------------------------------------------------------
    // Power-on reset. KEY[0] idles HIGH, so it cannot be the only source.
    // -----------------------------------------------------------------------
    reg [15:0] por_cnt = 16'd0;
    reg        rst_n   = 1'b0;

    always @(posedge CLOCK_50) begin
        if (!KEY[0]) begin
            por_cnt <= 16'd0;
            rst_n   <= 1'b0;
        end else if (por_cnt != 16'hFFFF) begin
            por_cnt <= por_cnt + 16'd1;
            rst_n   <= 1'b0;
        end else begin
            rst_n <= 1'b1;
        end
    end

    // -----------------------------------------------------------------------
    // Vector ROMs, all at module scope. Synchronous reads: the address is
    // presented on one cycle, the data is valid on the next, and the FSM
    // carries delayed copies of everything that has to line up with it.
    // -----------------------------------------------------------------------
    reg [7:0]     wrom [0:WBYTES-1];
    reg [7:0]     arom [0:ABYTES-1];
    reg [CW-1:0]  crom [0:M-1];
    reg [M*8-1:0] grom [0:NGOLD-1];

    initial begin
        $readmemb(WMIF, wrom);
        $readmemb(AMIF, arom);
        $readmemb(CMIF, crom);
        $readmemb(GMIF, grom);
    end

    // ROM outputs. The read block itself is further down, beside the
    // counters that form its addresses: those are not declared yet, and a
    // forward reference is rejected by Genus even where Icarus tolerates it.
    reg [7:0]     w_q;
    reg [7:0]     a_q;
    reg [M*8-1:0] g_q;

    // -----------------------------------------------------------------------
    // Sequencer
    // -----------------------------------------------------------------------
    localparam [3:0] S_LOADW = 4'd0,
                     S_LOADC = 4'd1,
                     S_CFG   = 4'd2,
                     S_FILL  = 4'd3,
                     S_GO    = 4'd4,
                     S_WAIT  = 4'd5,
                     S_SETTLE= 4'd6,
                     S_DRAIN = 4'd7,
                     S_DONE  = 4'd8;

    reg [3:0]  state;
    reg [15:0] timer;

    // Fill counters, ordered r -> c -> x to match test_layer_top's fill_band.
    reg [4:0]  fx;                      // 0 .. WIDTH-1
    reg [7:0]  fc;                      // 0 .. CHANNELS-1
    reg [1:0]  fr;                      // 0 .. 2
    reg [3:0]  frow;                    // output row y, 0 .. NROWS-1
    reg [WAW-1:0] wcnt;
    reg [3:0]  ccnt;

    // yy = y + r - 1, the input row this band slot reads. Outside the map it
    // is padding and must be written as ZERO, not left unwritten: an unwritten
    // byte reads X, and X times a zero weight is still X.
    wire signed [5:0] yy      = $signed({2'b00, frow}) + $signed({4'b0000, fr}) - 6'sd1;
    wire              y_valid = (yy >= 0) && (yy < $signed(6'(NROWS)));

    wire [AAW-1:0] fill_arom_addr =
        AAW'(fc * (NROWS * WIDTH) + (y_valid ? yy[3:0] : 4'd0) * WIDTH + fx);
    wire [11:0]    fill_band_addr = 12'(fc * WIDTH + fx);

    // ---- ROM reads --------------------------------------------------------
    // The addresses are WIRES off the counters, not registers.
    //
    // Registering them inserts a second pipeline stage: the address would not
    // reach the ROM until the cycle after the counter moved, so the data would
    // arrive a cycle after the write strobe that is supposed to carry it, and
    // every byte would land one address late. That is a silent corruption --
    // the sequencing, the result count and the timing all stay perfect, and
    // only the values are wrong, which is exactly how it presented: 60 of 60
    // mismatches with a flawless-looking run around them.
    wire [WAW-1:0] w_raddr = wcnt;
    wire [AAW-1:0] a_raddr = fill_arom_addr;
    wire [GAW-1:0] g_raddr = recv[GAW-1:0];

    always @(posedge CLOCK_50) begin
        w_q <= wrom[w_raddr];
        a_q <= arom[a_raddr];
        g_q <= grom[g_raddr];
    end

    // One-cycle delays so the write lines up with the ROM data.
    reg           fill_vld_d, y_valid_d;
    reg [1:0]     fr_d;
    reg [11:0]    band_addr_d;
    reg           loadw_vld_d;
    reg [WAW-1:0] wcnt_d;

    // ---- layer_top interface ----------------------------------------------
    // Driven combinationally from the one-cycle-delayed copies, so the
    // strobe, the address and the ROM data are all valid together.
    wire        bnd_wr_en   = fill_vld_d;
    wire [1:0]  bnd_wr_row  = fr_d;
    wire [11:0] bnd_wr_addr = band_addr_d;
    wire [7:0]  bnd_wr_data = y_valid_d ? a_q : 8'd0;
    wire        wm_wr_en    = loadw_vld_d;
    wire [14:0] wm_wr_addr  = 15'(wcnt_d);
    wire [7:0]  wm_wr_data  = w_q;
    reg         cfg_we;
    reg [3:0]   cfg_ch;
    reg signed [ACC_W-1:0] cfg_bias;
    reg [MULT_W-1:0]       cfg_mult;
    reg [5:0]   cfg_shift;
    reg         layer_start;
    reg         row_go;

    wire        busy;
    wire        out_vld;
    wire [M*8-1:0] out_vec;

    always @(posedge CLOCK_50) begin
        if (!rst_n) begin
            state       <= S_LOADW;
            timer       <= 16'd0;
            fx          <= 5'd0;
            fc          <= 8'd0;
            fr          <= 2'd0;
            frow        <= 4'd0;
            wcnt        <= {WAW{1'b0}};
            ccnt        <= 4'd0;
            loadw_vld_d <= 1'b0;
            fill_vld_d  <= 1'b0;
            y_valid_d   <= 1'b0;
            fr_d        <= 2'd0;
            band_addr_d <= 12'd0;
            wcnt_d      <= {WAW{1'b0}};
            cfg_we      <= 1'b0;
            layer_start <= 1'b0;
            row_go      <= 1'b0;
        end else begin
            // Default-off strobes; each state re-asserts what it needs.
            cfg_we      <= 1'b0;
            layer_start <= 1'b0;
            row_go      <= 1'b0;

            // ---- pipeline registers, always advancing --------------------
            loadw_vld_d <= (state == S_LOADW);
            wcnt_d      <= wcnt;
            fill_vld_d  <= (state == S_FILL);
            y_valid_d   <= y_valid;
            fr_d        <= fr;
            band_addr_d <= fill_band_addr;

            case (state)
                // Stream every weight byte in host address order. The ROM is
                // already in that order, so this is a plain count.
                S_LOADW: begin
                    if (wcnt == WAW'(WBYTES - 1)) begin
                        wcnt  <= {WAW{1'b0}};
                        ccnt  <= 4'd0;
                        state <= S_LOADC;
                    end else begin
                        wcnt <= wcnt + 1'b1;
                    end
                end

                // 16 requant words. crom is small enough to read
                // combinationally, so no delay pipeline is needed here.
                S_LOADC: begin
                    cfg_we    <= 1'b1;
                    cfg_ch    <= ccnt;
                    cfg_bias  <= crom[ccnt][CW-1 -: ACC_W];
                    cfg_mult  <= crom[ccnt][MULT_W+5 -: MULT_W];
                    cfg_shift <= crom[ccnt][5:0];
                    if (ccnt == 4'(M - 1)) state <= S_CFG;
                    else                   ccnt  <= ccnt + 4'd1;
                end

                S_CFG: begin
                    layer_start <= 1'b1;
                    fx    <= 5'd0;
                    fc    <= 8'd0;
                    fr    <= 2'd0;
                    frow  <= 4'd0;
                    state <= S_FILL;
                end

                // 3 rows x CHANNELS x WIDTH bytes, ordered r -> c -> x, the
                // same order test_layer_top.py writes them.
                S_FILL: begin
                    if (fx == 5'(WIDTH - 1)) begin
                        fx <= 5'd0;
                        if (fc == 8'(CHANNELS - 1)) begin
                            fc <= 8'd0;
                            if (fr == 2'd2) begin
                                fr    <= 2'd0;
                                state <= S_GO;
                            end else begin
                                fr <= fr + 2'd1;
                            end
                        end else begin
                            fc <= fc + 8'd1;
                        end
                    end else begin
                        fx <= fx + 5'd1;
                    end
                end

                // Let the last band write and its pipeline land before the
                // sweep starts reading the bank.
                S_GO: begin
                    if (timer == 16'd4) begin
                        timer  <= 16'd0;
                        row_go <= 1'b1;
                        state  <= S_WAIT;
                    end else begin
                        timer <= timer + 16'd1;
                    end
                end

                // row_go was pulsed last cycle; busy has not risen yet, so
                // wait for it before waiting for it to fall.
                S_WAIT: begin
                    if (busy) state <= S_SETTLE;
                end

                // busy going low does NOT mean the row is finished. The
                // write-back is a pipeline on the array's output path, and
                // its last pooled columns emerge well after the sweep ends --
                // the final three of every row pair, in the run that caught
                // this. Starting the next band fill immediately disturbed
                // them, and the damage was invisible everywhere else: right
                // result count, right timing, right first nine columns.
                //
                // test_layer_top.py waits 8 cycles past busy before touching
                // the band, and it is the sequence layer_top is verified
                // against. 32 here, because the cost is 32 cycles out of
                // ~14,000 per row and guessing tight on a margin that already
                // bit once is a poor trade.
                S_SETTLE: begin
                    if (busy) begin
                        timer <= 16'd0;
                    end else if (timer == 16'd32) begin
                        timer <= 16'd0;
                        if (frow == 4'(NROWS - 1)) begin
                            state <= S_DRAIN;
                        end else begin
                            frow  <= frow + 4'd1;
                            state <= S_FILL;
                        end
                    end else begin
                        timer <= timer + 16'd1;
                    end
                end

                S_DRAIN: begin
                    if (timer == 16'd255) state <= S_DONE;
                    else                  timer <= timer + 16'd1;
                end

                S_DONE:  state <= S_DONE;
                default: state <= S_DONE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Checker.
    //
    // The golden ROM read is synchronous, so a result is held for one cycle
    // while its expected value is fetched. If another result arrives during
    // that cycle the checker cannot keep up -- that is not a wrong answer,
    // but it IS an unchecked one, so it raises `overrun` and fails the test
    // rather than silently skipping a comparison.
    //
    // With 2x2 pooling enabled, outputs are at least two cycles apart, so
    // this should never fire. It exists because "should never" is not a
    // property the LEDs can otherwise distinguish from "did not".
    // -----------------------------------------------------------------------
    reg [15:0]     recv;
    reg [7:0]      errors;
    reg            fail, overrun, cmp_busy;
    reg [M*8-1:0]  out_hold;
    reg            seen_w, seen_a, seen_g;

    always @(posedge CLOCK_50) begin
        if (!rst_n) begin
            recv     <= 16'd0;
            errors   <= 8'd0;
            fail     <= 1'b0;
            overrun  <= 1'b0;
            cmp_busy <= 1'b0;
            seen_w   <= 1'b0;
            seen_a   <= 1'b0;
            seen_g   <= 1'b0;
        end else begin
            // Vectors-are-real evidence, reusing reads that already happen.
            if (wm_wr_en  && (wm_wr_data  != 8'd0))        seen_w <= 1'b1;
            if (bnd_wr_en && (bnd_wr_data != 8'd0))        seen_a <= 1'b1;
            if (cmp_busy  && (g_q != {(M*8){1'b0}}))       seen_g <= 1'b1;

            if (out_vld) begin
                if (cmp_busy) begin
                    overrun <= 1'b1;
                    fail    <= 1'b1;
                end
                out_hold <= out_vec;
                cmp_busy <= 1'b1;
            end else if (cmp_busy) begin
                cmp_busy <= 1'b0;
                if (recv < NGOLD) begin
                    if (out_hold !== g_q) begin
                        fail <= 1'b1;
                        if (errors != 8'hFF) errors <= errors + 8'd1;
                    end
                end else begin
                    fail <= 1'b1;               // more results than expected
                    if (errors != 8'hFF) errors <= errors + 8'd1;
                end
                recv <= recv + 16'd1;
            end
        end
    end

    wire vectors_ok = seen_w && seen_a && seen_g;
    wire done       = (state == S_DONE);
    wire passed     = done && !fail && !overrun
                           && (recv == 16'(NGOLD)) && vectors_ok;

    // -----------------------------------------------------------------------
    reg [24:0] beat = 25'd0;
    always @(posedge CLOCK_50) beat <= beat + 25'd1;

    always @(posedge CLOCK_50) begin
        LEDG       <= 9'd0;
        LEDR       <= 18'd0;
        LEDG[0]    <= !done;
        LEDG[1]    <= done;
        LEDG[2]    <= passed;
        LEDG[3]    <= done && !passed;
        LEDG[4]    <= vectors_ok;
        LEDG[5]    <= overrun;
        LEDG[7]    <= beat[24];
        LEDR[7:0]  <= errors;
        LEDR[15:8] <= recv[7:0];
    end

    // -----------------------------------------------------------------------
    layer_fpga_cfg u_layer (
        .clk           (CLOCK_50),
        .rst_n         (rst_n),
        .bnd_wr_en     (bnd_wr_en),
        .bnd_wr_bank   (1'b0),
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
        .cfg_relu_en   (1'b1),
        .cfg_pool_en   (1'b1),
        .cfg_width     (16'(WIDTH)),
        .cfg_ktiles    (8'(KTILES)),
        .cfg_rd_bank   (1'b0),
        .cfg_ch_stride (16'(WIDTH)),
        .layer_start   (layer_start),
        .row_go        (row_go),
        .busy          (busy),
        .out_vld       (out_vld),
        .out_vec       (out_vec)
    );

endmodule

`default_nettype wire
