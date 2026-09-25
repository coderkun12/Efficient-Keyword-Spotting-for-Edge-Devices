// ---------------------------------------------------------------------------
// tile_top -- the M2 single-tile INT8 accelerator.
//
//   AXI4-Lite  : control, status, weight staging, last-result readback
//   AXI4-Stream: activation vectors in (K bytes/beat), result vectors out
//   mac_array  : K x M weight-stationary INT8 systolic array
//
// INTERFACE SPLIT, and why (ACCELERATOR_PLAN.md section 3):
// Control is register-mapped because it is low rate and needs addressing.
// Activations and results are streams because they are high rate, sequential,
// and never randomly addressed -- one activation vector per cycle at 500 MHz is
// K bytes/cycle, which no register interface should carry.
//
// FLOW:
//   1. Host writes K*M weight bytes into the staging array at WEIGHT_BASE.
//   2. Host writes CTRL.w_load. A K-cycle FSM shifts the whole tile into the
//      array -- K cycles for all M columns, because every column shifts in
//      parallel down its own chain.
//   3. Host streams activation vectors. Results appear on the output stream
//      LATENCY = K + M - 1 cycles later, one per cycle, no stalls.
//
// The weight staging array is plain flops here with M combinational read
// ports, which is fine at K=M=16 (256 bytes) and deliberately simple for M2.
// It is the wrong structure at scale and M3 replaces it with a per-column SRAM,
// which is also where the on-chip activation scratchpad lands.
// ---------------------------------------------------------------------------

`default_nettype none

module tile_top #(
    parameter int K        = 16,
    parameter int M        = 16,
    parameter int ACC_W    = 32,
    parameter int PIPE     = 1,
    parameter int ADDR_W   = 16,
    parameter int FIFO_DEPTH = 16
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // ---------------- AXI4-Lite slave: control / status / weights ----------
    input  wire [ADDR_W-1:0]         s_axil_awaddr,
    input  wire [2:0]                s_axil_awprot,
    input  wire                      s_axil_awvalid,
    output reg                       s_axil_awready,
    input  wire [31:0]               s_axil_wdata,
    input  wire [3:0]                s_axil_wstrb,
    input  wire                      s_axil_wvalid,
    output reg                       s_axil_wready,
    output reg  [1:0]                s_axil_bresp,
    output reg                       s_axil_bvalid,
    input  wire                      s_axil_bready,
    input  wire [ADDR_W-1:0]         s_axil_araddr,
    input  wire [2:0]                s_axil_arprot,
    input  wire                      s_axil_arvalid,
    output reg                       s_axil_arready,
    output reg  [31:0]               s_axil_rdata,
    output reg  [1:0]                s_axil_rresp,
    output reg                       s_axil_rvalid,
    input  wire                      s_axil_rready,

    // ---------------- AXI4-Stream slave: activation vectors ----------------
    input  wire [K*8-1:0]            s_axis_tdata,
    input  wire                      s_axis_tvalid,
    output wire                      s_axis_tready,
    input  wire                      s_axis_tlast,

    // ---------------- AXI4-Stream master: result vectors -------------------
    output wire [M*ACC_W-1:0]        m_axis_tdata,
    output wire                      m_axis_tvalid,
    input  wire                      m_axis_tready,
    output wire                      m_axis_tlast
);

    // ---------------- Register map -----------------------------------------
    localparam [ADDR_W-1:0] ADDR_CTRL   = 16'h0000;
    localparam [ADDR_W-1:0] ADDR_STATUS = 16'h0004;
    localparam [ADDR_W-1:0] ADDR_ID     = 16'h0008;
    localparam [ADDR_W-1:0] ADDR_CFG    = 16'h000C;
    localparam [ADDR_W-1:0] WEIGHT_BASE = 16'h1000;
    localparam [ADDR_W-1:0] RESULT_BASE = 16'h2000;

    localparam [31:0] ID_VALUE = 32'h4B575301;   // "KWS" rev 1

    localparam int NUM_WEIGHTS = K * M;
    localparam int WEIGHT_WORDS = (NUM_WEIGHTS + 3) / 4;
    localparam int CNT_W = (K > 1) ? $clog2(K) : 1;

    // ---------------- Weight staging ---------------------------------------
    reg [7:0] wstage [0:NUM_WEIGHTS-1];

    // ---------------- Weight load FSM --------------------------------------
    reg              w_busy;
    reg              w_loaded;
    reg [CNT_W-1:0]  w_cnt;
    reg              w_load_req;     // CTRL[0]: shift + auto-commit (blocking)
    reg              w_load_nb_req;  // CTRL[1]: shift only, overlaps streaming
    reg              w_commit_req;   // CTRL[2]: commit shadow -> active now
    reg              w_nb;           // this load was started non-blocking
    reg              w_commit_pend;  // commit arrived mid-shift, deferred
    reg              w_switch_r;     // one-cycle commit pulse

    wire [M*8-1:0] w_top;

    genvar gm;
    generate
        for (gm = 0; gm < M; gm = gm + 1) begin : g_wtop
            // Shift cycle i must present W[m][K-1-i] so that after K shifts
            // PE(k,m) holds W[m][k]. Staging index is m*K + k.
            assign w_top[gm*8 +: 8] = wstage[gm*K + (K-1-w_cnt)];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_busy     <= 1'b0;
            w_loaded   <= 1'b0;
            w_cnt      <= {CNT_W{1'b0}};
            w_switch_r <= 1'b0;
            w_nb       <= 1'b0;
            w_commit_pend <= 1'b0;
        end else begin
            w_switch_r <= 1'b0;

            // A commit request is always LATCHED, never applied directly.
            //
            // Applying one while the shift chain is still moving would splice
            // two tiles together: rows already shifted would commit the new
            // tile and the rest the old one, leaving an array that holds
            // NEITHER. The arithmetic still looks plausible, so nothing
            // downstream would flag it. Latching and releasing the commit only
            // when the chain is settled makes that impossible to get wrong
            // from software.
            if (w_commit_req)
                w_commit_pend <= 1'b1;

            if (!w_busy) begin
                if (w_load_req || w_load_nb_req) begin
                    w_busy   <= 1'b1;
                    w_nb     <= w_load_nb_req;
                    w_loaded <= 1'b0;
                    w_cnt    <= {CNT_W{1'b0}};
                end
            end else begin
                if (w_cnt == CNT_W'(K-1)) begin
                    w_busy   <= 1'b0;
                    w_loaded <= 1'b1;
                    w_cnt    <= {CNT_W{1'b0}};
                    // A blocking load auto-commits; a non-blocking one waits
                    // for CTRL[2] so the host can place the switch at a stream
                    // boundary and overlap the load with the previous tile.
                    if (!w_nb)
                        w_switch_r <= 1'b1;
                end else begin
                    w_cnt <= w_cnt + 1'b1;
                end
            end

            // Release a latched commit as soon as the chain is settled: either
            // idle now, or finishing its last shift on this very edge. Placed
            // last so it wins over the branches above, which is what stops a
            // request arriving on the final shift cycle from being stranded.
            if (w_commit_pend && (!w_busy || w_cnt == CNT_W'(K-1))) begin
                w_switch_r    <= 1'b1;
                w_commit_pend <= 1'b0;
            end
        end
    end

    // ---------------- Array -------------------------------------------------
    // Activations are refused while weights are shifting, so the host cannot
    // stream into a half-loaded tile.
    assign s_axis_tready = !(w_busy && !w_nb) && !w_switch_r;

    wire a_vld = s_axis_tvalid && s_axis_tready;

    wire                 r_vld;
    wire [M*ACC_W-1:0]   r_vec;

    mac_array #(.K(K), .M(M), .ACC_W(ACC_W), .PIPE(PIPE)) u_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .w_shift_en (w_busy),
        .w_top      (w_top),
        .w_switch   (w_switch_r),
        .a_vld      (a_vld),
        .a_vec      (s_axis_tdata),
        .r_vld      (r_vld),
        .r_vec      (r_vec)
    );

    // ---------------- Result framing ---------------------------------------
    // The tile has no idea where a convolution layer ends, so it does not
    // invent framing: it MIRRORS the host's. s_axis_tlast rides a shift
    // register of exactly the array's latency and comes back out on the result
    // beat computed from that input beat. Send a burst of N vectors as one
    // frame, get N results back as one frame.
    localparam int LAT = PIPE * K + M - 1;

    reg [LAT-1:0] last_pipe;
    integer lp;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_pipe <= {LAT{1'b0}};
        end else begin
            last_pipe[0] <= a_vld && s_axis_tlast;
            for (lp = 1; lp < LAT; lp = lp + 1)
                last_pipe[lp] <= last_pipe[lp-1];
        end
    end

    wire r_last = last_pipe[LAT-1];

    // ---------------- Result path ------------------------------------------
    wire fifo_empty, fifo_full, fifo_overflow;
    wire [M*ACC_W:0] fifo_dout;

    axis_result_fifo #(.WIDTH(M*ACC_W + 1), .DEPTH(FIFO_DEPTH)) u_fifo (
        .clk           (clk),
        .rst_n         (rst_n),
        .wr_en         (r_vld),
        .wr_data       ({r_last, r_vec}),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tdata  (fifo_dout),
        .empty         (fifo_empty),
        .full          (fifo_full),
        .overflow      (fifo_overflow)
    );

    assign m_axis_tdata = fifo_dout[M*ACC_W-1:0];
    assign m_axis_tlast = fifo_dout[M*ACC_W];

    // Last result vector, latched for AXI4-Lite readback. Debug visibility
    // only -- the stream is the real data path.
    reg [M*ACC_W-1:0] result_latch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)      result_latch <= {(M*ACC_W){1'b0}};
        else if (r_vld)  result_latch <= r_vec;
    end

    // ---------------- AXI4-Lite write channel ------------------------------
    // Registered ready pulses: ready is asserted the cycle AFTER both valids
    // are seen, so ready never combinationally depends on valid.
    wire wr_fire = s_axil_awready && s_axil_wready;
    integer bi;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_awready <= 1'b0;
            s_axil_wready  <= 1'b0;
            s_axil_bvalid  <= 1'b0;
            s_axil_bresp   <= 2'b00;
            w_load_req     <= 1'b0;
            w_load_nb_req  <= 1'b0;
            w_commit_req   <= 1'b0;
        end else begin
            s_axil_awready <= 1'b0;
            s_axil_wready  <= 1'b0;
            w_load_req     <= 1'b0;      // one-cycle pulses
            w_load_nb_req  <= 1'b0;
            w_commit_req   <= 1'b0;

            if (!s_axil_awready && !s_axil_wready && !s_axil_bvalid &&
                s_axil_awvalid && s_axil_wvalid) begin
                s_axil_awready <= 1'b1;
                s_axil_wready  <= 1'b1;
            end

            if (wr_fire) begin
                s_axil_bvalid <= 1'b1;
                s_axil_bresp  <= 2'b00;              // OKAY

                if (s_axil_awaddr == ADDR_CTRL) begin
                    if (s_axil_wdata[0]) w_load_req    <= 1'b1;  // load + commit
                    if (s_axil_wdata[1]) w_load_nb_req <= 1'b1;  // load only
                    if (s_axil_wdata[2]) w_commit_req  <= 1'b1;  // commit now
                end else if (s_axil_awaddr >= WEIGHT_BASE &&
                             s_axil_awaddr < WEIGHT_BASE + ADDR_W'(WEIGHT_WORDS*4)) begin
                    for (bi = 0; bi < 4; bi = bi + 1) begin
                        if (s_axil_wstrb[bi]) begin
                            if ((((s_axil_awaddr - WEIGHT_BASE) >> 2) * 4 + bi)
                                    < NUM_WEIGHTS) begin
                                wstage[(((s_axil_awaddr - WEIGHT_BASE) >> 2) * 4) + bi]
                                    <= s_axil_wdata[bi*8 +: 8];
                            end
                        end
                    end
                end else begin
                    s_axil_bresp <= 2'b10;           // SLVERR: unmapped write
                end
            end

            if (s_axil_bvalid && s_axil_bready)
                s_axil_bvalid <= 1'b0;
        end
    end

    // ---------------- AXI4-Lite read channel -------------------------------
    reg [31:0] rd_mux;
    integer    ri;

    always @(*) begin
        rd_mux = 32'h0;
        if (s_axil_araddr == ADDR_STATUS) begin
            rd_mux = {27'd0, fifo_overflow, fifo_full, fifo_empty, w_busy, w_loaded};
        end else if (s_axil_araddr == ADDR_ID) begin
            rd_mux = ID_VALUE;
        end else if (s_axil_araddr == ADDR_CFG) begin
            rd_mux = {8'd0, 8'(ACC_W), 8'(M), 8'(K)};
        end else if (s_axil_araddr == ADDR_CTRL) begin
            rd_mux = 32'h0;
        end else if (s_axil_araddr >= RESULT_BASE &&
                     s_axil_araddr < RESULT_BASE + ADDR_W'(M*4)) begin
            for (ri = 0; ri < M; ri = ri + 1)
                if (ADDR_W'(ri) == ((s_axil_araddr - RESULT_BASE) >> 2))
                    rd_mux = result_latch[ri*ACC_W +: 32];
        end else if (s_axil_araddr >= WEIGHT_BASE &&
                     s_axil_araddr < WEIGHT_BASE + ADDR_W'(WEIGHT_WORDS*4)) begin
            for (ri = 0; ri < 4; ri = ri + 1)
                if ((((s_axil_araddr - WEIGHT_BASE) >> 2) * 4 + ri) < NUM_WEIGHTS)
                    rd_mux[ri*8 +: 8] =
                        wstage[(((s_axil_araddr - WEIGHT_BASE) >> 2) * 4) + ri];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_arready <= 1'b0;
            s_axil_rvalid  <= 1'b0;
            s_axil_rdata   <= 32'h0;
            s_axil_rresp   <= 2'b00;
        end else begin
            s_axil_arready <= 1'b0;

            if (!s_axil_arready && !s_axil_rvalid && s_axil_arvalid)
                s_axil_arready <= 1'b1;

            if (s_axil_arready) begin
                s_axil_rdata  <= rd_mux;
                s_axil_rresp  <= 2'b00;
                s_axil_rvalid <= 1'b1;
            end

            if (s_axil_rvalid && s_axil_rready)
                s_axil_rvalid <= 1'b0;
        end
    end

endmodule

`default_nettype wire
