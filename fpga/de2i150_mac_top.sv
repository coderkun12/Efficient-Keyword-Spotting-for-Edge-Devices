// ---------------------------------------------------------------------------
// de2i150_mac_top -- standalone FPGA self-test for the mac_array.
//
// WHAT THIS PROVES
// Simulation says the array is correct. This says the array is correct AFTER
// synthesis, placement, routing and timing closure on real silicon, driven by
// a real crystal. Those are different claims: the second one catches inferred
// latches, X-optimism in simulation, and paths that only fail at temperature.
//
// It is deliberately SELF-CONTAINED. No PCIe, no Atom N2600, no UART, no host
// driver. Weights, activations and golden results are compiled into on-chip
// ROMs by $readmemb, from vector files that fpga/gen_test_vectors.py writes
// out of the same Python reference model the cocotb tests use. The board
// powers up, runs the whole vector set, and reports on the LEDs. Bring-up
// therefore has exactly one failure mode to debug -- the array -- not five.
//
// The PCIe path to the on-board Atom is the NEXT step, and is what turns this
// from "the array works in hardware" into "the board runs inferences".
//
// SEQUENCE
//   POR      65k cycles of power-on reset, because KEY[0] idles HIGH
//   LOAD     K cycles of w_shift_en, streaming weights in shift order
//   COMMIT   one cycle of w_switch
//   SETTLE   K cycles -- the commit is skewed one row per PIPE cycles, so the
//            last row does not hold the new weights until commit + PIPE*(K-1)
//   STREAM   NVEC cycles of a_vld with one activation vector each
//   DRAIN    LATENCY + slack cycles for the final results to fall out
//   DONE     hold the verdict
//
// LEDS
//   LEDG[0]      running
//   LEDG[1]      done
//   LEDG[2]      PASS   -- all NVEC results matched, exactly NVEC arrived,
//                AND the vectors were real (see LEDG[4])
//   LEDG[3]      FAIL
//   LEDG[4]      vectors are non-zero, i.e. $readmemb actually loaded them.
//                Dark with LEDG[2] dark means the ROMs are empty and the run
//                proved nothing -- a different fault from a wrong result.
//   LEDG[7]      heartbeat, ~1.5 Hz. If this is dark the clock is the problem,
//                not the array. Check this LED first.
//   LEDR[7:0]    mismatch count, saturating
//   LEDR[15:8]   results received
// ---------------------------------------------------------------------------

`default_nettype none

module de2i150_mac_top #(
    parameter int K     = 16,
    parameter int M     = 16,
    parameter int ACC_W = 32,
    parameter int PIPE  = 1,
    parameter int NVEC  = 64,            // must match gen_test_vectors.py
    parameter     WMIF  = "weights.txt",   // resolved relative to the Quartus
    parameter     AMIF  = "acts.txt",      // project dir; the testbench
    parameter     EMIF  = "expected.txt"   // overrides to run from repo root
) (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,              // active low, KEY[0] = manual restart
    output reg  [8:0]  LEDG,
    output reg  [17:0] LEDR
);

    localparam int LATENCY = PIPE * K + M - 1;
    localparam int AW      = $clog2(NVEC);
    localparam int KW      = $clog2(K);

    // -----------------------------------------------------------------------
    // Power-on reset. KEY[0] idles HIGH on Terasic boards, so it cannot be the
    // only reset source -- nothing would ever reset at power-up.
    // -----------------------------------------------------------------------
    reg [15:0] por_cnt = 16'd0;
    reg        rst_n   = 1'b0;

    always @(posedge CLOCK_50) begin
        if (!KEY[0]) begin                       // pressed: restart the test
            por_cnt <= 16'd0;
            rst_n   <= 1'b0;
        end else if (por_cnt != 16'hFFFF) begin
            por_cnt <= por_cnt + 16'd1;
            rst_n   <= 1'b0;
        end else begin
            rst_n   <= 1'b1;
        end
    end

    // -----------------------------------------------------------------------
    // Vector ROMs, initialised at configuration time.
    //
    // ONE mechanism, visible to both tools: $readmemb in an unguarded initial
    // block. Quartus supports this for inferring initialised ROM, and Icarus
    // supports it natively.
    //
    // DO NOT go back to (* ram_init_file = "...mif" *). That attribute sets
    // the power-up contents of a RAM that HAS a write port; it does not make
    // a ROM out of a memory nothing writes. Quartus accepted it silently and
    // then reported all three arrays as "used but never assigned", which
    // would have put undefined data through the array on real hardware --
    // and the checker compares X to X, so every LED would have said PASS.
    //
    // Paths are parameters and resolve relative to the Quartus PROJECT
    // directory, which is why the project must live in fpga/ next to the
    // .txt files. The testbench overrides them to run from the repo root.
    // -----------------------------------------------------------------------
    reg [M*8-1:0]     wrom [0:K-1];
    reg [K*8-1:0]     arom [0:NVEC-1];
    reg [M*ACC_W-1:0] erom [0:NVEC-1];

    initial begin
        $readmemb(WMIF, wrom);
        $readmemb(AMIF, arom);
        $readmemb(EMIF, erom);
    end

    // -----------------------------------------------------------------------
    // Control FSM
    // -----------------------------------------------------------------------
    localparam [2:0] S_LOAD   = 3'd0,
                     S_COMMIT = 3'd1,
                     S_SETTLE = 3'd2,
                     S_STREAM = 3'd3,
                     S_DRAIN  = 3'd4,
                     S_DONE   = 3'd5;

    reg [2:0]          state;
    reg [AW:0]         idx;              // one spare bit: counts up TO NVEC
    reg [15:0]         timer;

    reg                w_shift_en;
    reg [M*8-1:0]      w_top;
    reg                w_switch;
    reg                a_vld;
    reg [K*8-1:0]      a_vec;

    wire               r_vld;
    wire [M*ACC_W-1:0] r_vec;

    // 16 bits, not the $clog2(NVEC)+1 the count strictly needs: the LED bus
    // displays recv[7:0], and a part-select wider than the register reads X.
    // Costing eight flops to keep every slice in range is the right trade.
    reg [15:0]         recv;
    reg [7:0]          errors;
    reg                fail;

    always @(posedge CLOCK_50) begin
        if (!rst_n) begin
            state      <= S_LOAD;
            idx        <= {(AW+1){1'b0}};
            timer      <= 16'd0;
            w_shift_en <= 1'b0;
            w_switch   <= 1'b0;
            a_vld      <= 1'b0;
            w_top      <= {(M*8){1'b0}};
            a_vec      <= {(K*8){1'b0}};
        end else begin
            w_shift_en <= 1'b0;
            w_switch   <= 1'b0;
            a_vld      <= 1'b0;

            case (state)
                // Stream the weight tile into the shadow registers. The .mif
                // is already in shift order -- W[m][K-1] first -- so this is a
                // straight walk with no address arithmetic to get wrong.
                S_LOAD: begin
                    w_shift_en <= 1'b1;
                    w_top      <= wrom[idx[KW-1:0]];
                    if (idx == K - 1) begin
                        idx   <= {(AW+1){1'b0}};
                        state <= S_COMMIT;
                    end else begin
                        idx <= idx + 1'b1;
                    end
                end

                S_COMMIT: begin
                    w_switch <= 1'b1;
                    timer    <= 16'd0;
                    state    <= S_SETTLE;
                end

                // The commit walks down the array one row per PIPE cycles, so
                // row K-1 does not hold the new tile until PIPE*(K-1) cycles
                // after the pulse. Injecting activations before that mixes old
                // and new weights into one result -- the exact bug that made
                // k-tile 0 produce garbage during integration.
                S_SETTLE: begin
                    if (timer == PIPE * K) begin
                        timer <= 16'd0;
                        state <= S_STREAM;
                    end else begin
                        timer <= timer + 16'd1;
                    end
                end

                S_STREAM: begin
                    a_vld <= 1'b1;
                    a_vec <= arom[idx[AW-1:0]];
                    if (idx == NVEC - 1) begin
                        idx   <= {(AW+1){1'b0}};
                        timer <= 16'd0;
                        state <= S_DRAIN;
                    end else begin
                        idx <= idx + 1'b1;
                    end
                end

                S_DRAIN: begin
                    if (timer == LATENCY + 16) state <= S_DONE;
                    else                       timer <= timer + 16'd1;
                end

                S_DONE: state <= S_DONE;

                default: state <= S_DONE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Result checker. Results emerge in input order at fixed latency, so one
    // counter indexes the golden ROM.
    //
    // The COUNT check matters as much as the value check: an array that emits
    // 63 correct results and drops one is broken, and comparing only the
    // results that do arrive would call that a pass.
    // -----------------------------------------------------------------------
    always @(posedge CLOCK_50) begin
        if (!rst_n) begin
            recv   <= 16'd0;
            errors <= 8'd0;
            fail   <= 1'b0;
        end else if (r_vld) begin
            if (recv < NVEC) begin
                if (r_vec !== erom[recv[AW-1:0]]) begin
                    fail <= 1'b1;
                    if (errors != 8'hFF) errors <= errors + 8'd1;
                end
                recv <= recv + 16'd1;
            end else begin
                fail <= 1'b1;                    // more results than vectors
                if (errors != 8'hFF) errors <= errors + 8'd1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Vector sanity -- the last vacuous-pass hole, closed on hardware.
    //
    // If $readmemb finds nothing, all three ROMs power up as zero. The array
    // then computes zero, the golden ROM reads zero, zero matches zero, and
    // every LED reports PASS on a board that proved nothing. The simulation
    // guard catches the X case; this catches the ZERO case, which is what a
    // failed load looks like in silicon.
    //
    // These OR-reductions reuse w_top, a_vec and the checker's existing erom
    // read, so no new read port is added and the ROMs stay in M9K.
    // -----------------------------------------------------------------------
    reg seen_w, seen_a, seen_e;

    always @(posedge CLOCK_50) begin
        if (!rst_n) begin
            seen_w <= 1'b0;
            seen_a <= 1'b0;
            seen_e <= 1'b0;
        end else begin
            if (w_shift_en && (w_top != {(M*8){1'b0}}))        seen_w <= 1'b1;
            if (a_vld      && (a_vec != {(K*8){1'b0}}))        seen_a <= 1'b1;
            if (r_vld && (recv < NVEC)
                      && (erom[recv[AW-1:0]] != {(M*ACC_W){1'b0}})) seen_e <= 1'b1;
        end
    end

    wire vectors_ok = seen_w && seen_a && seen_e;

    // -----------------------------------------------------------------------
    // Heartbeat and status
    // -----------------------------------------------------------------------
    reg [24:0] beat = 25'd0;
    always @(posedge CLOCK_50) beat <= beat + 25'd1;

    wire done   = (state == S_DONE);
    wire passed = done && !fail && (recv == NVEC) && vectors_ok;

    always @(posedge CLOCK_50) begin
        LEDG       <= 9'd0;
        LEDR       <= 18'd0;
        LEDG[0]    <= !done;
        LEDG[1]    <= done;
        LEDG[2]    <= passed;
        LEDG[3]    <= done && !passed;
        LEDG[4]    <= vectors_ok;
        LEDG[7]    <= beat[24];
        LEDR[7:0]  <= errors;
        LEDR[15:8] <= recv[7:0];
    end

    // -----------------------------------------------------------------------
    mac_array #(
        .K     (K),
        .M     (M),
        .ACC_W (ACC_W),
        .PIPE  (PIPE)
    ) u_array (
        .clk        (CLOCK_50),
        .rst_n      (rst_n),
        .w_shift_en (w_shift_en),
        .w_top      (w_top),
        .w_switch   (w_switch),
        .a_vld      (a_vld),
        .a_vec      (a_vec),
        .r_vld      (r_vld),
        .r_vec      (r_vec)
    );

endmodule

`default_nettype wire
