// ---------------------------------------------------------------------------
// tb_de2i150_layer_top -- simulate the stage C4 harness before building it.
//
// The harness reports on LEDs. A broken harness and a broken layer light the
// same one, and the board gives no way to separate them, so this runs the
// whole thing in Icarus first -- ~150,000 cycles, a few seconds of wall time,
// against a 20-minute Quartus build.
//
//   iverilog -g2012 -DSIMULATION -s tb_de2i150_layer_top -o /tmp/l.vvp \
//       fpga/tb_de2i150_layer_top.sv fpga/de2i150_layer_top.sv \
//       fpga/layer_fpga_cfg.sv rtl/rtl_design/layer_top.sv \
//       rtl/rtl_design/byte_ram.sv rtl/rtl_design/band_sram.sv \
//       rtl/rtl_design/mac_array.sv rtl/rtl_design/writeback.sv \
//       rtl/rtl_design/pe_int8.sv
//   vvp /tmp/l.vvp
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps
`default_nettype none

module tb_de2i150_layer_top;

    localparam int NGOLD = 60;

    reg         clk = 1'b0;
    reg  [3:0]  key = 4'hF;
    wire [8:0]  ledg;
    wire [17:0] ledr;

    always #10 clk = ~clk;                       // 50 MHz

    de2i150_layer_top #(
        .NGOLD (NGOLD),
        .WMIF  ("fpga/layer_w.txt"),
        .AMIF  ("fpga/layer_a.txt"),
        .CMIF  ("fpga/layer_c.txt"),
        .GMIF  ("fpga/layer_g.txt")
    ) dut (
        .CLOCK_50 (clk),
        .KEY      (key),
        .LEDG     (ledg),
        .LEDR     (ledr)
    );

    // Skip the power-on reset wait; it exists for real power rails.
    initial begin
        @(negedge clk);
        dut.por_cnt = 16'hFFFE;
    end

    // Guard against a vacuous pass: all-zero or X ROMs would compute nothing
    // and match nothing, and every LED would still say PASS.
    initial begin
        #1;
        if (^dut.wrom[0] === 1'bx || ^dut.grom[0] === 1'bx
                                  || ^dut.crom[0] === 1'bx) begin
            $display("");
            $display("  VECTOR ROMS ARE X -- run from the repo root so");
            $display("  fpga/layer_*.txt resolve. Any PASS would be vacuous.");
            $fatal(1);
        end
    end

    integer cycles = 0;
    always @(posedge clk) cycles = cycles + 1;

    // Progress, so a long run does not look like a hang.
    reg [3:0] last_state = 4'hF;
    always @(posedge clk) begin
        if (dut.state !== last_state) begin
            last_state <= dut.state;
            $display("  [%8d] state -> %0d   row %0d   recv %0d",
                     cycles, dut.state, dut.frow, dut.recv);
        end
    end

    // Which comparisons fail, and where they sit in the pooled output grid.
    // A scatter of wrong values and a whole wrong row are different bugs.
    integer nbad = 0;
    always @(posedge clk) begin
        if (rst_n_probe && dut.cmp_busy && !dut.out_vld) begin
            if (dut.out_hold !== dut.g_q) begin
                if (nbad < 20)
                    $display("  MISMATCH idx=%0d  (prow %0d, pcol %0d)",
                             dut.recv, dut.recv / 12, dut.recv % 12);
                nbad = nbad + 1;
            end
        end
    end
    wire rst_n_probe = dut.rst_n;

    initial begin
        wait (ledg[1] === 1'b1 || cycles > 400000);

        $display("");
        $display("  cycles          : %0d", cycles);
        $display("  results recv    : %0d of %0d", dut.recv, NGOLD);
        $display("  mismatches      : %0d", ledr[7:0]);
        $display("  LEDG[2] PASS    : %b", ledg[2]);
        $display("  LEDG[3] FAIL    : %b", ledg[3]);
        $display("  LEDG[4] vectors : %b", ledg[4]);
        $display("  LEDG[5] overrun : %b", ledg[5]);
        $display("");

        if (ledg[2] === 1'b1 && ledg[3] === 1'b0) begin
            $display("  LAYER HARNESS PASS -- safe to build in Quartus");
        end else begin
            $display("  LAYER HARNESS FAIL");
            $fatal(1);
        end
        $finish;
    end

endmodule

`default_nettype wire
