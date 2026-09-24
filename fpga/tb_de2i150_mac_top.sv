// ---------------------------------------------------------------------------
// tb_de2i150_mac_top -- simulate the FPGA harness itself.
//
// The harness reports on eight LEDs. If it is wrong, a broken harness and a
// broken array light the same LED, and the board gives no way to tell them
// apart. So run the harness in Icarus first, against the same .txt vectors
// that become the .mif ROMs, and only then open Quartus.
//
// The POR counter is overridden to 63 so the test does not spend 65,536
// cycles waiting for a reset that only exists for real power rails.
//
//   iverilog -g2012 -s tb_de2i150_mac_top -o /tmp/h.vvp \
//       fpga/tb_de2i150_mac_top.sv fpga/de2i150_mac_top.sv \
//       rtl/rtl_design/mac_array.sv rtl/rtl_design/pe_int8.sv
//   vvp /tmp/h.vvp
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps
`default_nettype none

module tb_de2i150_mac_top;

    localparam int NVEC = 64;

    reg         clk = 1'b0;
    reg  [3:0]  key = 4'hF;
    wire [8:0]  ledg;
    wire [17:0] ledr;

    always #10 clk = ~clk;                       // 50 MHz, like CLOCK_50

    de2i150_mac_top #(
        .NVEC (NVEC),
        .WMIF ("fpga/weights.txt"),
        .AMIF ("fpga/acts.txt"),
        .EMIF ("fpga/expected.txt")
    ) dut (
        .CLOCK_50 (clk),
        .KEY      (key),
        .LEDG     (ledg),
        .LEDR     (ledr)
    );

    // Skip the power-on reset wait; it exists for real rails, not simulation.
    initial begin
        @(negedge clk);
        dut.por_cnt = 16'hFFFE;
    end

    integer cycles = 0;
    always @(posedge clk) cycles = cycles + 1;

    // Guard against a vacuous pass.
    //
    // If the ROMs never got loaded they read X, the array produces X, and the
    // checker compares X !== X -- which is FALSE, so nothing is flagged and
    // every LED says PASS on a design that computed nothing. This is not
    // hypothetical: Quartus hit exactly this when the ROMs were initialised
    // by a ram_init_file attribute instead of $readmemb.
    //
    // A test that reports success when it was never given any data to check is
    // worse than no test, so prove the golden data is real before trusting the
    // verdict that rests on it.
    initial begin
        #1;
        if (^dut.erom[0] === 1'bx || ^dut.arom[0] === 1'bx
                                  || ^dut.wrom[0] === 1'bx) begin
            $display("");
            $display("  VECTOR ROMS ARE X -- nothing was loaded.");
            $display("  Compile with -DSIMULATION, and run from the repo root");
            $display("  so fpga/*.txt resolve. Any PASS here would be vacuous.");
            $fatal(1);
        end
    end

    initial begin
        // Generous bound: LOAD + COMMIT + SETTLE + NVEC + DRAIN is under 200.
        wait (ledg[1] === 1'b1 || cycles > 5000);

        $display("");
        $display("  cycles        : %0d", cycles);
        $display("  results recv  : %0d of %0d", ledr[15:8], NVEC);
        $display("  mismatches    : %0d", ledr[7:0]);
        $display("  LEDG[1] done  : %b", ledg[1]);
        $display("  LEDG[2] PASS  : %b", ledg[2]);
        $display("  LEDG[3] FAIL  : %b", ledg[3]);
        $display("");

        if (ledg[2] === 1'b1 && ledg[3] === 1'b0) begin
            $display("  HARNESS PASS -- safe to build in Quartus");
        end else begin
            $display("  HARNESS FAIL -- fix this before touching the board");
            $fatal(1);
        end
        $finish;
    end

endmodule

`default_nettype wire
