# ---------------------------------------------------------------------------
# Timing constraints for the KWS INT8 accelerator.
#
# The period comes from the CLK_PERIOD_NS environment variable, so you do NOT
# normally edit this file:
#
#     CLK_PERIOD_NS=1.0 SYN_TOP=mac_array genus -f run_genus.tcl
#
# DEFAULT_PERIOD_NS below is only the fallback when that variable is unset.
# Change it if you would rather fix the target here than on the command line.
#
#   2.0 ns =  500 MHz   MEASURED: +993 ps slack, 0 violating paths
#   1.0 ns = 1000 MHz   MEASURED: +299 ps slack, 0 violating paths  <- default
#   0.7 ns = 1430 MHz   projected +104 ps from the measured 347 ps data path
# ---------------------------------------------------------------------------

set DEFAULT_PERIOD_NS 1.0

if {[info exists ::env(CLK_PERIOD_NS)] && $::env(CLK_PERIOD_NS) ne ""} {
    set clk_period $::env(CLK_PERIOD_NS)
} else {
    set clk_period $DEFAULT_PERIOD_NS
}
puts "  constraints.sdc: clock period $clk_period ns ([expr {1000.0/$clk_period}] MHz)"

create_clock -name clk -period $clk_period [get_ports clk]

# A laptop-class SoC interface; tighten once the host side is real.
set_clock_uncertainty [expr {$clk_period * 0.05}] [get_clocks clk]
set_clock_transition  [expr {$clk_period * 0.02}] [get_clocks clk]

# Reset is asynchronous and externally synchronised.
set_false_path -from [get_ports rst_n]

# Budget 30% of the period either side of the block boundary. This is why every
# reported worst path is input-port to first register rather than register to
# register: at 1.0 ns the budget is 300 ps against only 347 ps of real logic.
# Tighten it once the host interface is defined.
set io_delay [expr {$clk_period * 0.30}]
set all_in  [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]
set_input_delay  -clock clk $io_delay $all_in
set_output_delay -clock clk $io_delay [all_outputs]

set_load 0.010 [all_outputs]
set_max_fanout 32 [current_design]
