# ---------------------------------------------------------------------------
# Timing for the stage C4 full-layer self-test.
#
# 50 MHz straight off the oscillator, no PLL -- same reasoning as stage A: a
# PLL is one more thing that can fail to lock, and a failed lock looks exactly
# like a failed layer on the LEDs.
#
# Expect a LOWER Fmax than stage A's 66 MHz. That build was the array alone;
# this adds rowacc (32 entries x 448 bits, read asynchronously at two
# independent indices) and the write-back's requantise-multiply. Those mux
# trees and the 16 MULT_W multipliers are the new long paths. 50 MHz should
# still close comfortably.
# ---------------------------------------------------------------------------

create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]

derive_pll_clocks
derive_clock_uncertainty

# KEY is debounced by the 65k-cycle power-on counter it feeds, and nothing
# samples the LEDs, so neither has a real I/O timing requirement.
set_false_path -from [get_ports {KEY[*]}]
set_false_path -to   [get_ports {LEDG[*]}]
set_false_path -to   [get_ports {LEDR[*]}]
