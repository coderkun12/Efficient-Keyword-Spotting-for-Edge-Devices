# ---------------------------------------------------------------------------
# Timing constraints for the DE2i-150 self-test.
#
# Bring-up runs straight off the 50 MHz oscillator with no PLL. That is a
# deliberate choice: a PLL is one more thing that can fail to lock, and a
# failed lock looks exactly like a failed array on the LEDs. Get a PASS at
# 50 MHz first, then add the PLL to find fmax.
#
# The 14 nm synthesis closed at 1 GHz. Cyclone IV GX is a 60 nm-class part
# with no accumulator in its DSP blocks, so the psum adder lands in LE carry
# chains; expect fmax somewhere around 80-120 MHz. 50 MHz should close with
# room to spare, and if it does not, the report will say which path is long.
# ---------------------------------------------------------------------------

create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]

derive_pll_clocks
derive_clock_uncertainty

# KEY and the LEDs are asynchronous to everything. KEY[0] is debounced by the
# 65k-cycle power-on counter it feeds, and nothing samples the LEDs, so there
# is no real I/O timing requirement on either.
set_false_path -from [get_ports {KEY[*]}]
set_false_path -to   [get_ports {LEDG[*]}]
set_false_path -to   [get_ports {LEDR[*]}]
