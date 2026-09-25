# ---------------------------------------------------------------------------
# Cadence Genus synthesis for the KWS INT8 accelerator.
#
#   genus -f run_genus.tcl
#
# Driven by environment variables so the same script covers every stage:
#   SYN_TOP        module to synthesise   (default mac_array)
#   CLK_PERIOD_NS  target period          (default 1.0 = 1 GHz, measured to close)
#   SAED14_LIB     one .lib file          (or SAED14_LIBS = a directory)
#
# SYN_PARAMS is accepted but should be left unset: Genus 17.14 rejects
# "elaborate -parameters", and every RTL default already is the design point.
# To try a different shape, edit the default in the .sv file.
#
# SYNTHESISE IN STAGES. See README.md: mac_array first, because it is pure
# logic and its timing and area are the numbers that gate every other decision.
# Running layer_top first buries them under 376 kbit of memory mapped to flops.
# ---------------------------------------------------------------------------

if {[info exists env(SYN_TOP)] && $env(SYN_TOP) ne ""} {
    set TOP $env(SYN_TOP)
} else {
    set TOP "mac_array"
}
if {![info exists env(CLK_PERIOD_NS)]} { set env(CLK_PERIOD_NS) 1.0 }
set PERIOD  $env(CLK_PERIOD_NS)
if {[info exists env(SYN_PARAMS)]} {
    set PARAMS $env(SYN_PARAMS)
} else {
    set PARAMS ""
}

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set RTL     [file join $SCRIPT_DIR .. rtl_design]
set OUTDIR  [file join $SCRIPT_DIR results "${TOP}_${PERIOD}ns"]
file mkdir $OUTDIR

puts "=========================================================="
puts "  top = $TOP   period = $PERIOD ns   params = $PARAMS"
puts "=========================================================="

# --- library --------------------------------------------------------------
# Load exactly ONE corner. Globbing *.lib pulls in every corner plus the
# power-ground files, which a single-corner run must not do.
#
#   SAED14_LIB   full path to one .lib          (wins if set)
#   SAED14_LIBS  directory; the TT corner is picked out of it automatically
#
set LIBFILE ""
if {[info exists env(SAED14_LIB)] && $env(SAED14_LIB) ne ""} {
    set LIBFILE $env(SAED14_LIB)
} elseif {[info exists env(SAED14_LIBS)] && $env(SAED14_LIBS) ne ""} {
    set dir $env(SAED14_LIBS)
    # Typical-typical, and never a pg (power-ground) file.
    set cands {}
    foreach f [lsort [glob -nocomplain -directory $dir *.lib]] {
        set b [file tail $f]
        if {[string match "*_pg_*" $b]} { continue }
        if {[string match "*tt*" $b]}   { lappend cands $f }
    }
    if {[llength $cands] > 0} {
        set LIBFILE [lindex $cands 0]
    } else {
        puts "ERROR: no typical-corner .lib found in $dir"
        puts "       files there:"
        foreach f [glob -nocomplain -directory $dir *.lib] { puts "         [file tail $f]" }
        exit 1
    }
} else {
    puts "ERROR: set SAED14_LIBS (a directory) or SAED14_LIB (one .lib file)."
    exit 1
}

if {![file exists $LIBFILE]} {
    puts "ERROR: library not found: $LIBFILE"
    exit 1
}

puts "  library : $LIBFILE"
if {[catch {read_libs $LIBFILE} err]} {
    puts "ERROR: read_libs failed: $err"
    exit 1
}

# Multi-threading. 'catch' because the attribute name varies by release;
# a wrong name would otherwise abort the whole run.
catch {set_db max_cpus_per_server 8}
catch {set_db super_thread_servers {localhost}}

set_db hdl_error_on_blackbox true
set_db hdl_max_loop_limit 8192      ;# generate loops up to 32x32 arrays
set_db syn_generic_effort high
set_db syn_map_effort     high
set_db syn_opt_effort     high

# --- low power ------------------------------------------------------------
# Attribute names drift between Genus releases, and an unknown attribute aborts
# the entire run. Set each one inside a catch and report what actually took, so
# a naming difference costs a warning rather than the whole synthesis.
proc try_db {attr val} {
    if {[catch {set_db $attr $val} err]} {
        puts "  \[skip\] set_db $attr $val   ($err)"
        return 0
    }
    puts "  \[ ok \] set_db $attr $val"
    return 1
}

puts ""
puts "--- low-power settings ---"
# CLOCK GATING. The array holds ~15k flops and large parts of it idle during
# weight loads, drains and flushes. Genus inserts the enable logic itself, so
# this is a switch rather than an RTL change, and it is the largest dynamic
# power lever available before layout.
try_db lp_insert_clock_gating true
try_db lp_clock_gating_min_flops 4
try_db lp_clock_gating_prefix CG_

# OPERAND ISOLATION. Between 14% and 51% of post-ReLU activations are zero.
# A lockstep systolic array cannot skip those cycles -- no aligned 16-tap
# vector was ever entirely zero -- but it can stop them switching.
try_db lp_insert_operand_isolation true
try_db lp_operand_isolation_logic and
puts ""

# --- read ----------------------------------------------------------------
read_hdl -sv [list \
    $RTL/pe_int8.sv \
    $RTL/mac_array.sv \
    $RTL/band_sram.sv \
    $RTL/writeback.sv \
    $RTL/axis_result_fifo.sv \
    $RTL/tile_top.sv \
    $RTL/layer_top.sv ]

if {$PARAMS ne ""} {
    elaborate -parameters $PARAMS $TOP
} else {
    elaborate $TOP
}

current_design $TOP
read_sdc [file join $SCRIPT_DIR constraints.sdc]
check_design -unresolved

# --- synthesise ----------------------------------------------------------
syn_generic
syn_map
syn_opt

# --- reports -------------------------------------------------------------
# EVERY report is wrapped in catch. These run last, after the synthesis is
# already done, so a single bad flag must never cost the whole run -- which it
# did twice: "-verbose" needs "-lint", and "-lint" refuses every other option.
catch {report_timing -max_paths 20  > $OUTDIR/timing.rpt}
catch {report_timing -lint          > $OUTDIR/timing_lint.rpt}
catch {report_area                  > $OUTDIR/area.rpt}
catch {report_area -depth 3         > $OUTDIR/area_hier.rpt}
catch {report_power                 > $OUTDIR/power.rpt}
catch {report_gates                 > $OUTDIR/gates.rpt}
catch {report_qor                   > $OUTDIR/qor.rpt}
catch {report_clock_gating          > $OUTDIR/clock_gating.rpt}

write_hdl                              > $OUTDIR/${TOP}_netlist.v
write_sdc                              > $OUTDIR/${TOP}.sdc

puts "\n=========================================================="
puts "  WNS / TNS summary"
puts "=========================================================="
report_timing -max_paths 1
puts "\nReports written to $OUTDIR"

# Genus stays in its interactive shell after a -f script. In a piped run
# (run_all_stages.sh pipes through tee) that hangs waiting on stdin, so exit
# explicitly.
exit
