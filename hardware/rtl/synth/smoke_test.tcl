# Smoke test for ONE module: load the library, read the RTL, elaborate.
#
# Genus discards the parsed-but-unelaborated HDL once you elaborate a top, so
# a second "elaborate" in the same session fails with CDFG-210 "Could not find
# an HDL design". One module per Genus invocation is the only reliable way.
# smoke_test.sh drives this once per module.
#
#   SMOKE_TOP     module to elaborate   (default pe_int8)
#   SMOKE_PARAMS  parameters            (e.g. "K 16 M 16 PIPE 1")

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set RTL [file join $SCRIPT_DIR .. rtl_design]

if {[info exists env(SMOKE_TOP)] && $env(SMOKE_TOP) ne ""} {
    set TOP $env(SMOKE_TOP)
} else {
    set TOP "pe_int8"
}
if {[info exists env(SMOKE_PARAMS)]} {
    set PARAMS $env(SMOKE_PARAMS)
} else {
    set PARAMS ""
}

# Genus needs a target library even to ELABORATE (LBR-163). Load exactly ONE
# corner: globbing *.lib pulls in every corner plus the _pg_ power-ground
# files, which are for UPF flows and are not a synthesis target.
set LIBFILE ""
if {[info exists env(SAED14_LIB)] && $env(SAED14_LIB) ne ""} {
    set LIBFILE $env(SAED14_LIB)
} elseif {[info exists env(SAED14_LIBS)] && $env(SAED14_LIBS) ne ""} {
    foreach f [lsort [glob -nocomplain -directory $env(SAED14_LIBS) *.lib]] {
        set b [file tail $f]
        if {[string match "*_pg_*" $b]} { continue }
        if {[string match "*tt0p8v25c*" $b]} { set LIBFILE $f ; break }
        if {$LIBFILE eq "" && [string match "*tt*" $b]} { set LIBFILE $f }
    }
}
if {$LIBFILE eq "" || ![file exists $LIBFILE]} {
    puts "SMOKE FAIL: no technology library. Run: source setup_env.sh"
    exit 1
}

if {[catch {read_libs $LIBFILE} err]} {
    puts "SMOKE FAIL: read_libs: $err"
    exit 1
}

set_db hdl_max_loop_limit 8192

if {[catch {
    read_hdl -sv [list \
        [file join $RTL pe_int8.sv] \
        [file join $RTL mac_array.sv] \
        [file join $RTL band_sram.sv] \
        [file join $RTL writeback.sv] \
        [file join $RTL axis_result_fifo.sv] \
        [file join $RTL tile_top.sv] \
        [file join $RTL layer_top.sv] ]
} err]} {
    puts "SMOKE FAIL: read_hdl: $err"
    exit 1
}

if {[catch {
    if {$PARAMS eq ""} {
        elaborate $TOP
    } else {
        elaborate -parameters $PARAMS $TOP
    }
} err]} {
    puts "SMOKE FAIL: elaborate $TOP: $err"
    exit 1
}

puts "SMOKE OK: $TOP"
exit 0
