# ---------------------------------------------------------------------------
# Source this to set SAED14_LIB. Nothing to paste, nothing to fill in.
#
#   source setup_env.sh
#
# Picks the RVT (regular-Vt) typical corner, which is what a 500 MHz target
# wants. HVT leaks less but switches slower; LVT and SLVT are faster but leak
# far more, and leakage is what drains an always-on part's battery.
#
# Override the flavour if you want to compare:
#   SAED14_FLAVOUR=hvt source setup_env.sh
# ---------------------------------------------------------------------------

_saed_root=""
for _r in /pkgs/synopsys/2020/saed14nm /pkgs/synopsys/*/saed14nm; do
    [ -d "$_r" ] && { _saed_root="$_r"; break; }
done

if [ -z "$_saed_root" ]; then
    echo "setup_env.sh: no saed14nm tree found under /pkgs/synopsys."
    echo "  look by hand:  find /pkgs -maxdepth 6 -type d -name 'saed14nm' 2>/dev/null"
else
    _flav="${SAED14_FLAVOUR:-rvt}"
    _dir="$_saed_root/stdcell_${_flav}/db_nldm"

    if [ ! -d "$_dir" ]; then
        echo "setup_env.sh: $_dir does not exist. Available flavours:"
        ls -d "$_saed_root"/stdcell_* 2>/dev/null | sed 's|.*/stdcell_|    |'
    else
        # Typical corner, 0.8 V, 25 C, and it must be the MAIN standard-cell
        # library. The SAED14 kit ships several kinds in the same directory:
        #   saed14rvt_tt0p8v25c.lib            <- logic gates. This one.
        #   saed14rvt_dlvl_tt0p8v25c_*.lib     <- down-level shifters only
        #   saed14rvt_ulvl_tt0p8v25c_*.lib     <- up-level shifters only
        #   saed14rvt_pg_*.lib                 <- power-gating, for UPF flows
        # Picking a dlvl/ulvl file gets you a library with no inverters, which
        # Genus rejects at syn_generic with LBR-171 / LBR-172.
        _lib=$(ls "$_dir"/*tt0p8v25c*.lib 2>/dev/null                | grep -v -E '_pg_|_dlvl_|_ulvl_' | head -1)
        [ -z "$_lib" ] && _lib=$(ls "$_dir"/*tt*.lib 2>/dev/null                | grep -v -E '_pg_|_dlvl_|_ulvl_' | head -1)

        if [ -z "$_lib" ]; then
            echo "setup_env.sh: no typical-corner .lib in $_dir. Contents:"
            ls "$_dir"/*.lib 2>/dev/null | sed 's|.*/|    |'
        else
            export SAED14_LIB="$_lib"
            echo "SAED14_LIB = $SAED14_LIB"
            echo "  flavour   : $_flav   (SAED14_FLAVOUR=hvt to compare)"
            echo "  corner    : typical, 0.80 V, 25 C"
            echo
            echo "Next:  bash smoke_test.sh     then     bash run_all_stages.sh"
        fi
    fi
fi
unset _saed_root _r _flav _dir _lib
