#!/bin/bash
# ---------------------------------------------------------------------------
# Pre-flight check. Run this BEFORE run_all_stages.sh.
# It changes nothing; it only reports what is and is not ready.
#
#   cd <wherever rtl/ landed>/rtl/synth
#   bash check_setup.sh
# ---------------------------------------------------------------------------

ok ()   { printf '  [ OK ] %s\n' "$1"; }
bad ()  { printf '  [FAIL] %s\n' "$1"; FAILED=1; }
warn () { printf '  [warn] %s\n' "$1"; }
FAILED=0

echo "=============================================================="
echo " 1. Where am I, and did everything copy across?"
echo "=============================================================="
echo "  pwd = $(pwd)"

if [ -f run_genus.tcl ]; then
    ok "running from rtl/synth (run_genus.tcl is here)"
    SYNTH_DIR="."
    RTL_DIR="../rtl_design"
elif [ -f synth/run_genus.tcl ]; then
    warn "you are in rtl/, not rtl/synth -- 'cd synth' first"
    SYNTH_DIR="synth"
    RTL_DIR="rtl_design"
else
    bad "cannot find run_genus.tcl. Are you inside the copied rtl/ folder?"
    echo "       Contents here: $(ls 2>/dev/null | tr '\n' ' ')"
    exit 1
fi

N_SV=$(ls "$RTL_DIR"/*.sv 2>/dev/null | wc -l)
if [ "$N_SV" -eq 7 ]; then
    ok "all 7 RTL files present in $RTL_DIR"
else
    bad "found $N_SV .sv files in $RTL_DIR, expected 7"
    echo "       expected: pe_int8 mac_array band_sram writeback"
    echo "                 axis_result_fifo tile_top layer_top"
fi

echo
echo "=============================================================="
echo " 2. Windows-to-Linux damage (CRLF line endings, lost +x bit)"
echo "=============================================================="
CRLF=0
for f in "$SYNTH_DIR"/*.sh "$SYNTH_DIR"/*.tcl "$SYNTH_DIR"/*.sdc; do
    [ -f "$f" ] || continue
    if grep -qU $'\r' "$f" 2>/dev/null; then
        bad "$f has CRLF line endings -- bash and Genus will both choke"
        CRLF=1
    fi
done
[ "$CRLF" -eq 0 ] && ok "no CRLF line endings found"
if [ "$CRLF" -eq 1 ]; then
    echo "       FIX:  sed -i 's/\r\$//' $SYNTH_DIR/*.sh $SYNTH_DIR/*.tcl $SYNTH_DIR/*.sdc"
fi

if [ -x "$SYNTH_DIR/run_all_stages.sh" ]; then
    ok "run_all_stages.sh is executable"
else
    warn "run_all_stages.sh is not executable (the +x bit does not survive Windows)"
    echo "       FIX:  chmod +x $SYNTH_DIR/run_all_stages.sh"
    echo "       or just run it as:  bash $SYNTH_DIR/run_all_stages.sh"
fi

echo
echo "=============================================================="
echo " 3. Is Genus on the PATH?"
echo "=============================================================="
if command -v genus >/dev/null 2>&1; then
    ok "genus found at $(command -v genus)"
    echo "       version: $(genus -version 2>&1 | head -1)"
else
    bad "genus is not on your PATH"
    echo "       If 'genus -version' works in a fresh shell, it is already"
    echo "       installed and no module is needed. Otherwise try:"
    echo "         source /pkgs/cadence/setup.sh"
    echo "         ls -d /pkgs/cadence* /opt/cadence* 2>/dev/null"
fi

echo
echo "=============================================================="
echo " 4. SAED14nm standard-cell library"
echo "=============================================================="

# Discover the cell flavours that exist on this machine.
FOUND_DIRS=$(ls -d /pkgs/synopsys/*/saed14nm/stdcell_*/db_nldm 2>/dev/null)
# Put rvt first so the recommendation lands on it.
FOUND_DIRS=$(printf '%s
' $FOUND_DIRS | grep rvt; printf '%s
' $FOUND_DIRS | grep -v rvt)
if [ -z "$FOUND_DIRS" ]; then
    # Bounded search only. An unbounded "find /" can take many minutes.
    FOUND_DIRS=$(find /pkgs /opt /tools /cad /usr/local -maxdepth 6                       -type d -name 'db_nldm' -path '*saed14*' 2>/dev/null | head -5)
fi

if [ -n "$FOUND_DIRS" ]; then
    echo "  cell flavours found on this machine:"
    BEST=""
    for d in $FOUND_DIRS; do
        FLAV=$(basename "$(dirname "$d")")
        TT=$(ls "$d"/*tt*.lib 2>/dev/null | grep -v '_pg_' | head -1)
        printf '    %-16s %s
' "$FLAV" "$d"
        [ -n "$TT" ] && printf '        typical corner: %s
' "$(basename "$TT")"
        # Prefer RVT (regular Vt) over HVT (high Vt, low leakage but SLOW),
        # because the target here is 500 MHz, not minimum standby power.
        case "$FLAV" in
            *rvt*) [ -n "$TT" ] && BEST="$TT" ;;
            *hvt*) [ -z "$BEST" ] && [ -n "$TT" ] && BEST="$TT" ;;
            *)     [ -z "$BEST" ] && [ -n "$TT" ] && BEST="$TT" ;;
        esac
    done
    if [ -n "$BEST" ]; then
        echo
        echo "  EASIEST -- nothing to paste:"
        echo
        echo "    source setup_env.sh"
        echo
        echo "  or set it by hand (this is a complete command, copy it whole):"
        echo
        echo "    export SAED14_LIB=$BEST"
        echo
        echo "  Notes: RVT is preferred over HVT for a 500 MHz target; HVT is"
        echo "  lower leakage but slower. Use the tt (typical) corner for the"
        echo "  first run, and never a *_pg_* file -- those are for UPF flows."
    fi
else
    warn "no SAED14nm db_nldm directory found automatically"
    echo "       search by hand (bounded, so it finishes):"
    echo "         find /pkgs -maxdepth 8 -name 'saed14*tt*.lib' 2>/dev/null | head"
fi

echo
if [ -n "${SAED14_LIB:-}" ]; then
    if [ -f "$SAED14_LIB" ]; then ok "SAED14_LIB=$SAED14_LIB"
    else bad "SAED14_LIB=$SAED14_LIB -- file does not exist"; fi
elif [ -n "${SAED14_LIBS:-}" ]; then
    if [ "$SAED14_LIBS" = "/path/to/saed14nm/rvt/lib" ]; then
        bad "SAED14_LIBS is still the placeholder from the README"
    elif [ -d "$SAED14_LIBS" ]; then ok "SAED14_LIBS=$SAED14_LIBS"
    else bad "SAED14_LIBS=$SAED14_LIBS -- directory does not exist"; fi
else
    warn "neither SAED14_LIB nor SAED14_LIBS is set yet (use the line above)"
fi

echo
echo "=============================================================="
if [ "$FAILED" -eq 0 ]; then
    echo " READY. Next:  bash smoke_test.sh     (elaborate only, ~1 min)"
else
    echo " NOT READY -- fix the [FAIL] items above, then re-run this script."
    echo " Note: you can still run smoke_test.sh without the libraries;"
    echo " elaboration does not need them."
fi
echo "=============================================================="
