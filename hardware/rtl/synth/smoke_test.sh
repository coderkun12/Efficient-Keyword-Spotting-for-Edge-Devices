#!/bin/bash
# Elaborate every module, one Genus invocation each.
#
# Genus discards the parsed HDL after the first elaborate, so looping inside a
# single session fails with CDFG-210. One process per module is the fix.
cd "$(dirname "$0")" || exit 1

command -v genus >/dev/null 2>&1 || { echo "genus not on PATH"; exit 1; }
if [ -z "${SAED14_LIB:-}" ] && [ -z "${SAED14_LIBS:-}" ]; then
    echo "No library set. Run:  source setup_env.sh"; exit 1
fi

# Self-heal: if the .tcl was pasted into an editor, the shell heredoc wrapper
# ends up inside the file. Strip it rather than failing cryptically.
if head -1 smoke_test.tcl 2>/dev/null | grep -q '^cat >'; then
    echo "note: stripping heredoc wrapper from smoke_test.tcl"
    sed -i "/^cat > smoke_test.tcl/d; /^GENUS_EOF\$/d; /^TCLEOF\$/d" smoke_test.tcl
fi

fails=0
run () {
    printf '%-12s ' "$1"
    SMOKE_TOP="$1" SMOKE_PARAMS="$2" genus -f smoke_test.tcl > "smoke_$1.log" 2>&1
    if grep -q "SMOKE OK" "smoke_$1.log"; then
        echo "OK"
    else
        echo "FAILED   (smoke_$1.log)"
        grep -m4 -E '^Error|SMOKE FAIL' "smoke_$1.log" | sed 's/^/               /'
        fails=$((fails + 1))
    fi
}

# No parameters. Genus 17.14 rejects "elaborate -parameters" -- passing them
# is what made mac_array, writeback and layer_top fail during bring-up while
# the two parameterless modules passed. Every RTL default is the design point.
echo "=== elaborating each module, smallest first ==="
run pe_int8   ""
run mac_array ""
run writeback ""
run band_sram ""
run layer_top ""

echo
if [ "$fails" -eq 0 ]; then
    echo "SMOKE TEST PASSED -- all 5 modules elaborated."
    echo "Next:  bash run_all_stages.sh"
else
    echo "SMOKE TEST FAILED -- $fails module(s) did not elaborate."
fi
exit $fails
