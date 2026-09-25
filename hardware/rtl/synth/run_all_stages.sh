#!/bin/bash
# Staged synthesis for the KWS INT8 accelerator.
#
#   source setup_env.sh
#   bash run_all_stages.sh
#
# NO PARAMETERS ARE PASSED. Genus 17.14 rejects "elaborate -parameters", which
# is what made mac_array, writeback and layer_top fail during bring-up. Every
# RTL default already IS the design point, so plain "elaborate" is correct:
#   mac_array  K=16 M=16 ACC_W=32 PIPE=1
#   writeback  M=16 MAXW=64
#   layer_top  K=16 M=16 ACC_W=28 PIPE=1 MAXW=128 MAX_KTILES=72 BAND_DEPTH=3232
# To try a different shape, edit the default in the .sv file.
set -u
cd "$(dirname "$0")" || exit 1

PERIOD="${CLK_PERIOD_NS:-1.0}"     # 1 GHz is measured: +299 ps, 0 violations

if [ -z "${SAED14_LIB:-}" ] && [ -z "${SAED14_LIBS:-}" ]; then
    echo "Neither SAED14_LIB nor SAED14_LIBS is set."
    echo "Easiest fix, nothing to paste:    source setup_env.sh"
    exit 1
fi
if [ -n "${SAED14_LIB:-}" ] && [ ! -f "$SAED14_LIB" ]; then
    echo "SAED14_LIB=$SAED14_LIB is not a file."; exit 1
fi
if [ -n "${SAED14_LIBS:-}" ] && [ ! -d "$SAED14_LIBS" ]; then
    echo "SAED14_LIBS=$SAED14_LIBS does not exist."; exit 1
fi
command -v genus >/dev/null 2>&1 || {
    echo "genus is not on PATH. Run 'bash check_setup.sh' first."; exit 1; }

stage () {   # stage <top>
    echo
    echo "############################################################"
    echo "###  $1  @  $PERIOD ns"
    echo "############################################################"
    SYN_TOP="$1" CLK_PERIOD_NS="$PERIOD" genus -f run_genus.tcl \
        2>&1 | tee "log_$1_${PERIOD}ns.txt"
    if [ -f "results/$1_${PERIOD}ns/qor.rpt" ]; then
        echo "--- $1 summary ---"
        grep -E "Critical Path Slack|Violating|Leaf Instance|Cell Area" \
             "results/$1_${PERIOD}ns/qor.rpt" 2>/dev/null | sed 's/^/    /'
    fi
}

# Stage 1: the array alone, pure logic. Its timing and area gate every
# remaining decision, and at 60,607 cells it is 92x smaller than the design
# that failed place-and-route in the reference project.
stage mac_array

# Stage 2: the fused write-back. The requantiser multiply is the other
# plausible critical path. It runs at a quarter of the array's rate after
# pooling, so if it is slow it can be time-multiplexed rather than pipelined.
stage writeback

# Stage 3: the whole accelerator. Its AREA is dominated by 376 kbit of memory
# mapped to flip-flops -- see the memory caveat in README.md before quoting it.
stage layer_top

echo
echo "Done. Reports are under results/<top>_${PERIOD}ns/."
echo "Start with qor.rpt: worst slack, violating paths, cell count, area."
