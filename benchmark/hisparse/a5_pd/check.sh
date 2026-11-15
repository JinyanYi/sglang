#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/environment.sh"
configure_deepep_environment
activate_joint_opp
export MF_HYBM_USE_VMM_SEGMENT=1
cd "$FRAMEWORK_ROOT"
init_stage "a5-pd-probe-${1:-local}"
case "${1:-local}" in
  local)
    run_logged mf-readback "$PYTHON_BIN" "$HERE/probe_mf.py"
    # MF initialize is process-global. Each numerical case gets a fresh process.
    for graph in False True; do
      for width in 1 4; do
        xml="$RUN_DIR/copysfa-$graph-$width.xml"
        run_logged "copysfa-$graph-$width" "$PYTHON_BIN" -m pytest -q \
          "$HERE/test_mf_copysfa.py::test_real_lim_copysfa_mf[$graph-$width]" --junitxml="$xml"
        "$PYTHON_BIN" -c 'import sys, xml.etree.ElementTree as E; cases=E.parse(sys.argv[1]).findall(".//testcase"); assert len(cases)==1 and not any(c.find(tag) is not None for c in cases for tag in ("skipped","failure","error")), "Numerical MF case was skipped or failed"' "$xml"
      done
    done
    ;;
  tp)
    run_logged mf-tp-readback "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node "${PROBE_TP_SIZE:-2}" "$HERE/probe_mf.py"
    ;;
  *) echo "Use check.sh local|tp" >&2; exit 2 ;;
esac
