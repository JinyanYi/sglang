#!/usr/bin/env bash
# Source CANN/ATB before adding framework and optional private MF packages.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../a5_copysfa_mtp/common.sh"
source_ascend_environment
prepend_framework_python
export FRAMEWORK_ROOT KERNEL_ROOT
if [[ -n "${MF_PYTHON_PATH:-}" ]]; then
  export PYTHONPATH="$MF_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}"
fi
mf_package="$("$PYTHON_BIN" -c 'import importlib.util; s=importlib.util.find_spec("memfabric_hybrid"); assert s and s.origin, "Install an A5-compatible memfabric_hybrid wheel"; from pathlib import Path; print(Path(s.origin).parent)')"
export LD_LIBRARY_PATH="$mf_package/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset mf_package
