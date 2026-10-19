#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CL_GPU_LOG_DIR="${CL_GPU_LOG_DIR:-${TMPDIR:-/tmp}/cl-gpu-opencl-logs}"
mkdir -p "$CL_GPU_LOG_DIR"
export OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd
export POCL_SIGFPE_HANDLER=0
{
    uname -a
    cat /etc/os-release
    git rev-parse HEAD
    printf 'OpenCL ICD: %s\nPoCL SIGFPE handler: %s\n' "$OCL_ICD_VENDORS" "$POCL_SIGFPE_HANDLER"
    dpkg-query -W sbcl ocl-icd-libopencl1 pocl-opencl-icd
    sbcl --noinform --no-userinit --script scripts/opencl/validate.lisp environment
    clinfo
} 2>&1 | tee "$CL_GPU_LOG_DIR/environment.log"
grep -q 'Portable Computing Language' "$CL_GPU_LOG_DIR/environment.log"
for suite in core runtime; do
    sbcl --noinform --no-userinit --script scripts/opencl/validate.lisp "$suite" \
        2>&1 | tee "$CL_GPU_LOG_DIR/$suite.log"
done
