#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
sudo apt-get update
sudo apt-get install -y sbcl ocl-icd-libopencl1 pocl-opencl-icd clinfo curl ca-certificates build-essential
export CL_GPU_QUICKLISP="${CL_GPU_QUICKLISP:-$HOME/.cache/cl-gpu/quicklisp}"
if [[ ! -f "$CL_GPU_QUICKLISP/setup.lisp" ]]; then
    bootstrap=$(mktemp)
    curl --fail --location https://beta.quicklisp.org/quicklisp.lisp --output "$bootstrap"
    sbcl --noinform --no-userinit --non-interactive --load "$bootstrap" \
        --eval '(quicklisp-quickstart:install :path (sb-ext:posix-getenv "CL_GPU_QUICKLISP"))'
fi
sbcl --noinform --no-userinit --script scripts/opencl/validate.lisp bootstrap
