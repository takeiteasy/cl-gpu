# OpenCL on Linux

Run the core and OpenCL runtime suites on Ubuntu 24.04 with SBCL, the system ICD loader and PoCL's CPU device. The validation requires `cl_khr_fp64` and checks the runnable SAXPY example.

| Validated target | Lisp and runtime |
|---|---|
| GitHub-hosted Ubuntu 24.04 x86-64 | SBCL 2.2.9, PoCL 5.0, ocl-icd 2.3.2 |
| Ubuntu 24.04 ARM64 in Lima on Apple M1 | SBCL 2.2.9, PoCL 5.0, ocl-icd 2.3.2 |

## Ubuntu setup

From a checkout of this repository:

```sh
bash scripts/opencl/setup.sh
bash scripts/opencl/run.sh
```

Setup uses `sudo apt-get` to install SBCL, the ICD loader, PoCL, `clinfo` and build tools. It installs Quicklisp and the Lisp test dependencies without changing your SBCL init file.

| Setting | Default |
|---|---|
| `CL_GPU_QUICKLISP` | `$HOME/.cache/cl-gpu/quicklisp` |
| `CL_GPU_LOG_DIR` | `${TMPDIR:-/tmp}/cl-gpu-opencl-logs` |
| OpenCL implementation | `/etc/OpenCL/vendors/pocl.icd` |
| `POCL_SIGFPE_HANDLER` | `0` during validation |

Set these variables before both commands to use different Quicklisp or log locations. Setup reuses an existing Quicklisp installation; a fresh installation uses the current Quicklisp distribution. The environment log records its version.[^versions]

Validation runs the core suite first and the runtime suite in a separate SBCL process. Logs include the source commit, OS, architecture, package and Lisp versions, OpenCL capabilities, test results and SAXPY output. A missing device, missing f64 extension or failing test exits unsuccessfully.

For direct SBCL sessions with PoCL 5 on Linux x86-64, set `POCL_SIGFPE_HANDLER=0` before starting Lisp. The validation script sets it automatically.

The runtime tests exercise f64 arithmetic and transfers, aligned sub-buffers, unaligned host transfers, static and dynamic local-memory limits, concurrent launches, queued buffer lifetime, error recovery and explicit resource release. Other OpenCL devices skip executed f64 arithmetic when the extension is absent.

## ARM64 Lima VM on macOS

Create a dedicated Ubuntu VM with this checkout mounted read-only. Run the commands from the repository root:

```sh
limactl create --name=cl-gpu-opencl --vm-type=vz --arch=aarch64 \
  --cpus=4 --memory=4 --disk=20 --containerd=none \
  --mount-only="$PWD" --tty=false template:ubuntu-24.04
limactl start --tty=false cl-gpu-opencl
limactl shell --workdir=/tmp cl-gpu-opencl git clone --no-hardlinks \
  "$PWD" /tmp/cl-gpu
limactl shell --workdir=/tmp/cl-gpu cl-gpu-opencl bash scripts/opencl/setup.sh
limactl shell --workdir=/tmp/cl-gpu cl-gpu-opencl bash scripts/opencl/run.sh
```

The guest clone contains committed files. Commit the changes you intend to validate before cloning. Builds, dependencies and logs stay in guest storage; the host checkout stays read-only.

Inspect logs before stopping the VM:

```sh
limactl shell --workdir=/tmp cl-gpu-opencl cat /tmp/cl-gpu-opencl-logs/core.log
limactl shell --workdir=/tmp cl-gpu-opencl cat /tmp/cl-gpu-opencl-logs/runtime.log
limactl stop cl-gpu-opencl
```

The stopped VM remains available for reuse with `limactl start cl-gpu-opencl`.[^guest-storage]

## CI

The **Linux OpenCL** GitHub Actions workflow runs on Ubuntu 24.04 x86-64 with the same scripts. Trigger it manually from Actions or push a tag. Branch pushes and pull requests do not trigger it. Runs on the same ref cancel earlier runs.

The `opencl-linux-x86-64` artifact contains setup, environment and test logs, including logs available when a step fails. Tag pushes run regardless of changed paths.[^tags]

## Troubleshooting

| Symptom | Check |
|---|---|
| No platforms or devices | Run `OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd clinfo`; confirm both `ocl-icd-libopencl1` and `pocl-opencl-icd` are installed |
| Missing `cl_khr_fp64` | Inspect `environment.log`; validation needs PoCL with double-precision support |
| Missing Lisp system | Rerun setup with the same `CL_GPU_QUICKLISP` used for validation |
| Compiler failure | Inspect the runtime log for the OpenCL build log; intentional invalid-source tests also print compiler diagnostics |

## Limitations

- Validation disables PoCL 5's integer divide-by-zero signal handler; kernel integer divisors must be nonzero ([host signal integration](https://github.com/communal-software/cl-gpu/issues/91)).
- Automated Linux coverage uses SBCL on x86-64. The broader SBCL/ECL/CCL and Metal CI matrix remains separate ([CI matrix](https://github.com/communal-software/cl-gpu/issues/32)).
- PoCL CPU validation does not cover vendor GPU drivers. Higher-level operation support is described in [OpenCL limitations](opencl.md#limitations).

[^versions]: The setup does not pin Ubuntu package or Quicklisp versions. Preserve the environment log when reporting results; reuse the recorded Quicklisp distribution and package versions when reproducing a specific environment.
[^tags]: GitHub does not evaluate path filters for tag pushes; see [workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpushpull_requestpull_request_targetpathspaths-ignore).
[^guest-storage]: The example checkout and logs use guest `/tmp`, which is cleared on reboot. Preserve logs on the host and repeat the clone step after restarting. Quicklisp remains in the guest home directory.
