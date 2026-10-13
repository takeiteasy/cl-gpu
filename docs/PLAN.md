# Plan

cl-gpu compiles Lisp kernel forms to Metal, CUDA and OpenCL source and runs them on device buffers. It supersedes the GPU-array role of libgpuarray for Common Lisp. [cl-tensor](https://github.com/communal-software/cl-tensor) uses it for device storage.

## Design

| Part | Choice |
|---|---|
| Reader | Eclector |
| Walker | New, compute-only: fixed special forms plus kernel macroexpansion, walking CSTs into AST nodes |
| Types | Closed set: scalars, fixed vectors, typed pointers with address spaces |
| IR | Small C-like tree mapping near 1:1 onto MSL, CUDA C and OpenCL C |
| Printers | Shared C-like base with per-target overrides |
| First target | Metal (testable locally on Apple Silicon) |

The cl-webgpu shader DSL (a 3bgl-shaders fork) informs the inference tests but contributes no code.

## Phases

| # | Deliverable | Ticket | Status |
|---|---|---|---|
| 0 | Skeleton | | Done |
| 1 | Eclector reader and compute-subset walker | [#1](https://github.com/communal-software/cl-gpu/issues/1) | Done |
| 2 | Type set and kernel IR | [#2](https://github.com/communal-software/cl-gpu/issues/2) | Done |
| 3 | C-like printer base and MSL printer | [#3](https://github.com/communal-software/cl-gpu/issues/3) | Done |
| 4 | Metal runtime | [#4](https://github.com/communal-software/cl-gpu/issues/4) | Done |
| 5 | `define-kernel` and buffer API | [#5](https://github.com/communal-software/cl-gpu/issues/5) | Done |
| 6 | CUDA backend | [#6](https://github.com/communal-software/cl-gpu/issues/6) | Deferred |
| 7 | OpenCL backend | [#7](https://github.com/communal-software/cl-gpu/issues/7) | Deferred |

## Order

The kernel DSL and the Metal runtime stabilise first: the open `dsl`, `arch` and `api` tickets (#8–#18) come before CUDA and OpenCL, which are low priority.

| Backend | Testing without a GPU of that vendor |
|---|---|
| OpenCL | Apple's OpenCL 1.2 runs on the Mac, and [PoCL](https://portablecl.org) (`brew install pocl`, `pocl-opencl-icd` on Linux) is a CPU implementation of OpenCL 3.0. Apple's lacks `f64` on Apple silicon and OpenCL 2.0 atomics. |
| CUDA | Printer snapshot tests need nothing. `nvcc -ptx` or NVRTC checks that generated source compiles on a Linux machine with the toolkit, without a GPU. Running kernels needs a rented GPU, on `workflow_dispatch` only. |

OpenCL goes first when the backends start, as the cheaper second check that the backend protocol in `device.lisp` is not shaped around Metal.

## Later

| Idea | Ticket |
|---|---|
| GPU BLAS, parallel to trivial-simd/blas: DSL kernels with vendor fast paths | [#19](https://github.com/communal-software/cl-gpu/issues/19) |
| MLX for cl-inference, independent of cl-gpu | [cl-inference#43](https://github.com/communal-software/cl-inference/issues/43) |
| Body-only kernel output, so MLX can run DSL kernels as custom kernels | [#20](https://github.com/communal-software/cl-gpu/issues/20), [cl-inference#44](https://github.com/communal-software/cl-inference/issues/44) |

## Limitations

Kernels print as Metal source and run through [`define-kernel`](kernels.md) on the [Metal runtime](metal.md). There are no CUDA or OpenCL printers.

- No `loop`, `do`, `tagbody` or `go` ([kernel forms](kernel-forms.md)).
- Shared memory is not checked against the device limit ([#24](https://github.com/communal-software/cl-gpu/issues/24)).
- Swizzles are read-only ([#22](https://github.com/communal-software/cl-gpu/issues/22)).
- Vectors cannot be compared, and `min`, `max`, `clamp`, `pow` and `fma` do not broadcast a scalar ([#23](https://github.com/communal-software/cl-gpu/issues/23)).
- CUDA needs operators and broadcast for vector types ([#12](https://github.com/communal-software/cl-gpu/issues/12)).
- Errors in `define-kernel` bodies have no source location ([#16](https://github.com/communal-software/cl-gpu/issues/16)).
- Kernel pipelines of released devices stay cached ([#18](https://github.com/communal-software/cl-gpu/issues/18)).
- Buffers use shared storage only ([#15](https://github.com/communal-software/cl-gpu/issues/15)).
