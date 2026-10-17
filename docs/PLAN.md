# Plan

cl-gpu compiles Lisp kernel forms to Metal source and runs them on device buffers. It supersedes the GPU-array role of libgpuarray for Common Lisp. CUDA and OpenCL backends are planned.

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
| GPU BLAS Level 1 ([done](blas.md)), parallel to trivial-simd/blas | [#19](https://github.com/communal-software/cl-gpu/issues/19) |
| GPU `gemv` and `gemm`, with an MPS fast path ([done](blas-matrices.md)) | [#37](https://github.com/communal-software/cl-gpu/issues/37) |
| Constant-stride batched `gemm` ([done](blas-matrices.md#batched-gemm)) | [#38](https://github.com/communal-software/cl-gpu/issues/38) |
| Queued launches with batched command submission ([done](kernels.md#queued-launches)) | [#42](https://github.com/communal-software/cl-gpu/issues/42) |
| Metal stream: tuned commit interval, waiting only on the buffers in use, per-thread streams, staging ring for private uploads | [#48](https://github.com/communal-software/cl-gpu/issues/48), [#49](https://github.com/communal-software/cl-gpu/issues/49), [#50](https://github.com/communal-software/cl-gpu/issues/50), [#51](https://github.com/communal-software/cl-gpu/issues/51) |
| MPS for broadcast and negative-stride batched `gemm` | [#47](https://github.com/communal-software/cl-gpu/issues/47) |
| f16/bf16 Level 1 and the mixed dots ([done](blas.md#element-types)) | [#40](https://github.com/communal-software/cl-gpu/issues/40) |
| Complex routines; f64 routines; rotations | [#39](https://github.com/communal-software/cl-gpu/issues/39), [#55](https://github.com/communal-software/cl-gpu/issues/55), [#41](https://github.com/communal-software/cl-gpu/issues/41) |
| Sub-array views of a `gpu-array` ([done](kernels.md#views)), for cl-tensor storage | [#57](https://github.com/communal-software/cl-gpu/issues/57) |
| MLX for cl-inference, independent of cl-gpu | [cl-inference#43](https://github.com/communal-software/cl-inference/issues/43) |
| Embeddable Metal header, body and access descriptors | [Available](kernel-parts.md) |
| Strided axis reductions, softmax and RMSNorm | [Available](reductions.md) |

## Limitations

Kernels print as Metal source and run through [`define-kernel`](kernels.md) on the [Metal runtime](metal.md). There are no CUDA or OpenCL printers.

- Tested on SBCL, ECL and CCL; no CI matrix yet ([#32](https://github.com/communal-software/cl-gpu/issues/32)). Other implementations: [#33](https://github.com/communal-software/cl-gpu/issues/33)–[#36](https://github.com/communal-software/cl-gpu/issues/36).
- No `loop`, `do`, `tagbody` or `go` ([kernel forms](kernel-forms.md)).
- Swizzles are read-only ([#22](https://github.com/communal-software/cl-gpu/issues/22)).
- Vectors cannot be compared, and `min`, `max`, `clamp`, `pow` and `fma` do not broadcast a scalar ([#23](https://github.com/communal-software/cl-gpu/issues/23)).
- CUDA needs operators and broadcast for vector types ([#12](https://github.com/communal-software/cl-gpu/issues/12)).
- Kernel errors have no source location when compiled from an editor buffer with `C-c C-c` ([#27](https://github.com/communal-software/cl-gpu/issues/27)).
- Metal reports a fixed 256-byte alignment for `:constant` view bindings ([#72](https://github.com/communal-software/cl-gpu/issues/72)); OpenCL needs its own ([#73](https://github.com/communal-software/cl-gpu/issues/73)).
- Kernel pipelines of released devices stay cached ([#18](https://github.com/communal-software/cl-gpu/issues/18)).
- Tensor device dispatch awaits cl-tensor storage and adapter work ([cl-tensor#12](https://github.com/communal-software/cl-tensor/issues/12), [cl-tensor#21](https://github.com/communal-software/cl-tensor/issues/21)). Reduction-specific limits are listed on the [reductions](reductions.md#limitations) page.
