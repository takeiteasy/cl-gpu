# Plan

cl-gpu compiles Lisp kernel forms to Metal or OpenCL source and runs them on device buffers. It supersedes the GPU-array role of libgpuarray for Common Lisp. The CUDA backend is planned.

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
| 7 | [OpenCL backend](opencl.md) | | Done |

## Order

The core Metal and OpenCL backends are available. CUDA remains deferred.

| Backend | Testing without a GPU of that vendor |
|---|---|
| OpenCL | Apple's OpenCL 1.2 runs on the Mac, and [PoCL](https://portablecl.org) (`brew install pocl`, `pocl-opencl-icd` on Linux) is a CPU implementation of OpenCL 3.0. Apple's lacks `f64` on Apple silicon and OpenCL 2.0 atomics. |
| CUDA | No printer or runtime is implemented; the backend is deferred. |

The [OpenCL runtime](opencl.md) implements the same device protocol as Metal. [Linux validation](opencl-linux.md) covers SBCL with the ICD loader and PoCL; its x86-64 CI runs manually and on tags.

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
| Broadcast elementwise operations and dtype conversion | [Available](elementwise.md) |
| Native f64 strided arithmetic, conversion, reductions and normalization | [Available](native-f64.md) |
| Reusable fused expressions over strided device views | [Available](expressions.md) |
| Checked axis gather over strided device views | [Available](gather.md) |

## Limitations

Kernels print as Metal or OpenCL source and run through [`define-kernel`](kernels.md). There is no CUDA printer. OpenCL operation and platform limits are listed in [OpenCL limitations](opencl.md#limitations).

- Tested on SBCL, ECL and CCL; Linux OpenCL CI uses SBCL. The broader Lisp and Metal CI matrix remains pending ([#32](https://github.com/communal-software/cl-gpu/issues/32)). Other implementations: [#33](https://github.com/communal-software/cl-gpu/issues/33)–[#36](https://github.com/communal-software/cl-gpu/issues/36).
- No `loop`, `do`, `tagbody` or `go` ([kernel forms](kernel-forms.md)).
- Swizzles are read-only ([#22](https://github.com/communal-software/cl-gpu/issues/22)).
- Vectors cannot be compared, and `min`, `max`, `clamp`, `pow` and `fma` do not broadcast a scalar ([#23](https://github.com/communal-software/cl-gpu/issues/23)).
- CUDA needs operators and broadcast for vector types ([#12](https://github.com/communal-software/cl-gpu/issues/12)).
- Kernel errors have no source location when compiled from an editor buffer with `C-c C-c` ([#27](https://github.com/communal-software/cl-gpu/issues/27)).
- Metal reports a fixed 256-byte alignment for `:constant` view bindings ([#72](https://github.com/communal-software/cl-gpu/issues/72)).
- Kernel pipelines of released devices stay cached ([#18](https://github.com/communal-software/cl-gpu/issues/18)).
- Tensor device dispatch supports storage, elementwise operations, conversion, reductions, normalization and f32 matrix operations. Tensor expression integration is tracked in [cl-tensor#26](https://github.com/communal-software/cl-tensor/issues/26). Reduction-specific limits are listed on the [reductions](reductions.md#limitations) page.
