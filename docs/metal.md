# Metal runtime

`cl-gpu/metal` compiles Metal source and runs it on the GPU. It is the layer under [`define-kernel`](kernels.md), which most code uses instead. It needs macOS and Xcode's command line tools; loading the system builds a small Objective-C shim with `clang`.

```lisp
(asdf:load-system "cl-gpu/metal")

(let ((source (cl-gpu:kernel-source kernel :metal))   ; the lowered saxpy from printers.md
      (device (cl-gpu/metal:default-device)))
  (cl-gpu/metal:with-library (library device source)
    (cl-gpu/metal:with-pipeline (pipeline library "saxpy")
      (cl-gpu/metal:with-buffer (out device 4000)
        (cl-gpu/metal:with-buffer (x device 4000)
          (cl-gpu/metal:with-buffer (y device 4000)
            (cl-gpu/metal:copy-to-buffer x (make-array 1000 :initial-element 1.0))
            (cl-gpu/metal:copy-to-buffer y (make-array 1000 :initial-element 2.0))
            (cl-gpu/metal:dispatch pipeline
                                   (list out x y '(:bytes cl-gpu:f32 3.0) '(:bytes cl-gpu:u32 1000))
                                   1000)
            (cl-gpu/metal:copy-from-buffer out (make-array 1000))))))))
```

## API

| Function | Result |
|---|---|
| `(default-device)` | the system GPU; `device-name`, `device-max-threads`, `device-max-threadgroup-memory` (bytes) |
| `(make-buffer device bytes)` | a zeroed buffer the host and GPU share[^1] |
| `(copy-to-buffer buffer sequence &key element-type)` | stores numbers from the buffer's start; `element-type` is a scalar or vector type, default `f32` |
| `(copy-from-buffer buffer sequence &key element-type)` | fills `sequence` from the buffer's start |
| `(buffer-pointer buffer)`, `(buffer-size buffer)` | host address and size in bytes |
| `(compile-library device source)` | a library; a compile error signals `metal-compile-error` with the compiler's log |
| `(make-pipeline library name)` | the pipeline for kernel function `name`; `pipeline-max-threads`, `pipeline-execution-width` |
| `(dispatch pipeline arguments grid &key group wait)` | queues the kernel and returns `nil`; with `wait`, then waits for the device |
| `(release object)` | frees the object; releasing twice is harmless |

`with-buffer`, `with-library` and `with-pipeline` take the same arguments as their constructors, bind the result and release it on exit.

## Dispatch

`grid` is a thread count or a list of up to three. It need not divide by the group size.[^2] `group` defaults to the pipeline's execution width, capped at the grid.

`arguments` follow the kernel's parameter order:

| Argument | Binds |
|---|---|
| a buffer | the next `[[buffer(i)]]` |
| `(:bytes type value)` | the next `[[buffer(i)]]`, a scalar or vector copied by value |
| `(:threadgroup bytes)` | the next `[[threadgroup(k)]]`, for a `:shared` pointer; `bytes` is rounded up to a multiple of 16 |

`type` is a scalar or `(vec T n)`; a vector value is a sequence of its components. A 3-vector takes the space of a 4-vector, as in Metal.

`copy-to-buffer` and `copy-from-buffer` take the components of vector elements consecutively in `sequence`. A 3-vector takes the space of a 4-vector in the buffer, and the padding is left alone.

## Streams

Each device has one stream. `dispatch` encodes into the stream's open command buffer, which is committed after 64 dispatches and when the device is waited on. Dispatches and MPS products run in the order they were queued. The stream is thread-safe: launches from several threads interleave in one command buffer.

| Call | Waits for the device's pending work |
|---|---|
| `dispatch` with `:wait t` | yes |
| `buffer-pointer`, `copy-to-buffer`, `copy-from-buffer` | yes, before touching the buffer |
| `release` of a device | yes, then frees the stream |
| `release` of a buffer | no; pending work keeps the buffer alive |

`cl-gpu:synchronize` waits explicitly. A failed command signals `metal-error` from the first wait after it.

## MetalPerformanceShaders

The shim links MetalPerformanceShaders. With `cl-gpu/blas` loaded, `device-gemm` and `device-gemv` on a Metal device call it where the layouts allow, and bind `*use-mps*` to `nil` to use the DSL kernels instead. See [BLAS matrices](blas-matrices.md#metal-and-mps).

## Limitations

- The stream commits after a fixed 64 dispatches ([#48](https://github.com/communal-software/cl-gpu/issues/48)).
- Host access waits for the whole device, not only the dispatches that use the buffer ([#49](https://github.com/communal-software/cl-gpu/issues/49)).
- One lock guards each device's stream ([#50](https://github.com/communal-software/cl-gpu/issues/50)).

## Errors

`metal-error` is signalled for a missing device, a released object, a transfer that overflows the buffer, an unknown kernel name or a failed command. `metal-compile-error` is its subclass for source that does not compile. Both carry `metal-error-message`.

[^1]: Buffers use shared storage, so `buffer-pointer` reads and writes the same memory the GPU sees.
[^2]: `dispatch` uses Metal's non-uniform threadgroups, available on Apple silicon GPUs.
