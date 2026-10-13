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
| `(dispatch pipeline arguments grid &key group wait)` | runs the kernel and returns a command |
| `(wait command)` | blocks until the command finishes; `metal-error` if it failed |
| `(command-done-p command)` | whether the command has finished |
| `(release object)` | frees the object; releasing twice is harmless |

`with-buffer`, `with-library` and `with-pipeline` take the same arguments as their constructors, bind the result and release it on exit.

## Dispatch

`grid` is a thread count or a list of up to three. It need not divide by the group size.[^2] `group` defaults to the pipeline's execution width, capped at the grid. `dispatch` waits unless `:wait nil`.

`arguments` follow the kernel's parameter order:

| Argument | Binds |
|---|---|
| a buffer | the next `[[buffer(i)]]` |
| `(:bytes type value)` | the next `[[buffer(i)]]`, a scalar or vector copied by value |
| `(:threadgroup bytes)` | the next `[[threadgroup(k)]]`, for a `:shared` pointer; `bytes` is rounded up to a multiple of 16 |

`type` is a scalar or `(vec T n)`; a vector value is a sequence of its components. A 3-vector takes the space of a 4-vector, as in Metal.

`copy-to-buffer` and `copy-from-buffer` take the components of vector elements consecutively in `sequence`. A 3-vector takes the space of a 4-vector in the buffer, and the padding is left alone.

## Errors

`metal-error` is signalled for a missing device, a released object, a transfer that overflows the buffer, an unknown kernel name or a failed command. `metal-compile-error` is its subclass for source that does not compile. Both carry `metal-error-message`.

[^1]: Buffers use shared storage, so `buffer-pointer` reads and writes the same memory the GPU sees.
[^2]: `dispatch` uses Metal's non-uniform threadgroups, available on Apple silicon GPUs.
