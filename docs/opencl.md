# OpenCL

`cl-gpu/opencl` runs core kernels through the OpenCL 1.2 C API. The `:opencl` printer is part of `cl-gpu` and generates source without an OpenCL installation.

## Run a kernel

```lisp
(asdf:load-system "cl-gpu/opencl")

(cl-gpu:define-kernel saxpy
    ((out (cl-gpu:ptr cl-gpu:f32 :global))
     (x (cl-gpu:ptr cl-gpu:f32 :global)) (a cl-gpu:f32))
  (let ((i (cl-gpu:global-id 0)))
    (setf (aref out i) (+ (* a (aref x i)) (aref out i)))))

(let ((cl-gpu:*device* (cl-gpu/opencl:default-device)))
  (cl-gpu:with-gpu-array (x (cl-gpu:to-device #(1.0 2.0 3.0 4.0)))
    (cl-gpu:with-gpu-array (out (cl-gpu:make-gpu-array 4))
      (saxpy out x 2.0 :grid 4 :group 2)
      (cl-gpu:from-device out)))) ; #(2.0 4.0 6.0 8.0)
```

See the runnable [example](../examples/opencl.lisp).

## Devices and resources

| Interface | Behavior |
|---|---|
| `(available-devices)` | Device descriptions in platform enumeration order |
| `(device-name device)` | Name of a description or opened device |
| `(device-extensions device)` | Extension names as a list of strings |
| `(open-device description)` | Opens one context and ordered queue |
| `(default-device)` | Cached device, opened lazily; prefers a GPU, otherwise a CPU |
| `(release device)` | Waits, releases kernels/context/queue; repeated release is harmless |

These names belong to `cl-gpu/opencl`. Bind `cl-gpu:*device*` or pass `:device` to select an opened device. The first registered backend supplies the default when `*device*` is `nil`.

```lisp
(let ((device (cl-gpu/opencl:open-device
               (first (cl-gpu/opencl:available-devices)))))
  (unwind-protect
       (let ((cl-gpu:*device* device))
         ;; Allocate arrays and run kernels here.
         (cl-gpu:synchronize))
    (cl-gpu/opencl:release device)))
```

Buffers start at zero. Host transfers preserve scalar and vector layouts, including four-component storage for three-vectors. Free arrays with `free` or `with-gpu-array`; garbage collection also releases their buffers.

## Launches and views

- `:grid` is the exact number of threads in each dimension. An explicit `:group` must divide every grid dimension and fit device and compiled-kernel limits. Launches are never padded.
- Without `:group`, OpenCL chooses the local size. Cooperative kernels should specify their required group dimensions.
- Launches queue in order. `synchronize`, host reads and host writes wait for preceding launches and report execution errors.
- Global and constant views bind through sub-buffers. Their byte offsets must be multiples of the device's reported base-address alignment. Host transfers accept unaligned view offsets.
- Shared-memory checks include static arrays and dynamic local arguments. OpenCL also checks compiled local-memory usage before enqueueing.[^local-memory]

## Types and compilation

The printer supports baseline integer types, `f32`, vectors, local boolean expressions, address spaces, arrays, conversions, barriers and 32-bit integer atomics. Boolean kernel parameters and private pointer parameters signal `kernel-target-error`.

`f16` requires `cl_khr_fp16`; `f64` requires `cl_khr_fp64`. The source enables the required extensions. Unsupported device capabilities signal `opencl-compile-error`. Compilation uses OpenCL C 1.2; `(declare (precise-math))` disables fast math. Compiler failures include the build log.[^errors]

## Setup and tests

macOS uses the system OpenCL framework. Linux uses `libOpenCL.so.1` or `libOpenCL.so`; install an ICD loader and an OpenCL 1.2-capable device implementation such as PoCL. See [Linux setup and validation](opencl-linux.md) for Ubuntu, ARM64 Lima and CI commands.

```sh
sbcl --noinform --non-interactive --eval '(asdf:test-system "cl-gpu")'
sbcl --noinform --non-interactive --eval '(asdf:test-system "cl-gpu/opencl")'
```

The core suite includes printer tests without loading OpenCL. The runtime suite requires a device and fails explicitly when none is available. Executed f64 arithmetic is conditional on `cl_khr_fp64`; the Linux PoCL validation requires that extension.

## Limitations

- Higher-level operation parity is separate work: Metal-specific helpers, expression device restrictions and non-divisible operation launchers require porting ([operation parity](https://github.com/communal-software/cl-gpu/issues/86)).
- Kernel `bf16`, images, samplers, SVM and OpenCL 2.x features are unsupported.
- One queue lock serializes encoding; host staging adds memory allocation and copying ([runtime performance](https://github.com/communal-software/cl-gpu/issues/88)).
- Released devices remain keys in the core kernel cache ([pipeline cache](https://github.com/communal-software/cl-gpu/issues/18)); their native OpenCL kernels are released.

[^errors]: `opencl-error` derives from `cl-gpu:gpu-error`; `opencl-compile-error` derives from `opencl-error`. Runtime API errors include their operation and numeric OpenCL status.
[^local-memory]: The DSL accounts for static arrays from the kernel IR. Direct `device-launch` calls check the sum of dynamic local arguments plus the driver's reported static usage; that static report can be zero on PoCL.
