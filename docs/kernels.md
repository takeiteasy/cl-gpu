# Kernels and arrays

`define-kernel` defines a Lisp function that runs a kernel on the GPU. `gpu-array` holds the numbers it reads and writes.

```lisp
(asdf:load-system "cl-gpu/metal")
(in-package #:my-package)   ; uses cl-gpu

(define-kernel saxpy ((out (ptr f32 :global)) (x (ptr f32 :global))
                      (a f32) (n u32))
  (let ((i (global-id 0)))
    (when (< i n)
      (setf (aref out i) (+ (* a (aref x i)) (aref out i))))))

(with-gpu-array (x (to-device (make-array 1000 :initial-element 1.0)))
  (with-gpu-array (out (make-gpu-array 1000 :initial-contents (make-array 1000 :initial-element 2.0)))
    (saxpy out x 3.0 1000 :grid 1000)
    (from-device out)))   ; => #(5.0 5.0 ...)
```

Load `cl-gpu/metal` or `cl-gpu/opencl` to choose a runtime. See [OpenCL](opencl.md) for device selection and exact-grid launch requirements.

## `define-kernel`

`(define-kernel name ((parameter type) ...) [docstring] form...)` defines the function `name`. Its arguments are one value per parameter, then keywords:

| Keyword | Meaning |
|---|---|
| `:grid` | thread count, or a list of up to three; required |
| `:group` | threads per group; defaults to the device's choice |
| `:wait` | `t` waits for the device to finish; default `nil` queues the launch (see [queued launches](#queued-launches)) |
| `:device` | the [device](#devices); defaults to `(current-device)` |

The body is [lowered](kernel-ir.md) when the definition is compiled, so a type error is reported then. The first launch on a device prints the source, compiles it and caches the result.[^1]

Parameter names are free: `grid`, `wait` and `pi` are valid.

| Parameter type | Argument |
|---|---|
| `(ptr T :global)`, `(ptr T :constant)` | a `gpu-array` of element type `T` on the launch device |
| `(ptr T :shared)` | the number of `T` elements of group-shared memory |
| scalar | a number of that type; `bool` takes any value |
| `(vec T n)` | a sequence of `n` numbers |

## Precise math

A leading `(declare (precise-math))` compiles that kernel without fast math, so the compiler keeps the order and rounding of float operations. Use it for compensated sums such as two-sum. It goes before any other form (after the docstring) and is the only kernel declaration.

```lisp
(define-kernel two-sum ((in (ptr f32 :global)) (out (ptr f32 :global)))
  (declare (precise-math))
  (let* ((a (aref in 0)) (b (aref in 1)) (s (+ a b)) (bb (- s a)))
    (setf (aref out 0) (+ (- a (- s bb)) (- b bb)))))   ; the rounding error of a+b
```

Metal compiles libraries with fast math on by default; other kernels keep it.

`find-kernel` returns the kernel bound to a name and `kernel-ir` its lowered IR.

## Arrays

| Function | Result |
|---|---|
| `(make-gpu-array length &key element-type device initial-contents)` | a zeroed array; `element-type` is a scalar or `(vec T n)` type, default `f32` |
| `(to-device sequence &key element-type device)` | a new array holding the contents of `sequence` |
| `(copy-into array sequence)` | stores `sequence` in the first elements of an existing array or [view](#views), and returns it |
| `(from-device array &key into flat)` | a new vector of the contents, specialised to the element type, or fills the first elements of `into`[^6] |
| `(free array)` | releases the device memory; freeing twice is harmless |
| `(with-gpu-array (var form) body)` | binds the array and frees it on exit |
| `(make-gpu-array-view array length &key offset)` | a [view](#views) of `length` elements from element `offset` (default 0) |
| `gpu-array-length`, `gpu-array-element-type`, `gpu-array-device` | readers |

A vector array takes its contents as one sequence per element or as all the components in one flat sequence. `from-device` returns one vector per element, or with `:flat t` the components in one vector.[^3]

```lisp
(with-gpu-array (points (to-device '(#(1 2 3) #(4 5 6)) :element-type '(vec f32 3)))
  (from-device points)                ; => #(#(1.0 2.0 3.0) #(4.0 5.0 6.0))
  (from-device points :flat t))       ; => #(1.0 2.0 3.0 4.0 5.0 6.0)
```

An array not freed is released when it is garbage collected. Use `free` for large arrays: the collector does not see device memory pressure.

## Views

A view is a `gpu-array` that shares the memory of another array. It is accepted wherever an array is: kernel arguments, [BLAS](blas.md) and `from-device`.

```lisp
(with-gpu-array (x (to-device #(1 2 3 4 5 6)))
  (let ((tail (make-gpu-array-view x 3 :offset 3)))
    (copy-into tail #(7 8))      ; x is now #(1 2 3 7 8 6)
    (from-device tail)))         ; => #(7.0 8.0 6.0)
```

| | |
|---|---|
| `gpu-array-view-p`, `gpu-array-view-owner`, `gpu-array-view-offset` | the view test, the array that owns the memory, and the element offset in it |
| view of a view | a view of the owner, with the offsets added |
| `free` on a view | does nothing |
| `free` on the owner | later uses of its views signal `gpu-error` |
| a view past its array | `gpu-error` |

A view keeps its owner from being garbage collected. A kernel sees the view as a pointer to its first element, so `(aref p 0)` is the view's element 0.

A `(ptr T :constant)` parameter needs the view to start at a multiple of the device's alignment[^7]; any other view works. A launch with a misaligned view signals `gpu-error` naming the parameter.

## Queued launches

A launch returns `nil` at once and runs later, in order with the other launches on its device. The device runs queued launches in batches, so many small launches cost little each.[^5]

| Call | Effect |
|---|---|
| `(synchronize &optional device)` | blocks until every launch on `device` has finished; signals the first launch error since the last call |
| `:wait t` on a launch | the launch, then `synchronize` |
| `from-device`, `to-device` and BLAS reductions | wait for the device first |
| `free` | safe with launches pending |

```lisp
(dotimes (i 1000)
  (saxpy out x 3.0 1000 :grid 1000))   ; queued
(from-device out)                      ; waits, then reads
```

A launch that fails on the GPU reports at the next wait on its device, not at the launch.

## Devices

`*device*` or, when it is `nil`, the default device of the first loaded backend.

| Function | Result |
|---|---|
| `(current-device)` | the device used when `:device` is omitted |
| `(synchronize &optional device)` | blocks until the device's launches finish |

`cl-gpu/metal` is the only backend. `current-device` signals `gpu-error` when none is loaded.

## Errors

`gpu-error` covers a bad launch (missing `:grid`, wrong argument type, an array from another device or already freed, an element type that differs from the pointer's, shared memory over the [device limit](arrays.md#shared-memory)) and backend failures; `metal-error` and `opencl-error` are subclasses. `kernel-type-error` is signalled when the definition is compiled, with the `file:line:col` of the offending form or parameter type.[^4]

## Limitations

- A kernel error has no location when compiled from an editor buffer with `C-c C-c` ([#27](https://github.com/communal-software/cl-gpu/issues/27)).
- Pipelines of a released device stay cached ([#18](https://github.com/communal-software/cl-gpu/issues/18)).

## Backends

A backend implements `device-target`, `device-allocate`, `device-free`, `device-store`, `device-load`, `device-prepare` and `device-launch` for its device class and calls `register-backend`.[^2] `device-store` and `device-load` take a byte offset into the buffer and wait for pending launches. `device-synchronize` waits for all of them and defaults to doing nothing, for backends that run a launch before it returns. It may also implement `device-shared-memory-limit`, the bytes of shared memory a kernel may use; `nil`, the default, skips the check. `device-offset-alignment (device address-space)` is the multiple of bytes at which a buffer may be bound for a `:global` or `:constant` pointer; the default is 1. `cl-gpu/blas` adds the generics `device-gemm` and `device-gemv`, whose default methods run DSL kernels; a backend may specialize them to call a vendor library and use `call-next-method` for layouts it cannot handle.

[^1]: The cache is per kernel and per device. `device-prepare` receives the source and the entry function name and returns whatever `device-launch` runs.
[^2]: `device-launch` receives its arguments as `(:buffer b byte-offset)`, `(:bytes type value)` or `(:shared bytes)` in parameter order, and the grid and group as lists of three.
[^3]: A 3-vector takes the space of a 4-vector in the buffer, as in Metal; the padding is never read or written. `:flat` has no effect on scalar arrays.
[^4]: The location comes from re-reading the file being compiled or loaded and finding the `define-kernel` form in it. There is none for a form produced by another macro, written with `#.`, a backquote or an uninterned symbol, or evaluated at the REPL. Two identical `define-kernel` forms in one file both point at the first.
[^5]: On Metal, launches share one command buffer that is committed after 64 of them and at every wait. 2000 launches of a 256-thread kernel take 8 ms queued and 220 ms with `:wait t` on each (Apple M1).
[^6]: `f32`, `f16` and `bf16` arrays give `single-float` vectors, `f64` gives `double-float`, integers give `(signed-byte N)` or `(unsigned-byte N)`, and `bool` gives a simple vector. On Metal a simple vector of the matching type (`single-float` for `f32`, `(signed-byte 32)` for `i32`, and so on) transfers with one `memcpy`; 1,000,000 `f32` take about 1 ms, against about 30 ms for any other sequence (Apple M1).
[^7]: Metal's minimum for the `constant` address space depends on the GPU, so the Metal device reports a safe 256 bytes. Views of `f32` arrays start at multiples of 64 elements, `f16` of 128.
