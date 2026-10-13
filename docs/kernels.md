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

## `define-kernel`

`(define-kernel name ((parameter type) ...) [docstring] form...)` defines the function `name`. Its arguments are one value per parameter, then keywords:

| Keyword | Meaning |
|---|---|
| `:grid` | thread count, or a list of up to three; required |
| `:group` | threads per group; defaults to the device's choice |
| `:wait` | `t` (default) waits for the kernel; `nil` returns a command to pass to `wait` |
| `:device` | the [device](#devices); defaults to `(current-device)` |

The body is [lowered](kernel-ir.md) when the definition is compiled, so a type error is reported then. The first launch on a device prints the source, compiles it and caches the result.[^1]

Parameter names are free: `grid`, `wait` and `pi` are valid.

| Parameter type | Argument |
|---|---|
| `(ptr T :global)`, `(ptr T :constant)` | a `gpu-array` of element type `T` on the launch device |
| `(ptr T :shared)` | the number of `T` elements of group-shared memory |
| scalar | a number of that type; `bool` takes any value |
| `(vec T n)` | a sequence of `n` numbers |

`find-kernel` returns the kernel bound to a name and `kernel-ir` its lowered IR.

## Arrays

| Function | Result |
|---|---|
| `(make-gpu-array length &key element-type device initial-contents)` | a zeroed array; `element-type` is a scalar or `(vec T n)` type, default `f32` |
| `(to-device sequence &key element-type device)` | a new array holding the contents of `sequence` |
| `(from-device array &key into flat)` | a new vector of the contents, or fills the first elements of `into` |
| `(free array)` | releases the device memory; freeing twice is harmless |
| `(with-gpu-array (var form) body)` | binds the array and frees it on exit |
| `gpu-array-length`, `gpu-array-element-type`, `gpu-array-device` | readers |

A vector array takes its contents as one sequence per element or as all the components in one flat sequence. `from-device` returns one vector per element, or with `:flat t` the components in one vector.[^3]

```lisp
(with-gpu-array (points (to-device '(#(1 2 3) #(4 5 6)) :element-type '(vec f32 3)))
  (from-device points)                ; => #(#(1.0 2.0 3.0) #(4.0 5.0 6.0))
  (from-device points :flat t))       ; => #(1.0 2.0 3.0 4.0 5.0 6.0)
```

An array not freed is released when it is garbage collected. Use `free` for large arrays: the collector does not see device memory pressure.

## Devices

`*device*` or, when it is `nil`, the default device of the first loaded backend.

| Function | Result |
|---|---|
| `(current-device)` | the device used when `:device` is omitted |
| `(wait command)` | blocks until an asynchronous launch finishes |

`cl-gpu/metal` is the only backend. `current-device` signals `gpu-error` when none is loaded.

## Errors

`gpu-error` covers a bad launch (missing `:grid`, wrong argument type, an array from another device or already freed, an element type that differs from the pointer's, shared memory over the [device limit](arrays.md#shared-memory)) and backend failures; `metal-error` is a subclass. `kernel-type-error` is signalled when the definition is compiled.

## Limitations

- A type error in a `define-kernel` body has no source location ([#16](https://github.com/communal-software/cl-gpu/issues/16)).
- Pipelines of a released device stay cached ([#18](https://github.com/communal-software/cl-gpu/issues/18)).

## Backends

A backend implements `device-target`, `device-allocate`, `device-free`, `device-store`, `device-load`, `device-prepare` and `device-launch` for its device class and calls `register-backend`.[^2] It may also implement `device-shared-memory-limit`, the bytes of shared memory a kernel may use; `nil`, the default, skips the check.

[^1]: The cache is per kernel and per device. `device-prepare` receives the source and the entry function name and returns whatever `device-launch` runs.
[^2]: `device-launch` receives its arguments as `(:buffer b)`, `(:bytes type value)` or `(:shared bytes)` in parameter order, the grid and group as lists of three, and returns the backend's command.
[^3]: A 3-vector takes the space of a 4-vector in the buffer, as in Metal; the padding is never read or written. `:flat` has no effect on scalar arrays.
