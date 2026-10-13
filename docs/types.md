# Types

Kernels use a closed set of types. Kernel parameters are annotated; everything else is inferred.

```lisp
(cl-gpu:lower-kernel form
  :arguments '((out (ptr f32 :global)) (n u32)))
```

## Type set

| Type | Spelling | Notes |
|---|---|---|
| Floats | `f16` `bf16` `f32` `f64` | |
| Signed integers | `i8` `i16` `i32` `i64` | |
| Unsigned integers | `u8` `u16` `u32` `u64` | |
| Boolean | `bool` | |
| Vector | `(vec f32 4)` | element is a numeric scalar, length 2, 3 or 4 |
| Pointer | `(ptr f32 :global)` | pointee is a scalar, vector or atomic |
| Atomic | `(atomic u32)` | `i32` or `u32`; only as a pointee |

Type names are symbols exported from `cl-gpu`, so the kernel's package must use it.

## Address spaces

| Space | Metal | CUDA | OpenCL |
|---|---|---|---|
| `:global` | `device` | global | `__global` |
| `:shared` | `threadgroup` | `__shared__` | `__local` |
| `:constant` | `constant` | `__constant__` | `__constant` |
| `:private` | `thread` | local | `__private` |

## Inference

| Form | Type |
|---|---|
| `(let ((x init)) ...)` | the type of `init` |
| `(let ((x)) (declare (type u32 x)) ...)` | the declared type; the declaration is required without an init |
| `(for (i 0 (< i n) ...) ...)` | the declared type, else the type of the variable `i` is compared with, else `i32` |
| `(if c a b)` as a value | the type of `a` and `b`, which must agree |
| comparison, `and`, `or`, `not` | `bool` |

## Strict arithmetic

Operands share one type. Nothing converts implicitly.

```lisp
(+ x 1)       ; x is f32: the literal becomes 1.0f
(+ x i)       ; error: expected f32, got i32
(+ x (convert f32 i))
```

Literals adopt the type they meet:

| Literal | Default | Adopts |
|---|---|---|
| integer | `i32` (`i64` if larger) | any integer type it fits, any float type |
| `single-float` | `f32` | any float type |
| `double-float` | `f64` | any float type |
| `t`, `nil` | `bool` | `bool` |

A constant expression such as `(* 2 3)` adopts a type as one literal does. The reader decides whether `1.5` is a `single-float` or a `double-float`[^1].

## Conditions

Tests of `if`, `for`, `while`, `and`, `or` and `not` must be `bool`: write `(> n 0)`, not `n`.

## Per-target support

`type-support` answers whether a scalar exists on `:metal`, `:cuda` or `:opencl`. It returns `t`, `nil` or the extension to enable.

| Type | Metal | CUDA | OpenCL |
|---|---|---|---|
| `f16` | yes | yes | `cl_khr_fp16` |
| `bf16` | yes | yes | no |
| `f64` | no | yes | `cl_khr_fp64` |

`ir-kernel-scalar-types` lists the scalar types a lowered kernel uses, so a printer can reject it early.

[^1]: `*read-default-float-format*` applies when the kernel text is read, so `1.5` is `f64` under `double-float`. Write `1.5f0` or `1.5d0` to be explicit.
