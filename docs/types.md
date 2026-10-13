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

## Vectors

Arithmetic on a vector takes another vector of its type, or a scalar of its element type, which applies to every component.

```lisp
(* v 2.0)               ; v is (vec f32 4): every component doubled
(+ 1.0 v)
(mod k 3)               ; k is (vec i32 2)
(swizzle v :zx)         ; (vec f32 2), components 2 and 0
(swizzle v :w)          ; f32
```

- A scalar must match the element type exactly: `(* v i)` with `i32` is an error. Literals adopt it.
- `swizzle` takes 1 to 4 components, which may repeat. Use `xyzw` or `rgba`, not both. A swizzle is read-only; assign with `(setf (aref v 0) ...)`.
- Comparisons stay scalar-only, since the type set has no bool vectors. `min`, `max`, `clamp`, `pow` and `fma` need vectors of one type.

## Pointer arithmetic

| Form | Result |
|---|---|
| `(ptr+ p n)` | `p` offset by `n` elements, same pointer type |
| `(ptr- p q)` | `i64` element distance; `p` and `q` have one pointer type |
| `(incf q [n])`, `(decf q [n])` | moves the pointer place `q` |

`n` is an integer scalar. `+` and `-` stay numeric-only. `=` `/=` `<` `<=` `>` `>=` compare two pointers of one type.

```lisp
(let ((q (ptr+ out (global-id 0))))
  (setf (aref q 0) 1.0)
  (atomic-add (ptr+ counts 2) 1))   ; works on atomic pointers too
```

## Remainders

| Form | Result has the sign of | Float |
|---|---|---|
| `(rem a b)` | `a` (truncated) | `fmod` |
| `(mod a b)` | `b` (floored) | floored `fmod` |

`rem` and `mod` take two operands of one numeric type, or two vectors of one type. They agree on unsigned integers.[^2] `bf16` is not available on Metal.

```lisp
(rem -7 3)    ; => -1
(mod -7 3)    ; =>  2
(mod 7.5 -2.0) ; => -0.5
```

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
[^2]: A signed or float `mod` calls the helper `cl_gpu_mod` or `cl_gpu_fmod`, which the source defines when the kernel uses it. It adds the divisor only when the remainder is non-zero and the signs differ, so it cannot overflow. As with `/`, a zero divisor and `MIN % -1` are undefined.
