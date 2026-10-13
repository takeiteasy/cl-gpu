# BLAS

`cl-gpu/blas` runs the BLAS Level 1 routines on [GPU arrays](kernels.md#arrays), in f32, f16 and bf16. Matrix products are on the [matrices](blas-matrices.md) page. Names, arguments and results follow [trivial-simd/blas](https://github.com/communal-software/trivial-simd/blob/trunk/docs/blas.md).

```lisp
(asdf:load-system "cl-gpu/blas")
(asdf:load-system "cl-gpu/metal")

(cl-gpu:with-gpu-array (x (cl-gpu:to-device #(1.0 2.0 3.0)))
  (cl-gpu:with-gpu-array (y (cl-gpu:to-device #(4.0 5.0 6.0)))
    (cl-gpu/blas:saxpy 3 2.0 x 1 y 1)   ; y := 2x + y
    (list (cl-gpu:from-device y)           ; => #(6.0 9.0 12.0)
          (cl-gpu/blas:sdot 3 x 1 y 1)  ; => 60.0
          (cl-gpu/blas:isamax 3 y 1)))) ; => 2
```

## Routines

| Routine | Arguments | Result |
|---|---|---|
| `saxpy` | `n alpha x incx y incy` | `y`, set to `alpha*x + y` |
| `sscal` | `n alpha x incx` | `x`, set to `alpha*x` |
| `scopy` | `n x incx y incy` | `y`, set to `x` |
| `sswap` | `n x incx y incy` | `x` and `y`, exchanged |
| `sdot` | `n x incx y incy` | the dot product |
| `snrm2` | `n x incx` | the Euclidean norm |
| `sasum` | `n x incx` | the sum of absolute values |
| `isamax` | `n x incx` | zero-based index of the first element of largest magnitude |

`axpy`, `scal`, `copy`, `swap`, `dot`, `nrm2`, `asum` and `iamax` take the same arguments and take the [element type](#element-types) from `x`. `saxpy` and the rest of the table are the same routines for f32 arrays only.

Mixed-precision dots take f32 arrays:

| Routine | Arguments | Result |
|---|---|---|
| `dsdot` | `n x incx y incy` | the dot product as a `double-float` |
| `sdsdot` | `n alpha x incx y incy` | `alpha` plus the dot product, as a `single-float` |

Each takes `:x-offset` (and `:y-offset` for two vectors), default 0. Vectors are `f32` `gpu-array`s on one device. Routines are [queued](kernels.md#queued-launches) and return at once; reductions wait for the result and return Lisp numbers.[^1]

## Element types

| Array type | Routines | Arithmetic |
|---|---|---|
| `f32` | `s*` and the generic routines | f32 |
| `f16`, `bf16` | the generic routines | f32; results are rounded once when stored |
| `f64` | none | see Limitations |

Both arrays of a call have one element type. Reductions return `single-float`s accumulated in f32 whatever the element type, so a sum of f16 values does not overflow at 65504.[^3]

```lisp
(cl-gpu:with-gpu-array (x (cl-gpu:to-device #(1.0 2.0 3.0) :element-type 'cl-gpu:f16))
  (cl-gpu/blas:scal 3 0.5 x 1)         ; x := 0.5x, rounded to f16
  (cl-gpu/blas:asum 3 x 1))            ; => 3.0
```

`dsdot` accumulates in a pair of f32 values (about 48 bits), so a sum that cancels in f32 keeps its small terms.[^4]

## Increments and offsets

A span is `n` elements from the offset, `|inc|` apart. A negative increment visits the span from its far end, so `isamax` counts in visiting order.

| Case | Behaviour |
|---|---|
| `n = 0` | validated, then returns without launching; reductions return 0 |
| `alpha = 0` | `saxpy` leaves `y` alone; `sscal` writes zeros without reading `x` |
| zero increment, negative offset, span past the array | `gpu-error` |
| unsupported or mismatched element types, freed array, arrays on different devices | `gpu-error` |
| spans of one array that overlap | `gpu-error`.[^2] Identical spans are fine |

## Limitations

- No f64 routines ([#55](https://github.com/communal-software/cl-gpu/issues/55)): Metal has no f64. No rotations ([#41](https://github.com/communal-software/cl-gpu/issues/41)) or complex routines ([#39](https://github.com/communal-software/cl-gpu/issues/39)).
- `gemv` and `gemm`, including batches, are on the [matrices](blas-matrices.md) page.
- Reductions allocate a partials array per call ([#43](https://github.com/communal-software/cl-gpu/issues/43)).
- `isamax` of data holding NaN is not defined.

[^1]: Each group of 256 threads reduces its share in shared memory and the host combines at most 256 partial results, summing in double precision. `snrm2` keeps a (scale, sum of squares) pair as LAPACK does, so squares of very large or small values do not overflow or vanish.
[^2]: The test is exact for spans of one stride and conservative for different strides: it can refuse spans that interleave without sharing an element.
[^3]: Each element is widened to f32 on load. `copy` and `swap` move the stored values unchanged.
[^4]: Each thread keeps a (high, low) pair, adding products with `fma` for the exact product error and two-sum for the rounding error. Groups combine the pairs and the host sums them in double precision. The kernel is compiled with [precise math](kernels.md#precise-math), because fast math can rewrite the compensation away.
