# BLAS

`cl-gpu/blas` runs the f32 BLAS Level 1 routines on [GPU arrays](kernels.md#arrays). Matrix products are on the [matrices](blas-matrices.md) page. Names, arguments and results follow [trivial-simd/blas](https://github.com/communal-software/trivial-simd/blob/trunk/docs/blas.md).

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

Each takes `:x-offset` (and `:y-offset` for two vectors), default 0. Vectors are `f32` `gpu-array`s on one device. Every routine waits for the GPU. Reductions return Lisp numbers.[^1]

## Increments and offsets

A span is `n` elements from the offset, `|inc|` apart. A negative increment visits the span from its far end, so `isamax` counts in visiting order.

| Case | Behaviour |
|---|---|
| `n = 0` | validated, then returns without launching; reductions return 0 |
| `alpha = 0` | `saxpy` leaves `y` alone; `sscal` writes zeros without reading `x` |
| zero increment, negative offset, span past the array | `gpu-error` |
| non-`f32` array, freed array, arrays on different devices | `gpu-error` |
| spans of one array that overlap | `gpu-error`.[^2] Identical spans are fine |

## Limitations

- f32 only ([#40](https://github.com/communal-software/cl-gpu/issues/40)). No rotations ([#41](https://github.com/communal-software/cl-gpu/issues/41)) or complex routines ([#39](https://github.com/communal-software/cl-gpu/issues/39)).
- No batched `gemm` ([#38](https://github.com/communal-software/cl-gpu/issues/38)). `gemv` and `gemm` are on the [matrices](blas-matrices.md) page.
- Every launch waits ([#42](https://github.com/communal-software/cl-gpu/issues/42)).
- Reductions allocate a partials array per call ([#43](https://github.com/communal-software/cl-gpu/issues/43)).
- `isamax` of data holding NaN is not defined.

[^1]: Each group of 256 threads reduces its share in shared memory and the host combines at most 256 partial results, summing in double precision. `snrm2` keeps a (scale, sum of squares) pair as LAPACK does, so squares of very large or small values do not overflow or vanish.
[^2]: The test is exact for spans of one stride and conservative for different strides: it can refuse spans that interleave without sharing an element.
