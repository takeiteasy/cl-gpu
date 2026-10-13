# BLAS matrices

`sgemv` and `sgemm` run f32 matrix-vector and matrix-matrix products on [GPU arrays](kernels.md#arrays). Arguments and results follow [trivial-simd/blas](https://github.com/communal-software/trivial-simd/blob/trunk/docs/blas-level3.md). Vectors take the same increments and offsets as [Level 1](blas.md).

```lisp
(cl-gpu:with-gpu-array (a (cl-gpu:to-device #(1 2 3 4 5 6)))      ; 2x3, row-major
  (cl-gpu:with-gpu-array (b (cl-gpu:to-device #(1 0 0 1 1 1)))    ; 3x2
    (cl-gpu:with-gpu-array (c (cl-gpu:make-gpu-array 4))
      (cl-gpu/blas:sgemm :no-transpose :no-transpose 1.0
                         (cl-gpu/blas:make-matrix-view a 2 3)
                         (cl-gpu/blas:make-matrix-view b 3 2)
                         0.0
                         (cl-gpu/blas:make-matrix-view c 2 2))
      (cl-gpu:from-device c))))                                   ; => #(4.0 5.0 10.0 11.0)
```

## Routines

| Routine | Arguments | Result |
|---|---|---|
| `sgemv` | `transpose alpha view x incx beta y incy` | `y`, set to `alpha*op(view)*x + beta*y` |
| `sgemm` | `transa transb alpha a b beta c` | `c`, set to `alpha*op(a)*op(b) + beta*c` |

`sgemv` also takes `:x-offset` and `:y-offset`. A transpose flag is `:no-transpose`, `:transpose` or `:conjugate-transpose`; the last equals `:transpose` for real data. Every call waits for the GPU.

## Matrix views

A view is a window onto an f32 `gpu-array`. It shares the array's storage, so changing the array changes the view.

| Function | Result |
|---|---|
| `(make-matrix-view array rows cols &key layout leading-dimension offset row-stride column-stride)` | a view; `layout` is `:row-major` (default) or `:column-major` |
| `(matrix-subview view row column rows cols)` | the part of a view from `(row column)` |
| `matrix-view-rows`, `-cols`, `-layout`, `-leading-dimension`, `-row-stride`, `-column-stride`, `-offset`, `-data` | readers |

Element `(row, column)` is at `offset + row*row-stride + column*column-stride`. Pass both strides, which may be negative, instead of a layout and leading dimension. Regular strides report a layout; others report `:strided`.

```lisp
(cl-gpu/blas:make-matrix-view a 2 3 :leading-dimension 4)           ; rows 4 apart
(cl-gpu/blas:make-matrix-view a 2 3 :offset 5 :row-stride -3 :column-stride -1)   ; both reversed
```

## Edge cases

| Case | Behaviour |
|---|---|
| `m = 0` or `n = 0` | validated, then returns without launching |
| `alpha = 0` or `k = 0` | `y` or `c` is set to `beta` times itself; no launch when `beta = 1` |
| `beta = 0` | the old `y` or `c` is never read, so NaN or infinity there is replaced |
| mismatched dimensions, bad flag, non-`f32` or freed array, arrays on different devices | `gpu-error` |
| `c` with a zero stride, or elements addressed twice | `gpu-error`[^1] |
| `y`, or `c`, sharing a range of an array with an input | `gpu-error`[^2] |

## Metal and MPS

On a Metal device `sgemm` and `sgemv` use MetalPerformanceShaders (MPS) when every matrix is a regular layout with positive strides and, for `sgemv`, both increments are 1. Other cases run the DSL kernels, which accept any strides.[^3] Bind `cl-gpu/metal:*use-mps*` to `nil` to always use the DSL kernels.

| Size (square) | MPS | DSL kernel |
|---|---|---|
| 256 | 0.30 ms | 0.78 ms |
| 1024 | 2.2 ms | 7.7 ms |
| 2048 | 11 ms | 61 ms |

Measured on an Apple M1 with SBCL.

## Limitations

- f32 only ([#40](https://github.com/communal-software/cl-gpu/issues/40)). No batched `gemm` ([#38](https://github.com/communal-software/cl-gpu/issues/38)), complex routines ([#39](https://github.com/communal-software/cl-gpu/issues/39)) or other Level 2 and 3 routines.
- The DSL `sgemm` computes one element per thread ([#44](https://github.com/communal-software/cl-gpu/issues/44)); the DSL `sgemv` one row per thread ([#45](https://github.com/communal-software/cl-gpu/issues/45)).
- MPS kernel objects are created on every call ([#46](https://github.com/communal-software/cl-gpu/issues/46)).

[^1]: The test is sufficient but not necessary: it refuses strides that interleave without sharing an element, such as row stride 3 and column stride 2 over a 2 by 2 view.
[^2]: The test compares the lowest and highest index each operand reaches, so it also refuses disjoint views whose ranges interleave, such as the left and right halves of a row-major matrix. Inputs may overlap each other. Identical input and output views are refused.
[^3]: MPS reads a row-major matrix, or a column-major one as its transpose. A column-major `c` is computed as the transpose of the product. A view whose last row ends before the array does is also left to the DSL kernels. MPS reads the result even when `beta` is 0, so the Metal methods clear it first.
