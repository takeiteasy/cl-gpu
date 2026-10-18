# Reusable device expressions

`cl-gpu/expressions` evaluates reusable arithmetic and reduction plans over [strided views](strided-views.md). Results stay on one GPU. Calls wait for completion; there is no CPU computation fallback or tensor-value download.

```lisp
(asdf:load-system "cl-gpu/metal")
(asdf:load-system "cl-gpu/expressions")

(defparameter *multiply-add*
  (cl-gpu/expressions:make-expression-plan
    '(a b c) '(:add (:multiply a b) c)))

;; INPUT is a strided view. The caller owns RESULT's backing array.
(let ((result (cl-gpu/expressions:execute-expression
                *multiply-add* (list input 2 1))))
  (unwind-protect
       (cl-gpu:from-device (cl-gpu:strided-view-data result))
    (cl-gpu:free (cl-gpu:strided-view-data result))))
```

## Calls and operands

Names below belong to `cl-gpu/expressions`.

| Call | Result |
|---|---|
| `(make-expression-plan parameters expression)` | Validated reusable `expression-plan` |
| `(expression-result-info plan operands &key axis keepdims)` | Three values: shape vector, element-type symbol, device |
| `(execute-expression plan operands &key axis keepdims)` | Independent contiguous result view |
| `(execute-expression! out plan operands &key axis keepdims)` | Fills and returns `out` |

Parameters are distinct variable symbols. Leaves are parameter names or real constants. Operands are a list or vector with one entry per declared parameter. Unused parameters do not affect validation, broadcasting, dtype or device. Definitions are copied; treat plans as immutable.

At least one referenced operand is a numeric view. Referenced views share one f16, bf16 or f32 storage type and device. Scalars and constants convert to f32. Floating arithmetic and reduction scratch use f32; f16/bf16 results round only on final storage. Mixed view types require explicit conversion before execution.

`expression-result-info` validates the call without allocating device storage, compiling kernels or transferring data. Shape, stride and scalar changes reuse compiled kernels. Plans retain kernel variants and per-device pipelines, without retaining input or destination buffers.[^cache]

## Expression forms

| Operators | Arity and meaning |
|---|---|
| `:add`, `:subtract`, `:multiply`, `:divide`, `:min`, `:max` | One or more operands, folded left |
| `:subtract`, `:divide` with one operand | Negation or reciprocal |
| `:abs`, `:sqrt`, `:exp`, `:sin`, `:cos`, `:log`, `:tanh`, `:sigmoid` | One operand; angles use radians |
| `:fma` | Three operands; fused multiply-add |
| `:eq`, `:ne`, `:lt`, `:le`, `:gt`, `:ge` | Two operands; comparison used as a selection condition |
| `(:select comparison on-true on-false)` | Numeric selection; evaluates only the selected branch |
| `:sum`, `:prod`, `:minimum`, `:maximum`, `:argmin`, `:argmax` | One operand; group reduction |

The result is numeric; standalone comparison results and numeric selection conditions are rejected. Unknown operators, wrong arities and undeclared parameters signal `gpu-error` when constructing a plan, including in unused branches. Arbitrary Lisp functions, integer expressions, f64, complex types and mask outputs are unsupported.

Ordinary multiply/add expressions preserve separate f32 rounding. Only `:fma` requests contraction. Exceptional values follow Metal arithmetic: evaluated negative square roots, for example, produce NaNs rather than checked domain errors. Floating results need not match CPU results bit for bit.

## Reduction groups

Every reducer uses the same `axis` set over the broadcast shape of all referenced operands. NIL selects every axis, negative axes count from the end, and `#()` selects none. Duplicate and out-of-range axes signal errors.

A top-level reducer removes selected axes; `keepdims t` retains them at size one. Other expressions preserve the broadcast shape, and reject `keepdims t`. Nested reducers broadcast their group result within the complete group.

```lisp
(cl-gpu/expressions:make-expression-plan
  '(x) '(:divide (:exp (:subtract x (:maximum x)))
                 (:sum (:exp (:subtract x (:maximum x))))))
;; Execute with :axis -1 for stable row softmax.
```

Top-level arg reducers return i64 logical indices. Selected axes enumerate in ascending axis order, regardless of their supplied order or physical strides. Equal extrema retain the first index. Nested arg indices convert to f32.

An empty sum group produces zero; an empty product group produces one, without evaluating its operand. Required empty extreme/arg groups signal errors. Empty outputs validate but launch no work.

Selection remains lazy around and inside reductions. If any selected element needs a reducer, that reducer evaluates its operand across the whole group. Groups where no selected element needs it skip evaluation.[^passes]

## Layouts, destinations and errors

Broadcasting aligns dimensions from the right, including expansion from 1 to 0. Signed strides, zero-stride inputs, transposes, padding, GPU-array windows and rank-zero views retain the [shared view contracts](strided-views.md). Indexing fits signed 64-bit arithmetic.

Destinations must be live views with the inferred shape, dtype and device, and distinct addresses for every logical element. `execute-expression!` rejects NIL. Every destination call stages the complete result, including exact in-place and shifted aliases. Padding remains unchanged.

Validation, compilation, computation-launch and computation-completion failures leave the destination unchanged. Only after computation succeeds does the final device copy start. A failure during that copy can leave a partial update; it has no rollback guarantee.

Scratch, masks and layout buffers belong to one invocation and are released on success or failure. Allocating calls return a view whose backing array belongs to the caller. Freeing an input owner invalidates its views.

Plans support concurrent calls using independent inputs and destinations. Callers coordinate overlapping external writes. Synchronization follows the runtime's device-wide error reporting; errors from other queued work may surface during a call. Host uploads contain layout metadata, not packed tensor values. This expression subset needs no device status download.

## Tests and examples

```sh
sbcl --non-interactive --eval '(asdf:test-system "cl-gpu/expressions")'
sbcl --non-interactive --eval '(asdf:test-system "cl-gpu/expressions/metal/tests")'
sbcl --non-interactive --load examples/expressions.lisp
```

Protocol tests cover validation, cache reuse, launch boundaries, capability failures and cleanup. Metal tests compare independent scalar references across layouts and nested/conditional reductions, and inject computation failures against real device buffers. See [complete-call measurements](expressions-performance.md).

## Limitations

- Reduction operands and conditional demand masks use full-shape temporary storage, retained until completion. Liveness-based reuse, bounded storage and producer fusion are tracked in [#83](https://github.com/communal-software/cl-gpu/issues/83).
- A single 256-thread group handles each reduction group; long groups can underuse the device ([#74](https://github.com/communal-software/cl-gpu/issues/74)).
- Coordinate decoding costs O(rank) per element ([#76](https://github.com/communal-software/cl-gpu/issues/76)); irregular destination uniqueness may need O(output elements) host workspace ([#75](https://github.com/communal-software/cl-gpu/issues/75)).
- Metadata uploads and completion waits synchronize the device ([#49](https://github.com/communal-software/cl-gpu/issues/49)); pipelines of released devices remain cached ([#18](https://github.com/communal-software/cl-gpu/issues/18)).
- Metal is the execution target. f64 support on capable backends is tracked in [#81](https://github.com/communal-software/cl-gpu/issues/81).

[^cache]: A plan caches kernels by resolved expression, operand storage/scalar kinds, result type, mask presence and copy/predicate mode. Layout metadata carries dimensions, strides and axis grouping at runtime, so ranks and axis sets also reuse compatible variants. Each kernel caches prepared pipelines by device; publication is locked, and compilation failures remain retryable.

[^passes]: The planner fuses arithmetic regions, schedules dependent reduction passes and reuses equal reducers within the same demand context. Conditional reducers add device-side predicate masks and per-group demand reductions. Their operands compute under the expanded group mask; no host read decides which groups run. Reduction trees use f32 partials and i64 arg indices.
