# Normalization

`cl-gpu/reductions` supplies softmax and weighted RMSNorm over [strided views](strided-views.md). Both preserve the input shape and element type: f32, f16 or bf16. Arithmetic uses f32 and results round on store.

| Call | Group computation |
|---|---|
| `(softmax input &key (axis -1))` | `exp(x - maximum) / sum(exp(x - maximum))` |
| `(rmsnorm input &key (axis -1) weights (epsilon 1e-6))` | `x / sqrt(mean(x²) + epsilon)`, then multiply by weights |

Allocating forms return independent contiguous views. Destination forms take `out` first and return it: `(softmax! out input ...)` and `(rmsnorm! out input ...)`. They accept the same keywords. Results remain on the GPU.

## Groups

The default axis is the last dimension. Explicit NIL selects all axes; integers, lists and vectors follow the [reduction axis rules](reductions.md#axes-and-shape). There is no `keepdims` option because the output retains every dimension.

`#()` selects no axes: each element is a singleton group. Softmax produces one for ordinary finite singleton inputs; RMSNorm computes `x / sqrt(x² + epsilon)`, then applies weights. Rank-zero views require explicit `:axis nil` or `:axis #()`.

## Weights and epsilon

Weights default to one. A real scalar weight converts to f32. A weight view must have the input element type and device, and broadcast to the input shape without expanding it. Signed and zero weight strides work.

For input shape `(batch rows features)`, a vector of shape `(features)` shares feature weights, while `(batch 1 features)` supplies batch-specific feature weights.

Epsilon is a nonnegative real converted to f32. All-zero input with positive epsilon produces zeros. Zero epsilon with an all-zero group follows backend division-by-zero arithmetic. Maximum subtraction stabilizes softmax for ordinary finite inputs; extreme differences, square overflow, underflow, infinities and NaNs follow backend arithmetic. There is no widened accumulation option.

## Destinations

The destination must match the input shape, element type and device, with distinct output addresses. Reversed, transposed and padded destinations work. Input and weight aliases use device scratch so every input value is read before destination writes. Invalid arguments leave the destination unchanged.

Empty inputs return empty outputs without evaluating groups; axes, weights, epsilon and destinations are still validated.

```lisp
(asdf:load-system "cl-gpu/reductions")
(asdf:load-system "cl-gpu/metal")
(load "examples/reductions.lisp")
(cl-gpu/reductions-example:run-example)
;; => (:SUM #(6.0 15.0)
;;     :SOFTMAX #(0.09003057 0.24472848 0.66524094
;;                0.09003057 0.24472848 0.66524094)
;;     :RMSNORM #(...))
```

## Limitations

- Normalizers share the [reduction throughput and metadata-upload limits](reductions.md#limitations).
- Overlapping destinations require full-result device scratch. Output uniqueness checks have an [irregular-layout workspace limit](strided-views.md#limitations).
