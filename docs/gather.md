# Checked device gather

`cl-gpu/gather` copies selected slices from a [strided view](strided-views.md). Values and indices stay on one device. Copies preserve the source dtype and raw bits, including f16/bf16 NaN payloads and signed zeros.

```lisp
(asdf:load-system "cl-gpu/gather")

(cl-gpu/gather:gather embeddings token-ids :axis 0)
(cl-gpu/gather:gather! output embeddings token-ids :axis 0)
```

## Shapes and indices

`input`, `indices` and `output` are strided views. Source rank is positive. `:axis` defaults to 0 and accepts negative axes counting from the end.

Gather replaces the selected source axis with the entire index shape. Index dimensions insert into the result; they do not pair with source batch dimensions. Repeated indices repeat slices.

| Source shape | Index shape | Axis | Result shape |
|---|---|---|---|
| `(5 3)` | `(2)` | `0` | `(2 3)` |
| `(5 3)` | `(2 4)` | `0` | `(2 4 3)` |
| `(2 5 3)` | `(4)` | `1` | `(2 4 3)` |
| `(5 3)` | `()` | `-1` | `(5)` |

Source elements are f16, bf16, f32, or signed/unsigned 8-, 16-, 32- and 64-bit integers. Indices use any of those eight integer types, with any rank. A rank-zero index view selects one slice and removes the axis.

For an axis of length `n`, signed indices in `[-n, n)` are valid; negative values count from the end. Unsigned indices must be less than `n`. Out-of-range values signal `cl-gpu:gpu-error` identifying the axis and its length. Error reports do not download the offending value or its position.

An empty index view selects no slices. Nonempty indices are checked even when another source dimension makes the result empty. A zero-length selected axis accepts only empty indices. Shape, type, device and destination validation also applies to empty calls.

## Layouts and ownership

Source and index views support signed strides, zero-stride broadcasts, transposes, padding and GPU-array windows. The destination has the exact result shape and source dtype, resides on the same device, and addresses distinct storage for each logical element. Destination padding stays unchanged.

`gather` returns an independent contiguous view. The caller owns its data array and frees it through `cl-gpu:free`. `gather!` returns the supplied destination. It stages the result before writing, including when the destination aliases the source or indices.

```lisp
(let ((result (cl-gpu/gather:gather embeddings token-ids)))
  (unwind-protect
       (use-result result)
    (cl-gpu:free (cl-gpu:strided-view-data result))))
```

Host sequences are not accepted as indices. Upload indices explicitly. Operands on different devices signal an error without implicit transfers.

## Completion and failures

Calls with nonempty indices normalize indices and gather into independent device storage, then read one u32 status word. That read waits for computation before returning or starting destination writes. Source and index payloads are never downloaded. Empty index views need no launch or status read.[^execution]

Metadata, bounds, allocation, compilation, launch and completion failures during gathering leave supplied destinations unchanged. Scratch is released on success and errors, following queued-buffer lifetimes.

After gathering succeeds, `gather!` queues the final copy into the destination. It does not wait for that copy. A failure during final-copy preparation leaves the destination unchanged; once a copy launch is attempted, a failure may leave partial output. Completion errors can surface at the next host transfer or `cl-gpu:synchronize`. Final-copy failures have no rollback guarantee.

## Tests and example

```lisp
(asdf:test-system "cl-gpu/gather")
(asdf:test-system "cl-gpu/gather/metal/tests")
(load "examples/gather.lisp")
```

The example gathers a batch of embedding vectors. Protocol tests cover validation, scheduling, transfer sizes, launch boundaries and scratch cleanup. Metal tests compare independent coordinate references across storage types and layouts, exercise aliases and guard padding, copy every f16/bf16 encoding, and inject submission/completion failures.

Both suites run on SBCL, ECL and CCL.

## Limitations

- Generic kernels decode O(rank) coordinates per element; axis coalescing and contiguous specialization are tracked in [#84](https://github.com/communal-software/cl-gpu/issues/84).
- Irregular destination uniqueness can require O(result elements) host workspace ([#75](https://github.com/communal-software/cl-gpu/issues/75)).
- Metadata uploads can wait for earlier Metal work ([#49](https://github.com/communal-software/cl-gpu/issues/49)).
- Tensor `take` dispatch remains separate ([cl-tensor #30](https://github.com/communal-software/cl-tensor/issues/30)).
- f64, complex, bool and vector storage, host indices, boolean masks and scatter are outside this API. Execution uses Metal; other runtime backends are planned.

[^execution]: Workspace is O(index count + result size), plus O(rank) layout metadata. Normalized indices use i64 storage; supplied destinations always use a full result staging buffer. Work is split into bounded launches. The Metal status read waits for preceding work on the device, including work queued by other operations.
