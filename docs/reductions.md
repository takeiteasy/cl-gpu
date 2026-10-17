# Axis reductions

`cl-gpu/reductions` reduces selected axes of a [strided view](strided-views.md). Results stay on the device, including scalar results. Load a backend such as `cl-gpu/metal` to execute them.

| Function | Result |
|---|---|
| `sum`, `prod` | Sum or product |
| `mean` | Arithmetic mean |
| `minimum`, `maximum` | Smallest or largest value |
| `argmin`, `argmax` | Zero-based logical index of the first extreme |

Every function has allocating and destination forms:

```lisp
(sum input &key axis keepdims)
(sum! out input &key axis keepdims)
```

Names in these examples belong to `cl-gpu/reductions`. Allocating forms return an independent contiguous view; destination forms return `out`. The destination must match the result shape, element type and device. Invalid arguments signal `gpu-error` before destination writes.

## Axes and shape

Omitted or NIL `:axis` selects all axes. An integer selects one; a list or vector selects several. Negative axes count from the end. Duplicate and out-of-range axes signal errors. `:axis #()` selects no axes, so every group is a singleton.

Selected axes flatten in original dimension order, regardless of the order supplied. Arg indices describe this logical order, including reversed and broadcast inputs. Equal extrema select the first occurrence; values are compared directly, rather than by magnitude.

Results remove reduced dimensions unless `:keepdims t` retains them at size one. Reducing every axis produces shape `#()`.

```lisp
(cl-gpu:with-gpu-array (data (cl-gpu:to-device #(1.0 2.0 3.0 4.0 5.0 6.0)))
  (let* ((input (cl-gpu/reductions:make-strided-view data '(2 3)))
         (result (cl-gpu/reductions:sum input :axis 1 :keepdims t)))
    (unwind-protect
         (list (cl-gpu/reductions:strided-view-shape result)
               (cl-gpu:from-device (cl-gpu/reductions:strided-view-data result)))
      (cl-gpu:free (cl-gpu/reductions:strided-view-data result)))))
;; => (#(2 1) #(6.0 15.0))
```

## Precision and empty inputs

Inputs hold f32, f16 or bf16. Arithmetic and shared partials use f32; value results retain the input type and round on store. Arg results hold i64. There is no f64 or widened accumulation option. Floating-point reduction order differs from a sequential CPU sum; exceptional values follow backend arithmetic without a special NaN policy.

Empty groups produce zero for `sum` and one for `prod`. Mean and extrema signal an error if an output needs an empty group. Empty outputs perform no launches but still validate arguments.

Operations queue device work and return without downloading values.[^execution] Use `from-device` or `synchronize` to wait for completion. [Normalization](normalization.md) uses the same grouping rules. The [runnable example](../examples/reductions.lisp) demonstrates all three operation families.

## Tests

```sh
sbcl --noinform --non-interactive --eval '(asdf:test-system "cl-gpu/reductions")'
sbcl --noinform --non-interactive --eval '(asdf:test-system "cl-gpu/reductions/metal/tests")'
```

The first suite uses a mock device and needs no backend. The second checks device results against scalar references on Metal.

## Limitations

- One 256-thread group handles each reduction group. A few very long groups may underuse the GPU ([#74](https://github.com/communal-software/cl-gpu/issues/74)).
- Each nonempty call allocates and uploads layout metadata. Metal uploads wait for earlier work on the device ([#49](https://github.com/communal-software/cl-gpu/issues/49)).
- Destination forms accept `NIL` and allocate a result instead of rejecting the missing destination ([#77](https://github.com/communal-software/cl-gpu/issues/77)).
- Output validation has an [irregular-layout workspace limitation](strided-views.md#limitations).

[^execution]: Threads traverse group elements in strides of 256, combine partials in shared-memory trees and write final values directly to device storage. Metadata uses dimensions and signed strides, without host packing of input values. Kernels bind root buffers and incorporate GPU-array window offsets in their address arithmetic. Metal command buffers retain temporary buffers through completion.
