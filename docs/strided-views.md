# Strided views

`cl-gpu` describes N-dimensional layouts over existing GPU arrays. A view borrows its array; it does not copy elements or own a separate buffer.

```lisp
(asdf:load-system "cl-gpu")

(cl-gpu:make-strided-view data '(2 3))
;; Element (row column) is data[row*3 + column].

(cl-gpu:make-strided-view data '(2 3)
                          :strides '(3 -1) :offset 2)
;; Each row visits its three elements backward.
```

The view type and accessors also belong to `cl-gpu/reductions`; both packages expose the same symbols. [Elementwise operations](elementwise.md), reductions and [gather](gather.md) share these layouts.

## Layout

| Argument | Meaning |
|---|---|
| `data` | A live `gpu-array`, including a GPU-array window |
| `shape` | Sequence of nonnegative dimensions; `#()` is a scalar |
| `strides` | Signed element strides; omitted means contiguous row-major |
| `offset` | Nonnegative element offset relative to `data`; default 0 |

An element at coordinates `i` has address `offset + sum(i[a] * strides[a])`. Zero strides broadcast input elements. Transposed, padded and negative-stride layouts work. Every reachable address must fit inside `data`, even when `data` is itself a window. Empty layouts allow an offset at the end of the array.

`strided-view-data`, `strided-view-shape`, `strided-view-strides` and `strided-view-offset` expose the metadata. The constructor copies shape and strides. Treat the returned metadata as read-only. Indexing uses signed 64-bit arithmetic; layouts exceeding its range signal `gpu-error`.

## Storage lifetime

Views keep their data array alive. Freeing the owning GPU array invalidates its views. Free the array through `cl-gpu:free`; do not pass a strided view to `free`.

Allocating [reductions](reductions.md), [normalizers](normalization.md) and [gathers](gather.md) return views over independent contiguous storage. The caller owns that result array:

```lisp
(let ((result (cl-gpu/reductions:sum input :axis -1)))
  (unwind-protect
       (cl-gpu:from-device (cl-gpu/reductions:strided-view-data result))
    (cl-gpu:free (cl-gpu/reductions:strided-view-data result))))
```

Destination forms accept a matching view and return it. Its logical elements must address distinct storage locations. Input and weight views may broadcast; output views may not overlap themselves. Aliased inputs and outputs use device scratch before scattering results into the destination.[^aliases]

## Limitations

- Irregular output layouts may require O(output elements) host workspace to verify uniqueness ([#75](https://github.com/communal-software/cl-gpu/issues/75)). Ordinary row-major, transposed, padded and reversed layouts use an arithmetic check.

[^aliases]: Alias detection compares reachable ranges on the owning GPU array. Interleaved but disjoint ranges may use scratch unnecessarily. Exact in-place calls and overlaps spanning different reduction groups also use scratch.
