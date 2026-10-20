# Arrays

For device buffers, see [kernels and arrays](kernels.md#arrays) and [complex arrays](complex.md).

A kernel declares fixed-size arrays with `shared-array` and `local-array`. Each form returns a pointer to the first element, so `aref`, `ptr+` and the atomics work on it.

```lisp
(let ((tile (shared-array f32 64))        ; (ptr f32 :shared), one per group
      (acc  (local-array (vec f32 4) 8))) ; (ptr (vec f32 4) :private), one per thread
  (setf (aref tile (local-id 0)) (aref in (global-id 0)))
  (barrier)
  (setf (aref acc 0) (aref v 0)))
```

## Forms

| Form | Result |
|---|---|
| `(shared-array T n [:zero])` | `(ptr T :shared)`, shared by the threads of a group |
| `(local-array T n [:zero])` | `(ptr T :private)`, private to a thread |

- `n` is a literal positive integer.
- `T` is a numeric scalar, `bool` or a vector. A `shared-array` may also hold `(atomic i32)` or `(atomic u32)`.
- The contents are undefined unless `:zero` is given.

## Shared memory

- Shared arrays are static, so a launch does not size them. A `(ptr T :shared)` kernel parameter is still sized at launch.
- A shared array declared inside a branch or loop is one array for the group, not one per iteration.
- `:zero` clears a shared array before the body runs, and the whole group waits at one barrier.[^1] Threads that leave early with `return` still take part.
- A device limits the shared memory of a kernel, and the sum of all shared arrays and parameters counts toward it. A launch over the limit signals `gpu-error` naming the kernel, the bytes needed and the limit.[^2]

```lisp
(let ((hist (shared-array (atomic u32) 4 :zero)))
  (atomic-add (ptr+ hist (logand (local-id 0) 3)) 1)
  (barrier))
```

## Local arrays

A local array is declared where the form appears. `:zero` prints `= {}` on Metal and `= {0}` on OpenCL. Indexing with a runtime index may spill the array to slow memory.

[^1]: Metal needs threadgroup storage at kernel scope, so the printer declares every shared array at the top of the kernel and clears the `:zero` ones there. The clear uses `thread_index_in_threadgroup` and `threads_per_threadgroup`, added as kernel parameters.
[^2]: The common check ignores alignment padding, so the device compiler may still refuse a kernel just under the limit. OpenCL additionally checks compiled local-memory usage after setting dynamic local arguments.
