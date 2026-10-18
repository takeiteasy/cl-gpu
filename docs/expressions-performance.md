# Device expression measurements

Reusable expression plans reduce complete-call time for the multiply-add and stable-softmax cases below. Results include result allocation, layout uploads, execution, synchronization and cleanup; destination calls also include staged copying.

## Recorded run

Apple M1, SBCL 2.6.8, f32, 2026-10-18. Inputs and destinations are preallocated 256×256 views. Reversed inputs reverse the inner axis. Broadcast multiply-add uses a 1×256 right operand. Softmax reduces the last axis; its right operand is unused.[^measurement]

| Allocating case | Composed µs | Expression µs |
|---|---:|---:|
| Multiply-add, contiguous | 813.8 | 379.9 |
| Multiply-add, reversed | 724.2 | 377.2 |
| Multiply-add, broadcast | 729.5 | 355.7 |
| Softmax, contiguous | 1,758.8 | 1,007.6 |
| Softmax, reversed | 1,649.0 | 958.9 |

| Destination case | Composed µs | Expression µs |
|---|---:|---:|
| Multiply-add, contiguous | 1,156.5 | 721.5 |
| Multiply-add, reversed | 1,075.7 | 686.4 |
| Multiply-add, broadcast | 1,067.9 | 719.0 |
| Softmax, contiguous | 2,104.2 | 1,353.4 |
| Softmax, reversed | 1,986.0 | 1,286.2 |

Multiply-add uses one arithmetic kernel. Softmax schedules reduction operands and dependent reduction/arithmetic passes. Both destination paths stage a complete result before copying. The [retained output](benchmark-runs/2026-10-18-expressions-metal.txt) includes first-call timings and all cases.

## Reproduce

```sh
sbcl --noinform --non-interactive --load tests/expressions-bench.lisp
```

The script validates each case against an independent host reference. Warm timing is the median of three ten-call averages after three warm-up calls. Every invocation waits for completion and releases its allocating result. Validation downloads and input construction are outside the timed section.

## Limitations

- First-call timings include kernel preparation, but the compiler's persistent cache and composed-operation kernels can be shared with earlier runs/cases. They are first invocations, not guaranteed uncached compilation timings.
- These results cover one device, dtype and shape; they establish no general speedup guarantee.
- Full-shape reduction temporaries and masks remain tracked in [#83](https://github.com/communal-software/cl-gpu/issues/83).

[^measurement]: Plans are constructed outside timing. Expression allocating/destination cases share a plan, so the destination case can reuse computation pipelines. Composed operations use the existing elementwise/reduction caches. Timed calls include Lisp scheduling and buffer allocation; host validation does not. The complete-call timer uses `get-internal-real-time`.
