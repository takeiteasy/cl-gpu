# Broadcast elementwise operations

`cl-gpu/elementwise` applies operations to [strided views](strided-views.md) on one GPU. Inputs stay on the device; allocating calls return independent contiguous storage.

```lisp
(asdf:load-system "cl-gpu/metal")
(asdf:load-system "cl-gpu/elementwise")

(cl-gpu:with-gpu-array (data (cl-gpu:to-device #(1f0 2f0 3f0 10f0 20f0)))
  (let* ((rows (cl-gpu:make-strided-view data '(2 1) :offset 3))
         (columns (cl-gpu:make-strided-view data '(1 3)))
         (result (cl-gpu/elementwise:add rows columns)))
    (unwind-protect
         (cl-gpu:from-device (cl-gpu:strided-view-data result))
      (cl-gpu:free (cl-gpu:strided-view-data result)))))
;; => #(11.0 12.0 13.0 21.0 22.0 23.0)
```

## Calls

Names below belong to `cl-gpu/elementwise`. Every allocating operation has a destination-first `!` form: `(add! out a b)` writes and returns `out`.

| Calls | Inputs |
|---|---|
| `add`, `subtract`, `multiply`, `divide`, `min`, `max` | Two numeric operands |
| `negate`, `abs` | One numeric operand |
| `sqrt`, `reciprocal`, `exp`, `log`, `sin`, `cos`, `tanh` | One floating operand; angles use radians |
| `sigmoid`, `silu`, `gelu` | One floating operand |
| `clamp` | Input, lower bound, upper bound |
| `(compare operator a b)` | Operator is `:eq`, `:ne`, `:lt`, `:le`, `:gt` or `:ge`; result is a u8 mask |
| `(select mask on-true on-false)` | A u8 view mask; any nonzero byte selects `on-true` |
| `convert`, `convert!` | See [type conversion](conversion.md) |

Operands are views or Lisp real scalars. At least one numeric view establishes the element type and device; the mask does not establish the numeric type. Numeric views share one element type. Scalars must fit that type; floating scalars convert to its precision. Mixed view types require explicit conversion.

Supported elements are `f16`, `bf16`, `f32`, `f64`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64` and `u64`. Floating operations on f16/bf16 widen to f32, compute there and round once on storage. f64 operands, scalars and intermediates stay in double precision on [native-capable devices](native-f64.md).[^math]

## Broadcasting and destinations

Dimensions align from the right. Equal dimensions match; a dimension of 1 expands to its peer, including a peer of 0. Incompatible dimensions signal `gpu-error`. A rank-zero view is a scalar; any zero dimension makes the result empty.

Inputs support signed strides, transposes, padding and zero-stride broadcasts. Destination shape equals the broadcast result shape. Its type and device match the operation, and each logical output element addresses distinct storage. Padding stays unchanged. Overlapping input/output ranges use device scratch before scattering into the destination, including exact in-place calls.

Metadata and options validate before writes, including empty calls. Shapes, offsets, strides and indexing fit signed 64-bit arithmetic. Freeing an owning GPU array invalidates its views.

## Integer arithmetic and execution

Integer add, subtract, multiply, negate and abs wrap to the element width. Division truncates toward zero; signed minimum divided by -1 wraps to the signed minimum. A zero integer divisor signals `gpu-error`. Clamp signals `gpu-error` when a lower bound exceeds its upper bound.

Ordinary calls queue device work. Integer division, float-to-integer conversion and clamp calls with view bounds read a small error flag and wait before returning. Data-dependent errors may leave some destination elements updated. Host transfers and `cl-gpu:synchronize` wait for queued work. Scratch and metadata storage follow the runtime's queued-buffer lifetime rules.

The [cl-tensor device adapter](https://github.com/communal-software/cl-tensor/blob/trunk/docs/device.md) exposes elementwise operations through its tensor APIs.

## Tests and example

```lisp
(asdf:test-system "cl-gpu/elementwise")
(asdf:test-system "cl-gpu/elementwise/metal/tests")
(load "examples/elementwise.lisp")
```

Host tests cover layouts, validation, generated kernels, launch boundaries and cleanup. Metal tests cover arithmetic, masks, aliases, wrapping integers and conversion against exact references, including every f16/bf16 bit pattern.

## Limitations

- Each nonempty call uploads layout metadata. Metal uploads wait for earlier device work ([#49](https://github.com/communal-software/cl-gpu/issues/49)).
- Generic indexing uses O(rank) coordinate decoding per element; contiguous specialization and axis coalescing are tracked in [#76](https://github.com/communal-software/cl-gpu/issues/76).
- Irregular destination uniqueness may require O(output elements) host workspace ([#75](https://github.com/communal-software/cl-gpu/issues/75)).
- Complex, bool and vector elements are outside this API. Native Metal f64 computation is unsupported; see the [support matrix](native-f64.md#support).

[^math]: Kernels use precise math. Floating arithmetic and transcendental exceptional values follow the selected backend. OpenCL f64 kernels use precise math; comparisons involving NaN are false except `:ne`. Selection preserves the chosen value bits. Arithmetic does not promise NaN payload preservation. Sigmoid uses a sign-dependent formula to avoid exponential overflow; SiLU multiplies by sigmoid. GELU uses `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))`; its cubic intermediate may overflow. Floating results need not match CPU arithmetic bit for bit. Conversion has the explicit rules on its own page.
