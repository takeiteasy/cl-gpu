# Native double precision

Strided elementwise operations, conversions, reductions and normalization preserve f64 precision on OpenCL devices with `cl_khr_fp64`. Results stay on the selected device. Calls never fall back to f32 or host computation.

## Support

| Capability | OpenCL with `cl_khr_fp64` | OpenCL without it | Metal |
|---|---|---|---|
| f64 storage, explicit transfers and views | Yes | Yes | Yes |
| All existing elementwise operations, comparisons and selection | Native f64 | Unsupported | Unsupported |
| f64 ↔ f32/f16/bf16 | Direct conversion | Unsupported | Unsupported |
| f64 ↔ signed/unsigned 8/16/32/64-bit integers | Direct conversion | Unsupported | Unsupported |
| Sum, product, mean, extrema and arg extrema | Native f64 | Unsupported | Unsupported |
| Softmax and weighted RMSNorm | Native f64 | Unsupported | Unsupported |

An explicitly selected OpenCL CPU device, such as PoCL, is a valid execution device. No operation selects a different device automatically. Encoded half conversions require no native f16/bf16 arithmetic.[^storage]

## Capability checks

```lisp
(cl-gpu:device-native-type-p device 'cl-gpu:f64)
```

The query accepts scalar type symbols or parsed scalar types and returns a boolean. It reports native computation support, independently of allocation and transfers. Unknown backends return false unless they implement the protocol. Software emulation does not qualify as native support.

A missing native f64 capability signals `cl-gpu:gpu-unsupported-capability`, a subtype of `gpu-error`, before allocating results or scratch, compiling kernels, uploading metadata or launching work. Empty calls also check capability. Readers `unsupported-capability-device`, `unsupported-capability-operation` and `unsupported-capability-element-type` identify the missing capability.

## Precision

f64 operands, scalar factors, activation constants, reduction accumulators, shared partials, normalization weights and epsilon use double precision. Results retain f64; comparisons produce u8 and arg reductions produce i64. Mixed numeric view types require explicit conversion.

Conversion uses the [rounding and saturation rules](conversion.md), with one direct rounding step. Floating narrowing honors all four modes, including f64 to f32. Identical-type conversion copies raw bits, including signed zeros and NaN payloads.

Arithmetic uses precise backend math. NaNs, infinities, overflow and underflow follow backend arithmetic; arithmetic NaN payloads are unspecified. Comparisons with NaN are false except `:ne`; selection copies the chosen value. Extrema reductions use strict comparisons and first-index tie handling. Their NaN results depend on traversal and tree order, rather than a universal propagation policy. Softmax with a positive-infinity maximum evaluates infinity minus infinity and may produce NaNs; RMSNorm does not rescale squares to prevent overflow.

Broadcasting, signed/zero strides, windows, aliases, empty-group rules, destination validation and queued lifetimes follow the existing [elementwise](elementwise.md), [reduction](reductions.md) and [normalization](normalization.md) contracts.

## Example

```lisp
(asdf:load-system "cl-gpu/opencl")
(asdf:load-system "cl-gpu/elementwise")
(asdf:load-system "cl-gpu/reductions")
(load "examples/f64.lisp")
(cl-gpu/examples/f64:run-example)
;; => (:ADD #(16777218.0d0 2.0d0) :SUM #(16777218.0d0))
```

The example checks capability before allocating input storage. Bind or pass an explicitly opened capable OpenCL device when the default device lacks native f64.

## Tests

```lisp
(asdf:test-system "cl-gpu/elementwise/opencl/tests")
(asdf:test-system "cl-gpu/reductions/opencl/tests")
```

These suites require a native-f64 device and fail explicitly if none is selected. Linux PoCL validation runs them through the [existing scripts](opencl-linux.md). Host protocol tests cover incapable devices; Metal suites check rejection and existing-type regressions. References include f64 values distinguishable from f32, rational rounding checks and raw floating-format bit patterns.

## Limitations

- Native Metal f64 is unavailable. Optional software emulation is separate ([#56](https://github.com/communal-software/cl-gpu/issues/56)).
- This surface excludes BLAS f64 ([Level 1](blas.md#limitations), [matrix operations](blas-matrices.md#limitations)), gather and fused expressions. Broader OpenCL operation parity remains separate ([#86](https://github.com/communal-software/cl-gpu/issues/86)).
- The tensor adapter's f64 integration is separate ([cl-tensor#27](https://github.com/communal-software/cl-tensor/issues/27)).
- Kernel workgroups use 256 threads and must fit device/kernel limits. Reduction throughput and metadata-upload limits remain in [reduction limitations](reductions.md#limitations).

[^storage]: Conversion kernels bind f16/bf16 buffers as private, non-owning u16 aliases with the same byte layout and owning-buffer lifetime. No full-size floating intermediate or host element staging is used. Alias protection may allocate device scratch.
