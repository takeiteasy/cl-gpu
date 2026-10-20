# Complex arrays and operations

`c32` stores two f32 components; `c64` stores two f64 components. Complex arrays use the existing array, window and strided-view APIs. Computation stays on the selected device.

## Storage and transfers

| Type | Components | Bytes per element |
|---|---|---|
| `cl-gpu:c32` | real f32, imaginary f32 | 8 |
| `cl-gpu:c64` | real f64, imaginary f64 | 16 |

Lengths, offsets and strides count whole complex elements. Components are interleaved, real first. `gpu-array-element-type` retains the complex type; `storage-type` returns its `(vec f32 2)` or `(vec f64 2)` storage type. `gpu-array-storage-view` borrows an array or window with that vector type for explicit kernel binding. It retains the owner and byte offset; freeing it does not free the owner. Ordinary vector arrays have no implicit complex meaning.

```lisp
(cl-gpu:with-gpu-array
    (data (cl-gpu:to-device #(#c(1f0 2f0) #c(3f0 -4f0))))
  (cl-gpu:from-device data))
;; => #(#C(1.0 2.0) #C(3.0 -4.0))
```

Default uploads accept Lisp numbers; real values receive a positive-zero imaginary component. Downloads return Lisp complex values. Explicit `:element-type` selects component precision. Without it, complex input infers c64 when any complex value has double-float components, otherwise c32. Typed empty complex vectors establish precision; real-only and untyped empty inputs retain the f32 default.

| Transfer option | Host format |
|---|---|
| Default | One Lisp number per complex element |
| `:flat t` | Interleaved real/imaginary floating components |
| `:raw t` | Interleaved component bits: specialized u32 vector for c32, u64 for c64 |

`to-device` requires explicit `:element-type` with `:flat` or `:raw`. `copy-into` uses its destination's type. Component sequences contain complete pairs. `from-device :into` accepts a partial destination; flat/raw destinations contain complete pairs too. Raw transfers preserve signed zeros, subnormals and NaN payloads without constructing floating Lisp values.

```lisp
(cl-gpu:to-device #(1f0 2f0 3f0 4f0) :element-type 'cl-gpu:c32 :flat t)
(cl-gpu:to-device (make-array 2 :element-type '(unsigned-byte 32)
                              :initial-contents '(0 #x80000000))
                  :element-type 'cl-gpu:c32 :raw t)
```

Allocation, transfers and windows work for c64 even on devices without f64 computation. Ownership and queued lifetimes follow [kernels and arrays](kernels.md).

## Operations

Names belong to `cl-gpu/elementwise`. Each has an allocating form and a destination-first `!` form. Numeric views share a dtype; use explicit conversion between different view types. Real or complex scalars coerce componentwise to the view's precision.

| Calls | Result dtype |
|---|---|
| `add`, `subtract`, `multiply`, `divide`, `negate`, `conjugate` | Input complex dtype |
| `abs`, `realpart`, `imagpart` | f32 for c32; f64 for c64 |
| `compare :eq`, `compare :ne` | u8 |
| `select` | Selected complex dtype |

At least one numeric view establishes dtype and device. Selection uses a u8 mask; any nonzero byte selects the true value. Equality requires both components equal; inequality negates equality, including for NaNs. Ordering, min/max, clamp, complex transcendental functions and complex reductions are unsupported.

Component extraction returns independent real storage. Extraction, selection and identical-type conversion preserve component bits. Conjugation flips the imaginary sign bit, including for zeros and NaNs. Magnitude uses scaled components, returns positive zero for zero, and gives infinity precedence over NaN. Division scales finite inputs to reduce intermediate overflow.[^arithmetic]

Broadcasting, signed/zero strides, scalar and empty layouts, padding, destination uniqueness and aliases follow [elementwise operations](elementwise.md). Empty calls still validate operation, types and capability.

## Conversion

`convert` and `convert!` accept c32/c64 and every supported real dtype. Shape is preserved. Component conversion uses the existing [rounding and saturation rules](conversion.md), including direct conversion from source precision.

| Direction | Behavior |
|---|---|
| Real to complex | Convert real component; imaginary component is positive zero |
| Complex to complex | Convert both components independently |
| Complex to real | Require original imaginary component equal to zero, then convert real component |
| Identical complex types | Copy component bits |

Both signed imaginary zeros qualify. Nonzero or NaN imaginary components signal `gpu-error`, even if narrowing would round them to zero. NaN real components fail integer conversion. Checked calls read a small device status buffer and synchronize; errors may leave nonaliased destinations partly updated. Aliased calls check scratch results before scattering. Allocation and failure cleanup follow the existing elementwise contract.

## Backend support

| Capability | Metal | OpenCL without f64 | OpenCL with `cl_khr_fp64` |
|---|---|---|---|
| c32/c64 storage and transfers | Yes | Yes | Yes |
| c32 operations | Yes | Yes | Yes |
| c64 operations or conversions involving c64 | Unsupported | Unsupported | Yes |

`device-native-type-p` accepts c32/c64 and checks the corresponding component capability. Missing c64 capability signals `gpu-unsupported-capability` before result/scratch allocation, compilation, metadata upload or launch, including empty calls. Components never silently narrow and operations never fall back to the host.

## Tests and example

```lisp
(asdf:test-system "cl-gpu/elementwise")
(asdf:test-system "cl-gpu/elementwise/metal/tests")
(asdf:test-system "cl-gpu/elementwise/opencl/tests")
(load "examples/complex.lisp")
(cl-gpu/examples/complex:run-example)
```

Protocol tests cover validation and staged cleanup. Metal c32 and native OpenCL c32/c64 tests cover numerical references, transfers, raw bits, layouts, aliases and conversions.

## Limitations

- Complex kernel DSL types are separate; handwritten kernels use explicit vector storage ([#92](https://github.com/communal-software/cl-gpu/issues/92)).
- Division's shared numerator exponent can underflow a much smaller component; full-range per-term scaling and stronger exceptional-value handling remain separate ([#93](https://github.com/communal-software/cl-gpu/issues/93)).
- Complex BLAS and tensor integration remain separate ([#39](https://github.com/communal-software/cl-gpu/issues/39), [cl-tensor#28](https://github.com/communal-software/cl-tensor/issues/28)).
- Native Metal f64 computation is unavailable; optional emulation remains separate ([#56](https://github.com/communal-software/cl-gpu/issues/56)).

[^arithmetic]: Arithmetic uses precise backend math in component precision. Multiplication uses direct component products. Zero-divisor and nonfinite division inputs use direct component formulas; their infinities, NaNs and signed zeros follow backend arithmetic rather than a C99 complex recovery policy. Arithmetic NaN payload preservation is unspecified. Very unequal division components have the ceiling described in [Limitations](#limitations).
