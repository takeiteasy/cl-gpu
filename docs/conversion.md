# Device type conversion

`cl-gpu/elementwise:convert` converts a strided input into independent contiguous storage. `convert!` writes into an existing view of the same shape and uses its element type.

```lisp
(cl-gpu/elementwise:convert input 'cl-gpu:i8 :rounding :floor)
(cl-gpu/elementwise:convert! output input :rounding :truncate)
```

Both return the result view. Broadcasting applies to [elementwise arithmetic](elementwise.md); conversion preserves the input shape. Conversion uses the same destination layout, alias and ownership rules as arithmetic.

## Types and ranges

Supported types are f32, f16, bf16 and signed/unsigned 8-, 16-, 32- and 64-bit integers. Conversions involving f16/bf16 require both types to be real floating formats. Other integer/float pairs are supported.

| Direction | Behavior |
|---|---|
| Integer to integer | Clamp to the destination range |
| Integer to f32 | Round once to nearest-even, directly from the integer |
| f32 to integer | Apply rounding, then clamp to the destination range |
| Floating format to another floating format | Widen exactly or narrow with rounding |
| Identical types | Copy raw bits, including signed zeros and NaN payloads |

Positive and negative infinity clamp to the corresponding integer bound. NaN-to-integer conversion signals `gpu-error`; a data-dependent error can leave destination elements updated. These checked calls synchronize before returning.

## Rounding

`:rounding` accepts `:nearest-even` (default), `:truncate`, `:floor` and `:ceiling`. It controls float-to-integer conversion and f16/bf16 narrowing. Other conversions accept and validate the option but ignore it.

| Mode | Inexact finite value | Positive finite overflow to f16/bf16 | Negative finite overflow to f16/bf16 |
|---|---|---|---|
| `:nearest-even` | Nearest, ties to even | +infinity | -infinity |
| `:truncate` | Toward zero | Largest finite value | Negative largest finite value |
| `:floor` | Toward negative infinity | Largest finite value | -infinity |
| `:ceiling` | Toward positive infinity | +infinity | Negative largest finite value |

Floating widening preserves subnormals, infinities and signed zeros. Cross-format NaNs become quiet NaNs with the original sign and the leading payload bits that fit. Exact values remain unchanged. Directed rounding includes subnormal and overflow boundaries.[^bits]

## Limitations

- The supported format pairs exclude f64, complex values and direct f16/bf16-to-integer or integer-to-f16/bf16 calls.
- Tensor-level conversion requires the device adapter work listed in [elementwise limitations](elementwise.md#limitations).

[^bits]: Conversion computes rounding and format fields with integer operations, independently of the host floating-point environment. It uses no host element staging and no full-size intermediate format buffer. Overlapping destinations may still need device scratch.
