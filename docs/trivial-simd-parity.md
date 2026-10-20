# Trivial-simd feature parity

cl-gpu targets trivial-simd's numerical capabilities and practical convenience APIs while retaining device-native execution. Shared operation families use comparable names and argument conventions where their storage contracts allow it.

The published CPU API and open issue inventory define this comparison.[^audit]

## Shared behavior

Both libraries provide real/integer elementwise arithmetic, comparisons, selection, conversion and signed-stride layouts. Complex arithmetic, equality, selection and checked conversion share c32/c64 component precision. GPU arrays retain explicit device placement and ownership.

The target includes optional CPU-style bulk slices/strides and the BLAS convenience wrappers. Core device APIs remain available independently; wrappers reuse them rather than maintaining separate implementations.

## Deliberate device differences

| Area | GPU contract |
|---|---|
| Storage | GPU arrays and bounded views; host transfers are explicit rather than implicit foreign-memory access |
| Precision | Computation follows actual device capabilities; native Metal f64 is unavailable |
| Scalar types | GPU cores coerce accepted scalars to the established dtype; CPU bulk calls require exact scalar types. Compatibility wrappers review stricter checks |
| Encoded floats | GPU arrays carry f16/bf16 types; CPU u16 storage uses explicit encoding keywords |
| Execution | Calls retain their documented queue/completion contracts; host-scalar convenience returns make waits explicit |
| Results | Axis reductions return device views and logical arg indices; CPU bulk reductions return host scalars and physical vector indices. Wrappers review those differences and empty extrema explicitly |
| Numerical comparison | Compare permitted arithmetic/reduction reorderings with tolerances; bit-preserving copies, selection and conversion retain exact references |

Parity does not require identical exception classes, host fallback, CPU-specific backend switches or matching performance. CPU ISA optimizations are implementation details. cl-gpu also exposes encoded-half arithmetic, axis mean and normalization/gather APIs beyond the CPU core's standalone surface.

## Limitations

The tables track missing public capabilities and interfaces, not completed GPU support. Each ticket requires interface review, protocol tests, independent device references and current documentation. General parity and convenience work stays deferred in the inference roadmap unless a selected model establishes a prerequisite.

### Additional tracked gaps

| Capability | Ticket | Next action |
|---|---|---|
| Masks: count, any and all reductions | [#94](https://github.com/communal-software/cl-gpu/issues/94) | Add nonzero-byte count/any/all with empty identities and explicit scalar-return policy. |
| Reductions: integer and complex dtype coverage and missing core reducers | [#95](https://github.com/communal-software/cl-gpu/issues/95) | Extend reducer dtype/operator coverage; distinguish core complex magnitudes from BLAS absolute sums. |
| Reductions: selectable f64 accumulation for single-precision inputs | [#96](https://github.com/communal-software/cl-gpu/issues/96) | Review widened sums/dots/norms on native-f64 devices; preserve explicit unsupported behavior elsewhere. |
| Complex elementwise sqrt and reciprocal parity | [#97](https://github.com/communal-software/cl-gpu/issues/97) | Add principal complex sqrt and reciprocal using reviewed component-precision numerical helpers. |
| Expressions: integer and complex types, mask results and reducer parity | [#98](https://github.com/communal-software/cl-gpu/issues/98) | Extend high-level expression dtype/results and missing reducers; reuse standalone primitives and lazy staging. |
| Expressions: typed integer loads and repeated block inputs | [#99](https://github.com/communal-software/cl-gpu/issues/99) | Add converted/repeated input descriptors without full-size expanded device buffers. |
| Expressions: native f64 operands, constants and reductions | [#100](https://github.com/communal-software/cl-gpu/issues/100) | Supply the missing upstream native-f64 fusion capability for cl-tensor #32. |
| Device copy, fill and swap for all numeric storage types | [#101](https://github.com/communal-software/cl-gpu/issues/101) | Provide raw-preserving numeric storage operations beyond the BLAS float subset. |
| Bulk convenience APIs matching trivial-simd slices, strides and scalar calls | [#102](https://github.com/communal-software/cl-gpu/issues/102) | Add optional CPU-style bulk wrappers, including slices/strides, FMA and destination-established scalar calls. |
| Mixed f32 activation and f16/bf16 weight matmul | [#103](https://github.com/communal-software/cl-gpu/issues/103) | Enable direct GPU inference from encoded weight rows without full f32 weight expansion. |
| Packed GGUF Q8_0, Q4_0, Q4_K and Q6_K device matmul | [#104](https://github.com/communal-software/cl-gpu/issues/104) | Add device-resident packed GGUF products for inference; validate formats independently and retain packed weights. |
| BLAS: f64 structured, rank-update and triangular/compact routine parity | [#105](https://github.com/communal-software/cl-gpu/issues/105) | Cover the double-precision families outside #55/#71; reuse reviewed f32 family contracts. |
| Gather: f64 and complex value storage coverage | [#106](https://github.com/communal-software/cl-gpu/issues/106) | Close the existing tensor f64 gather prerequisite and preserve complex component bits. |
| Elementwise min/max signed-zero ties diverge from CPU left-operand semantics | [#107](https://github.com/communal-software/cl-gpu/issues/107) | Preserve left-operand signed-zero ties in elementwise/fused min/max; Metal divergence is reproduced with exact bits. |

The gather dtype ticket closes an existing tensor-roadmap prerequisite; trivial-simd has no corresponding public gather API. CPU f16/bf16-to-integer conversion is also excluded from its encoded conversion surface, so that shared restriction is not an uncovered parity gap.

### Existing coverage

| Gap | Existing work |
|---|---|
| Double Level 1 and dense matrix/vector/batch BLAS | [#55](https://github.com/communal-software/cl-gpu/issues/55), [#71](https://github.com/communal-software/cl-gpu/issues/71) |
| Complex BLAS, including conjugated dots and Hermitian families | [#39](https://github.com/communal-software/cl-gpu/issues/39) |
| Real/complex rotation variants | [#41](https://github.com/communal-software/cl-gpu/issues/41) |
| Rank-one, symmetric, triangular, band and packed f32 BLAS | [#67](https://github.com/communal-software/cl-gpu/issues/67), [#68](https://github.com/communal-software/cl-gpu/issues/68), [#69](https://github.com/communal-software/cl-gpu/issues/69), [#70](https://github.com/communal-software/cl-gpu/issues/70) |
| Strict BLAS slice/matrix convenience wrappers | [#66](https://github.com/communal-software/cl-gpu/issues/66) |
| Mixed-dot range and extreme-ratio complex division | [#63](https://github.com/communal-software/cl-gpu/issues/63), [#93](https://github.com/communal-software/cl-gpu/issues/93) |
| First-class low-level complex DSL types | [#92](https://github.com/communal-software/cl-gpu/issues/92) |
| OpenCL higher-level execution and shared conformance | [#86](https://github.com/communal-software/cl-gpu/issues/86), [#65](https://github.com/communal-software/cl-gpu/issues/65) |

Native-f64 capability and shared complex storage are available prerequisites; their remaining consumers are not blocked on representation or mandatory Metal emulation. Optional Metal emulation remains [#56](https://github.com/communal-software/cl-gpu/issues/56).

[^audit]: Capability audit: 20 October 2026. CPU references use published trivial-simd commit [`cd61d43`](https://github.com/communal-software/trivial-simd/tree/cd61d43c2986c05ac32503d7619b2e7b67d1b584), including core, N-D, kernel and BLAS documentation and exports. GPU baseline is [`f46ae2c`](https://github.com/communal-software/cl-gpu/tree/f46ae2c4b3495e1c66e41cd339cd0b9bbfd93f7e). The issue inventory is checked against open GitHub tickets; existing family tickets are reused rather than counted as new gaps.
