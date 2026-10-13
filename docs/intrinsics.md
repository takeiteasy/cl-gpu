# Intrinsics

Intrinsics are calls the compiler spells per target. `find-intrinsic` returns an entry and `intrinsic-spelling` gives its text for a target.

```lisp
(cl-gpu:intrinsic-spelling (cl-gpu:find-intrinsic 'cl-gpu:global-id) :cuda)
;; => :EXPR, "blockIdx.{axis} * blockDim.{axis} + threadIdx.{axis}", NIL
```

## Thread positions

The axis is a literal 0, 1 or 2. The result is `u32`.

| Intrinsic | Meaning |
|---|---|
| `(global-id axis)` | thread index in the whole grid |
| `(local-id axis)` | thread index in its group |
| `(group-id axis)` | group index in the grid |
| `(group-size axis)` | threads per group |
| `(grid-size axis)` | threads in the grid |

## Barrier

`(barrier [scope])` waits for the group. The scope is `:shared` (default), `:global` or `:all`. It has no value.[^1]

## Atomics

Operands are `(ptr (atomic T) :global)` or `:shared`. The result is the old value.

| Intrinsic | Operation |
|---|---|
| `atomic-add` `atomic-sub` `atomic-min` `atomic-max` | `(op ptr value)` |
| `atomic-and` `atomic-or` `atomic-xor` `atomic-exchange` | `(op ptr value)` |
| `atomic-cas` | `(atomic-cas ptr expected desired)` |
| `atomic-load` | `(atomic-load ptr)` |
| `atomic-store` | `(atomic-store ptr value)`, no value |

An atomic in value position runs where it is written and its result is bound to a variable.

## Math

| Intrinsic | Operands |
|---|---|
| `sqrt` `rsqrt` `exp` `log` `sin` `cos` `tan` `floor` `ceiling` | one float or float vector |
| `pow` | two floats |
| `fma` | three floats |
| `abs` | one numeric |
| `min` `max` | two numerics of one type |
| `clamp` | three numerics of one type |

`floor` and `ceiling` return a float of the operand's type.

## Remainders and pointer offsets

`rem`, `mod`, `ptr+` and `ptr-` are operators, described under [types](types.md#remainders). Float `rem` and signed or float `mod` lower to intrinsics.

## Conversions

| Form | Meaning |
|---|---|
| `(convert type x)` | value cast between numeric scalars, or between vectors of one length |
| `(bitcast type x)` | reinterpret bits between types of one size |
| `(vec type a b ...)` | build a vector from 2 to 4 components |
| `(aref v 0)` | vector component, the index is a literal |

## Spellings

A spelling is `(kind text . options)`.

| Kind | Meaning |
|---|---|
| `:call` | call the function `text`, then the strings of `:extra` |
| `:expr` | an expression; `{axis}` stands for `x`, `y` or `z` |
| `:param` | a Metal kernel parameter attribute, indexed by axis |
| `:helper` | a function the printer defines |

Options `:f16`, `:f32` and `:f64` replace `text` for that element type. Barrier options name the memory flags by scope.

[^1]: Metal's `threadgroup_barrier` takes `mem_flags`, OpenCL's `barrier` takes `CLK_*_MEM_FENCE`, and CUDA's `__syncthreads` ignores the scope.
