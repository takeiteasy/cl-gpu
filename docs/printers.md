# Printers

A printer writes a lowered kernel as source for one target. Metal is the only target so far; the [Metal runtime](metal.md) runs its output.

```lisp
(cl-gpu:kernel-source
 (cl-gpu:lower-kernel
  (first (cl-gpu:read-kernel-from-string
          "(let ((i (global-id 0)))
             (when (< i n)
               (setf (aref out i) (+ (* a (aref x i)) (aref y i)))))"))
  :name 'saxpy
  :arguments '((out (ptr f32 :global)) (x (ptr f32 :global)) (y (ptr f32 :global))
               (a f32) (n u32)))
 :metal)
```

```metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(
    device float* out [[buffer(0)]],
    device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant float& a [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint3 cl_gpu_thread_position_in_grid [[thread_position_in_grid]]) {
    uint i_6 = cl_gpu_thread_position_in_grid.x;
    if (i_6 < n) {
        out[i_6] = a * x[i_6] + y[i_6];
    }
}
```

For hosts that generate signatures, use [embeddable kernel parts](kernel-parts.md).

## API

| Function | Result |
|---|---|
| `(kernel-source kernel target)` | the source as a string; `target` is `:metal` |
| `(print-kernel kernel printer [stream])` | writes the source; `printer` is an instance such as `(make-instance 'msl-printer)` |

`c-printer` holds what the targets share: statements, operator precedence, literals and names. A target subclasses it, as `msl-printer` does.

`kernel-target-error` is signalled when the target cannot express the kernel: a scalar type it lacks (`f64` on Metal), an intrinsic with no spelling, or a parameter that cannot be bound.

## Names

| Variable | Spelled |
|---|---|
| parameter | its name in lowercase, `-` as `_` (`block-size` is `block_size`) |
| local | `name_id`, such as `i_6` |

A parameter whose name is reserved on the target, or that ends in `_` and a number, gets a trailing `_` (`min` is `min_`). Two parameters that spell alike are an error.

## Metal

Optional operation systems share named MSL helper definitions. A kernel includes only the helpers it uses.[^helpers]

Parameters are bound by position.

| Parameter | Declared | Bound by |
|---|---|---|
| `:global` or `:constant` pointer | `device T* p` or `constant T* p` | `[[buffer(i)]]` |
| scalar or vector | `constant T& n` | `[[buffer(i)]]` |
| `:shared` pointer | `threadgroup T* p` | `[[threadgroup(k)]]` |

`i` counts the buffer-bound parameters in order, and `k` counts the `:shared` ones. A `:private` pointer parameter is an error.

Thread positions are extra `uint3` parameters, named `cl_gpu_` and the attribute, after the kernel's own. A kernel that never calls `global-id` has no `thread_position_in_grid`.

A `shared-array` is declared at the top of the kernel as `threadgroup T cl_gpu_name[n]`, and a `local-array` in place as `thread T cl_gpu_name[n]`. See [arrays](arrays.md).

A scalar or vector parameter that the kernel assigns is passed as `cl_gpu_name` and copied into a local `name`.

Atomics take the pointer as written. `atomic-cas` calls the helper `cl_gpu_atomic_cas`, which the source defines when the kernel uses it.[^1]

## Operators and literals

| IR | Printed |
|---|---|
| nested operators | parentheses only where precedence needs them, and around bitwise, shift and comparison operators mixed with each other |
| `i8` `u8` `i16` `u16` arithmetic | cast back to the type[^2] |
| `u32`, `i64`, `u64` literals | `7u`, `7l`, `7ul` |
| `f16`, `f32` literals | `2.5h`, `1.5f` |
| `bf16` literal | `static_cast<bfloat>(0.5f)` |
| the minimum `i32` or `i64` | `(-2147483647 - 1)` |
| `convert`, `bitcast` | `static_cast<T>(x)`, `as_type<T>(x)` |

[^1]: Metal's compare-exchange takes its expected operand by pointer, overwrites it with the value seen, and may fail spuriously. The helper retries a spurious failure and returns the old value, as CUDA's `atomicCAS` does.
[^2]: C promotes 8- and 16-bit operands to `int`, so `(+ a b)` on `u8` would otherwise exceed 255 where the kernel wraps.

[^helpers]: Internal `cl-gpu::register-msl-helper` accepts a name and source string. Registration replaces that name only; gather and elementwise definitions coexist in either load order. Register during system initialization, before concurrent compilation. Unknown names signal an error.
