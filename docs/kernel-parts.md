# Embeddable Metal kernels

`kernel-parts` returns a kernel header, body and ordered binding descriptors for hosts that generate their own function signatures. It depends only on core cl-gpu; it does not run the Metal runtime or depend on MLX.

```lisp
(let* ((kernel (cl-gpu:find-kernel 'copy-values))
       (parts (cl-gpu:kernel-parts kernel :metal)))
  (values (cl-gpu:kernel-parts-header parts)
          (cl-gpu:kernel-parts-body parts)
          (mapcar #'cl-gpu:kernel-parameter-name
                  (cl-gpu:kernel-parts-inputs parts))))
```

The first argument accepts a kernel object or lowered `ir-kernel`. The target is `:metal`.

## Result accessors

| Accessor, prefixed `kernel-parts-` | Contents |
|---|---|
| `header` | Metal prelude and required helper definitions |
| `body` | Prologue and statements without the function signature or enclosing braces |
| `parameters` | Binding descriptors in original declaration order |
| `inputs` | Read-only buffers and scalar/vector value descriptors |
| `outputs` | Write-only and initialized read/write buffer descriptors |
| `attributes` | Sorted Metal thread-attribute names used by the body |
| `precise-math` | Kernel object's math requirement; nil for a bare IR |

Descriptors expose `kernel-parameter-variable`, `kernel-parameter-name` and `kernel-parameter-mode`. Modes are `:input`, `:value`, `:output` and `:inout`. Names are the emitted host binding names, including scalar-copy names. Types come from `ir-variable-type` on the descriptor's variable.

Pointer aliases and arithmetic participate in access analysis. Analysis conservatively combines possible aliases. Pass `:initialized-outputs t` only when the host supplies initialized output storage; otherwise a read/write output signals `kernel-target-error`.

## Host integration

The body uses Metal's exact thread names, such as `thread_position_in_grid.x`. Hosts bind the listed attributes under those names. Static shared arrays stay at kernel scope, including cooperative zeroing and its barrier. Assigned scalar/vector values receive writable local copies.

The optional `:printer` argument accepts an `embedded-msl-printer` subclass for host-specific parameter and storage adaptation. Standalone `kernel-source` and Metal runtime dispatch retain their usual signature and buffer layout. [cl-mlx/gpu](https://github.com/communal-software/cl-mlx/blob/trunk/docs/gpu.md) supplies the MLX binding and packed-vector adapter.[^host]

## Limitations

- Metal is the only embeddable target.
- Shared/private pointer parameters are rejected; use static arrays declared in the kernel body.
- Writing through constant pointers is rejected.
- Hosts must honor initialization, math requirements, scalar/vector layouts and launch bounds. The default parts printer uses native Metal vector layout.

[^host]: A host may generate scalar buffer parameters for vector DSL types, or choose an input address space based on array size. Such hosts need an adapter; the default body is not automatically compatible with every host signature.
