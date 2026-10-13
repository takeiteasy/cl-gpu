# Plan

cl-gpu compiles Lisp kernel forms to Metal, CUDA and OpenCL source and runs them on device buffers. It supersedes the GPU-array role of libgpuarray for Common Lisp. [cl-tensor](https://github.com/communal-software/cl-tensor) uses it for device storage.

## Design

| Part | Choice |
|---|---|
| Reader | Eclector |
| Walker | New, compute-only: fixed special forms plus kernel macroexpansion, walking CSTs into AST nodes |
| Types | Closed set: scalars, fixed vectors, typed pointers with address spaces |
| IR | Small C-like tree mapping near 1:1 onto MSL, CUDA C and OpenCL C |
| Printers | Shared C-like base with per-target overrides |
| First target | Metal (testable locally on Apple Silicon) |

The cl-webgpu shader DSL (a 3bgl-shaders fork) informs the inference tests but contributes no code.

## Phases

| # | Deliverable | Ticket | Status |
|---|---|---|---|
| 0 | Skeleton | | Done |
| 1 | Eclector reader and compute-subset walker | [#1](https://github.com/communal-software/cl-gpu/issues/1) | Done |
| 2 | Type set and kernel IR | [#2](https://github.com/communal-software/cl-gpu/issues/2) | Open |
| 3 | C-like printer base and MSL printer | [#3](https://github.com/communal-software/cl-gpu/issues/3) | Open |
| 4 | Metal runtime | [#4](https://github.com/communal-software/cl-gpu/issues/4) | Open |
| 5 | `define-kernel` and buffer API | [#5](https://github.com/communal-software/cl-gpu/issues/5) | Open |
| 6 | CUDA backend | [#6](https://github.com/communal-software/cl-gpu/issues/6) | Open |
| 7 | OpenCL backend | [#7](https://github.com/communal-software/cl-gpu/issues/7) | Open |

## Limitations

No kernels compile yet: the walker produces an AST that nothing consumes.

- No `loop`, `do`, `tagbody` or `go` ([kernel forms](kernel-forms.md)).
- No `flet` or `labels` ([#8](https://github.com/communal-software/cl-gpu/issues/8)).
- `incf` and `decf` through a symbol macro evaluate the place's subforms twice ([#9](https://github.com/communal-software/cl-gpu/issues/9)).
