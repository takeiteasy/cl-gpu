# cl-gpu

> **Work in progress.** This project is under development; expect missing features and breaking changes.

S-expression GPU kernel DSL for Common Lisp with Metal, CUDA and OpenCL backends.

See the [plan](docs/PLAN.md) for status and phases. To run kernels, see [kernels and arrays](docs/kernels.md). For kernel internals: [forms](docs/kernel-forms.md), [types](docs/types.md), [intrinsics](docs/intrinsics.md), the [IR](docs/kernel-ir.md), the [printers](docs/printers.md) and the [Metal runtime](docs/metal.md).

## Tests

```sh
sbcl --noinform --non-interactive --eval '(asdf:test-system "cl-gpu")'
```

On macOS, the Metal runtime has its own tests:

```sh
sbcl --noinform --non-interactive --eval '(asdf:test-system "cl-gpu/metal")'
```

## License

```
cl-gpu
Copyright (C) 2026 George Watson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```
