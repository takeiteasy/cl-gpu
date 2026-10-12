# cl-gpu

> **Work in progress.** This project is under development; expect missing features and breaking changes.

S-expression GPU kernel DSL for Common Lisp with Metal, CUDA and OpenCL backends.

See the [plan](docs/PLAN.md) for status and phases.

## Tests

```sh
sbcl --noinform --non-interactive --eval '(asdf:test-system "cl-gpu")'
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
