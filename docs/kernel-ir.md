# Kernel IR

`lower-kernel` turns a kernel form into typed, C-like IR. A [printer](printers.md) writes the IR out for one target.

```lisp
(cl-gpu:lower-kernel
 (first (cl-gpu:read-kernel-from-string
         "(let ((i (global-id 0)))
            (when (< i n)
              (setf (aref out i) (* 2.0 (aref in i)))))"))
 :arguments '((out (ptr f32 :global)) (in (ptr f32 :global)) (n u32)))
```

## Entry point

| Argument | Meaning |
|---|---|
| `form` | a CST from `read-kernel`, or a plain form |
| `:arguments` | `(name type)` for each kernel parameter |
| `:name` | kernel name, default `kernel` |

The result is an `ir-kernel` with `parameters`, `body` (a list of statements) and `scalar-types`. A kernel returns nothing; the value of its last form is discarded, and `(return)` leaves the kernel.

Type errors are `kernel-type-error`, with the location of the offending form.

## Statements

| Node | C |
|---|---|
| `ir-declare` | `T v = init;` |
| `ir-array` | `T storage[n];` and a pointer to it. A shared array is lifted to kernel scope |
| `ir-assign` | `target = value;` |
| `ir-if` | `if (test) {...} else {...}` |
| `ir-for` | `for (T i = init; test; step) {...}` |
| `ir-while` | `while (test) {...}` |
| `ir-break`, `ir-return` | `break;`, `return;` |
| `ir-expr-stmt` | a call for its effect, such as a barrier |

## Expressions

Every expression has an `ir-expr-type`.

| Node | C |
|---|---|
| `ir-literal`, `ir-var` | constant, variable |
| `ir-op` | operator, named by `operator` (`:add`, `:lt`, `:and`, `:bit-xor`, `:shl`, ...). `:add` and `:sub` also offset and subtract pointers |
| `ir-ternary` | `test ? a : b` |
| `ir-index` | `p[i]` |
| `ir-component` | `v.x` for index 0 |
| `ir-swizzle` | `v.zx`, a read of 1 to 4 components |
| `ir-construct` | vector constructor. With one argument, a scalar broadcast |
| `ir-convert` | cast, or bit reinterpretation |
| `ir-intrinsic` | thread position, barrier, atomic or math call |

Variables are `ir-variable` objects. Two variables with one name are different objects, and each has an `id` unique within the kernel. A printer spells them, for example as `name_id`.

## Lowering

Lisp forms yield values; C statements do not. Lowering splits them.

| Form in value position | Lowered to |
|---|---|
| `if` with simple branches | `ir-ternary` |
| `if` with branches that need statements | a result variable assigned in each branch |
| `let`, `progn` | its statements, then its last form |
| `block` | a loop that runs once, with a result variable |
| `setf`, `incf`, `decf` | the assignment, then the variable |
| atomic call | a variable bound to the call |

Operands keep left-to-right order: an operand evaluated before a later operand's statements is bound to a variable first. `and` and `or` short-circuit; an operand that needs statements runs only when reached.

### Returns

| Return | Lowered to |
|---|---|
| `return` from the kernel | `ir-return` |
| `return-from` the innermost loop | `ir-break` |
| `return-from` an outer loop or block | set a flag, `break`, and test the flag after each inner loop |

Metal has no `goto`, so flags carry an exit across loops.

A loop has no value. Use `block` for a value.

## Limitations

- Swizzles are read-only ([#22](https://github.com/communal-software/cl-gpu/issues/22)).
- The IR allows vector arithmetic and one-argument broadcasts that CUDA needs helpers for ([#12](https://github.com/communal-software/cl-gpu/issues/12)).
