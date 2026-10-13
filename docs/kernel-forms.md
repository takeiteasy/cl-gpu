# Kernel forms

Kernels are Lisp forms. `read-kernel` reads text into forms that carry source locations, and `walk-kernel` turns a form into a tree of resolved AST nodes.

```lisp
(let ((forms (cl-gpu:read-kernel-from-string "(dotimes (i n) (incf (aref out i) 1))")))
  (cl-gpu:walk-kernel (first forms) :variables '(n out)))
```

## Entry points

| Function | Input | Result |
|---|---|---|
| `read-kernel-from-string` | text | list of CSTs |
| `read-kernel` | stream | list of CSTs |
| `read-kernel-file` | pathname | list of CSTs |
| `walk-kernel` | a CST, or a plain form | AST and the bindings of `:variables` |

- Symbols are read into `*package*`, or the `:package` argument. The package must use `cl-gpu` for `for` and `while` to be recognised.
- A plain form has no locations. Its nodes and errors have a `nil` source.
- `:variables` lists the names already in scope, such as kernel arguments. Any other free variable is an error.

## Forms

| Form | Notes |
|---|---|
| `(let ((v init) ...) body)` | `init` sees the outer scope |
| `(let* ((v init) ...) body)` | each `init` sees the earlier variables |
| `(setq v value ...)`, `(setf place value ...)` | pairs become one assign each. A place is a variable, a symbol macro or a call such as `(aref p i)` |
| `(if test then [else])` | |
| `(progn form ...)` | |
| `(block name form ...)`, `(return-from name [value])` | |
| `(for (v init [test [step]]) body)` | C-style loop |
| `(while test body)` | |
| `(macrolet ((name lambda-list body)) form ...)` | local kernel macros |
| `(symbol-macrolet ((name expansion)) form ...)` | `setf` works through them |
| `(f arg ...)` | a call. The IR resolves intrinsics |
| `(declare ...)` | leading form of `let`, `let*`, `for`, `while`; kept as written |

`for` and `while` wrap themselves in a `nil` block, so `(return)` leaves the loop.

## Built-in macros

| Macro | Expands to |
|---|---|
| `when`, `unless`, `cond` | `if` |
| `case` | `let` and `cond` over `=` tests. Keys are numbers |
| `dotimes` | `for` over `0` to the count, evaluated once. No result form |
| `return` | `return-from nil` |
| `incf`, `decf` | `setf`. The subforms of a call place are evaluated once |

`and` and `or` stay calls, so the IR can print `&&` and `||`.

## Defining macros

```lisp
(cl-gpu:define-kernel-macro square (x)
  `(* ,x ,x))
```

- A kernel macro receives the form's arguments and returns a form. It runs on the host.
- Expansions keep the source location of the macro call, so errors point at it.
- `*max-expansion-depth*` (default 256) limits nested expansion.

## Scoping

Every variable gets one `binding` object, and every reference holds that object. Shadowed names are told apart by identity, not by name.

```lisp
(let ((a 1)) (let ((a 2) (b a)) b))
;; b's init is a reference to the outer a's binding
```

Spelling shadowed names for a target language is up to the printer.

## Errors

| Condition | Signalled for |
|---|---|
| `kernel-read-error` | unbalanced or invalid text |
| `kernel-syntax-error` | unsupported forms, bad shapes, unbound variables, unknown blocks |

Both carry a message, the form and a `source-location` (file, line and column, 1-based). They print as `file:line:col: message`.

## Limitations

- Standard CL forms not listed above are errors, including `loop`, `do`, `tagbody`, `go`, `function`, `flet` and `labels`.
- Other CL macros are errors. Define a kernel macro instead.
- `incf` and `decf` through a symbol macro evaluate the place's subforms twice.[^1]

[^1]: `(incf x)` with `x` a symbol macro for `(aref p (f i))` expands to `(setf x (+ x 1))`. Writing the place as a call evaluates its subforms once.
