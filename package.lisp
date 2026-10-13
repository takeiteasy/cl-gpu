(defpackage #:cl-gpu
  (:use #:cl)
  (:local-nicknames (#:tg #:trivial-garbage) (#:bt #:bordeaux-threads))
  (:export
   ;; Kernels and arrays
   #:define-kernel #:find-kernel #:kernel-ir
   #:gpu-array #:make-gpu-array #:to-device #:from-device #:free #:with-gpu-array
   #:gpu-array-length #:gpu-array-element-type #:gpu-array-device
   #:wait
   ;; Backends
   #:*device* #:current-device #:register-backend
   #:device-target #:device-shared-memory-limit #:device-allocate #:device-free #:device-store #:device-load
   #:device-prepare #:device-launch
   ;; Reader
   #:read-kernel
   #:read-kernel-from-string
   #:read-kernel-file
   ;; Walker
   #:walk-kernel
   #:map-nodes
   #:define-kernel-macro
   #:*max-expansion-depth*
   #:for
   #:while
   ;; Types
   #:f16 #:bf16 #:f32 #:f64
   #:i8 #:i16 #:i32 #:i64
   #:u8 #:u16 #:u32 #:u64
   #:bool #:vec #:ptr #:atomic #:ptr+ #:ptr-
   #:parse-type #:type-spec #:type-description #:type-support
   #:gpu-type
   #:scalar-type #:scalar-type-name #:scalar-type-kind #:scalar-type-bits
   #:vector-type #:vector-type-element #:vector-type-length
   #:pointer-type #:pointer-type-pointee #:pointer-type-space
   #:atomic-type #:atomic-type-element
   ;; Intrinsics
   #:global-id #:local-id #:group-id #:group-size #:grid-size #:barrier
   #:atomic-add #:atomic-sub #:atomic-min #:atomic-max
   #:atomic-and #:atomic-or #:atomic-xor #:atomic-exchange
   #:atomic-cas #:atomic-load #:atomic-store
   #:convert #:bitcast #:swizzle #:shared-array #:local-array #:shl #:shr
   #:pow #:rsqrt #:fma #:clamp
   #:find-intrinsic #:intrinsic-name #:intrinsic-signature #:intrinsic-impure-p
   #:intrinsic-spelling
   ;; Lowering
   #:lower-kernel
   #:kernel-type-error
   #:kernel-target-error
   #:map-ir
   #:ir-variable #:ir-variable-name #:ir-variable-type #:ir-variable-id #:ir-variable-temp
   #:ir-kernel #:ir-kernel-name #:ir-kernel-parameters #:ir-kernel-body #:ir-kernel-scalar-types
   #:ir-expr-type
   #:ir-literal #:ir-literal-value
   #:ir-var #:ir-var-variable
   #:ir-op #:ir-op-operator #:ir-op-operands
   #:ir-ternary #:ir-ternary-test #:ir-ternary-then #:ir-ternary-else
   #:ir-index #:ir-index-base #:ir-index-index
   #:ir-component #:ir-component-base #:ir-component-index
   #:ir-swizzle #:ir-swizzle-base #:ir-swizzle-components
   #:ir-construct #:ir-construct-args
   #:ir-convert #:ir-convert-kind #:ir-convert-operand
   #:ir-intrinsic #:ir-intrinsic-intrinsic #:ir-intrinsic-args #:ir-intrinsic-options
   #:ir-declare #:ir-declare-variable #:ir-declare-init
   #:ir-array #:ir-array-variable #:ir-array-size #:ir-array-zero
   #:ir-assign #:ir-assign-target #:ir-assign-value
   #:ir-if #:ir-if-test #:ir-if-then #:ir-if-else
   #:ir-for #:ir-for-variable #:ir-for-init #:ir-for-test #:ir-for-step #:ir-for-body
   #:ir-while #:ir-while-test #:ir-while-body
   #:ir-break #:ir-return
   #:ir-expr-stmt #:ir-expr-stmt-expr
   ;; Printing
   #:kernel-source #:print-kernel #:c-printer #:msl-printer
   ;; Conditions
   #:source-location
   #:source-location-file
   #:source-location-start
   #:source-location-end
   #:source-location-line
   #:source-location-column
   #:gpu-error #:gpu-error-message
   #:kernel-error
   #:kernel-error-form
   #:kernel-error-location
   #:kernel-error-message
   #:kernel-read-error
   #:kernel-syntax-error
   ;; AST
   #:node #:node-source
   #:binding #:binding-name #:binding-source
   #:literal #:literal-value
   #:var-ref #:var-ref-binding
   #:let-node #:let-node-sequential #:let-node-bindings #:let-node-inits
   #:let-node-declarations #:let-node-body
   #:assign #:assign-target #:assign-value
   #:modify #:modify-target #:modify-operator #:modify-delta
   #:if-node #:if-node-test #:if-node-then #:if-node-else
   #:progn-node #:progn-node-body
   #:loop-node #:loop-node-binding #:loop-node-init #:loop-node-test
   #:loop-node-step #:loop-node-declarations #:loop-node-body
   #:block-node #:block-node-name #:block-node-body
   #:return-node #:return-node-block #:return-node-value
   #:call #:call-operator #:call-args))
