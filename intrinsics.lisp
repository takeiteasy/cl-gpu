(in-package #:cl-gpu)

;;; SIGNATURE selects how lowering types a call:
;;;   (:axis)          literal axis 0-2, result u32
;;;   (:barrier)       optional literal scope, no result
;;;   (:float N)       N operands of one float type (or float vector), same result
;;;   (:number N)      the same for any numeric type
;;;   (:atomic KIND)   :rmw (ptr value), :cas (ptr expected desired), :load (ptr), :store (ptr value)
;;;
;;; A spelling is (KIND TEXT . OPTIONS) per target (:metal, :cuda, :opencl); NIL means
;;; the target lacks the intrinsic. KIND is
;;;   :call    call function TEXT with the arguments, then the :extra option's strings
;;;   :expr    TEXT is an expression with {axis} standing for the axis letter x, y or z
;;;   :param   a Metal kernel parameter attribute TEXT, indexed by axis
;;;   :helper  the printer defines function TEXT
;;; The options :f16, :f32 and :f64 override TEXT for that element type.
(defstruct (intrinsic (:constructor make-intrinsic (name signature impure-p spellings)))
  name signature impure-p spellings)

(defvar *intrinsics* (make-hash-table :test 'eq))

(defun find-intrinsic (name)
  (gethash name *intrinsics*))

(defmacro define-intrinsic (name signature &key impure metal cuda opencl)
  "Define the intrinsic NAME. IMPURE marks calls with side effects."
  `(setf (gethash ',name *intrinsics*)
         (make-intrinsic ',name ',signature ,impure
                         (list :metal ',metal :cuda ',cuda :opencl ',opencl))))

(defun intrinsic-spelling (intrinsic target &optional type)
  "Return the spelling kind, text and options of INTRINSIC on TARGET for operands of TYPE."
  (destructuring-bind (&optional kind text &rest options)
      (getf (intrinsic-spellings intrinsic) target)
    (when kind
      (let* ((scalar (typecase type
                       (scalar-type type)
                       (vector-type (vector-type-element type))))
             (override (and scalar
                            (getf options (intern (string-upcase (symbol-name (scalar-type-name scalar)))
                                                  :keyword)))))
        (values kind (or override text) options)))))

;;; Thread positions
(define-intrinsic global-id (:axis)
  :metal (:param "thread_position_in_grid")
  :cuda (:expr "blockIdx.{axis} * blockDim.{axis} + threadIdx.{axis}")
  :opencl (:call "get_global_id"))

(define-intrinsic local-id (:axis)
  :metal (:param "thread_position_in_threadgroup")
  :cuda (:expr "threadIdx.{axis}")
  :opencl (:call "get_local_id"))

(define-intrinsic group-id (:axis)
  :metal (:param "threadgroup_position_in_grid")
  :cuda (:expr "blockIdx.{axis}")
  :opencl (:call "get_group_id"))

(define-intrinsic group-size (:axis)
  :metal (:param "threads_per_threadgroup")
  :cuda (:expr "blockDim.{axis}")
  :opencl (:call "get_local_size"))

(define-intrinsic grid-size (:axis)
  :metal (:param "threads_per_grid")
  :cuda (:expr "gridDim.{axis} * blockDim.{axis}")
  :opencl (:call "get_global_size"))

;;; The scope is :shared (default), :global or :all.
(define-intrinsic barrier (:barrier) :impure t
  :metal (:call "threadgroup_barrier"
           :shared "mem_flags::mem_threadgroup"
           :global "mem_flags::mem_device"
           :all "mem_flags::mem_device_and_threadgroup")
  :cuda (:call "__syncthreads")
  :opencl (:call "barrier"
            :shared "CLK_LOCAL_MEM_FENCE"
            :global "CLK_GLOBAL_MEM_FENCE"
            :all "CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE"))

;;; Atomics. Operands are (ptr (atomic T) :global|:shared); the result is the old value.
(defmacro define-atomic-rmw (name metal cuda opencl)
  `(define-intrinsic ,name (:atomic :rmw) :impure t
     :metal (:call ,metal :extra ("memory_order_relaxed"))
     :cuda (:call ,cuda)
     :opencl (:call ,opencl)))

(define-atomic-rmw atomic-add "atomic_fetch_add_explicit" "atomicAdd" "atomic_add")
(define-atomic-rmw atomic-sub "atomic_fetch_sub_explicit" "atomicSub" "atomic_sub")
(define-atomic-rmw atomic-min "atomic_fetch_min_explicit" "atomicMin" "atomic_min")
(define-atomic-rmw atomic-max "atomic_fetch_max_explicit" "atomicMax" "atomic_max")
(define-atomic-rmw atomic-and "atomic_fetch_and_explicit" "atomicAnd" "atomic_and")
(define-atomic-rmw atomic-or "atomic_fetch_or_explicit" "atomicOr" "atomic_or")
(define-atomic-rmw atomic-xor "atomic_fetch_xor_explicit" "atomicXor" "atomic_xor")
(define-atomic-rmw atomic-exchange "atomic_exchange_explicit" "atomicExch" "atomic_xchg")

;;; Returns the old value. Metal's compare-exchange updates its expected operand instead.
(define-intrinsic atomic-cas (:atomic :cas) :impure t
  :metal (:helper "cl_gpu_atomic_cas")
  :cuda (:call "atomicCAS")
  :opencl (:call "atomic_cmpxchg"))

(define-intrinsic atomic-load (:atomic :load) :impure t
  :metal (:call "atomic_load_explicit" :extra ("memory_order_relaxed"))
  :cuda (:call "atomicAdd" :extra ("0"))
  :opencl (:call "atomic_add" :extra ("0")))

(define-intrinsic atomic-store (:atomic :store) :impure t
  :metal (:call "atomic_store_explicit" :extra ("memory_order_relaxed"))
  :cuda (:call "atomicExch")
  :opencl (:call "atomic_xchg"))

;;; Math. CUDA spells the f32 versions with an f suffix.
(defmacro define-float-math (name metal cuda cuda-f32 opencl &optional (arity 1))
  `(define-intrinsic ,name (:float ,arity)
     :metal (:call ,metal)
     :cuda (:call ,cuda :f32 ,cuda-f32)
     :opencl (:call ,opencl)))

(define-float-math sqrt "sqrt" "sqrt" "sqrtf" "sqrt")
(define-float-math rsqrt "rsqrt" "rsqrt" "rsqrtf" "rsqrt")
(define-float-math exp "exp" "exp" "expf" "exp")
(define-float-math log "log" "log" "logf" "log")
(define-float-math sin "sin" "sin" "sinf" "sin")
(define-float-math cos "cos" "cos" "cosf" "cos")
(define-float-math tan "tan" "tan" "tanf" "tan")
(define-float-math floor "floor" "floor" "floorf" "floor")
(define-float-math ceiling "ceil" "ceil" "ceilf" "ceil")
(define-float-math pow "pow" "pow" "powf" "pow" 2)
(define-float-math fma "fma" "fma" "fmaf" "fma" 3)

(define-intrinsic abs (:number 1)
  :metal (:call "abs")
  :cuda (:call "abs" :f32 "fabsf" :f64 "fabs")
  :opencl (:call "abs" :f16 "fabs" :f32 "fabs" :f64 "fabs"))

(define-intrinsic min (:number 2)
  :metal (:call "min")
  :cuda (:call "min" :f32 "fminf" :f64 "fmin")
  :opencl (:call "min"))

(define-intrinsic max (:number 2)
  :metal (:call "max")
  :cuda (:call "max" :f32 "fmaxf" :f64 "fmax")
  :opencl (:call "max"))

(define-intrinsic clamp (:number 3)
  :metal (:call "clamp")
  :cuda (:helper "cl_gpu_clamp")
  :opencl (:call "clamp"))
