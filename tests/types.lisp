(in-package #:cl-gpu/tests)

(in-suite :cl-gpu)

(test parses-scalar-types
  (dolist (name '("f16" "bf16" "f32" "f64" "i8" "i16" "i32" "i64"
                  "u8" "u16" "u32" "u64" "bool"))
    (is (typep (type-of-spec name) 'gpu:scalar-type))
    (is (string-equal name (symbol-name (gpu:scalar-type-name (type-of-spec name)))))))

(test compound-types-are-interned
  (is (eq (type-of-spec "(vec f32 4)") (type-of-spec "(vec f32 4)")))
  (is (not (eq (type-of-spec "(vec f32 4)") (type-of-spec "(vec f32 3)"))))
  (is (eq (type-of-spec "(ptr f32 :global)") (type-of-spec "(ptr f32 :global)")))
  (is (eq (type-of-spec "(ptr (atomic u32) :shared)")
          (type-of-spec "(ptr (atomic u32) :shared)"))))

(test type-specs-round-trip
  (dolist (text '("f32" "(vec i8 2)" "(ptr (vec f32 4) :shared)" "(ptr (atomic i32) :global)"))
    (let ((type (type-of-spec text)))
      (is (eq type (gpu:parse-type (gpu:type-spec type)))))))

(test describes-types-for-messages
  (is (string= "(ptr f32 :global)" (gpu:type-description (type-of-spec "(ptr f32 :global)"))))
  (is (string= "nothing" (gpu:type-description nil))))

(test rejects-bad-types
  (dolist (text '("float" "(vec f32 5)" "(vec bool 2)" "(vec f32)" "(ptr f32 :nowhere)"
                  "(ptr (ptr f32 :global) :global)" "(atomic f32)" "(atomic)" "(array f32 4)" "3"))
    (signals gpu:kernel-type-error (type-of-spec text))))

(test type-predicates-look-through-vectors
  (is (gpu::float-type-p (type-of-spec "(vec f32 2)")))
  (is (gpu::integer-type-p (type-of-spec "u8")))
  (is (gpu::signed-type-p (type-of-spec "i32")))
  (is (gpu::unsigned-type-p (type-of-spec "(vec u32 4)")))
  (is (not (gpu::numeric-type-p (type-of-spec "bool"))))
  (is (not (gpu::numeric-type-p (type-of-spec "(ptr f32 :global)")))))

(test per-target-support
  (is (eq t (gpu:type-support (type-of-spec "f32") :metal)))
  (is (null (gpu:type-support (type-of-spec "f64") :metal)))
  (is (eq t (gpu:type-support (type-of-spec "f64") :cuda)))
  (is (string= "cl_khr_fp64" (gpu:type-support (type-of-spec "f64") :opencl)))
  (is (string= "cl_khr_fp16" (gpu:type-support (type-of-spec "f16") :opencl)))
  (is (null (gpu:type-support (type-of-spec "bf16") :opencl)))
  (is (eq t (gpu:type-support (type-of-spec "bf16") :metal))))

(test type-names-read-as-literals-in-kernels
  (let ((node (walk-text "(convert f32 1)")))
    (is (typep (first (gpu:call-args node)) 'gpu:literal))
    (is (eq (ksym "f32") (gpu:literal-value (first (gpu:call-args node)))))))

(test variables-shadow-type-names
  (is (typep (walk-text "(let ((f32 1)) f32)") 'gpu:let-node)))
