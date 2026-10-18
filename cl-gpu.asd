;;;; cl-gpu.asd

;;; Objective-C source built into a dylib at compile time and loaded through CFFI.
(defclass objc-dylib (asdf:source-file) ())

(defmethod asdf:source-file-type ((component objc-dylib) system)
  (declare (ignore system))
  "m")

(defmethod asdf:output-files ((operation asdf:compile-op) (component objc-dylib))
  (values (list (asdf:apply-output-translations
                 (make-pathname :type "dylib" :defaults (asdf:component-pathname component))))
          t))

(defmethod asdf:perform ((operation asdf:compile-op) (component objc-dylib))
  (unless (member :darwin *features*)
    (error "~a needs macOS" (asdf:component-name component)))
  (let ((output (first (asdf:output-files operation component))))
    (ensure-directories-exist output)
    (uiop:run-program (list "xcrun" "clang" "-dynamiclib" "-x" "objective-c" "-fno-objc-arc"
                            "-framework" "Metal" "-framework" "MetalPerformanceShaders"
                            "-framework" "Foundation"
                            (uiop:native-namestring (asdf:component-pathname component))
                            "-o" (uiop:native-namestring output))
                      :output :interactive :error-output :interactive)))

(defmethod asdf:perform ((operation asdf:load-op) (component objc-dylib))
  (uiop:symbol-call :cffi :load-foreign-library
                    (first (asdf:output-files (asdf:make-operation 'asdf:compile-op) component))))

(asdf:defsystem "cl-gpu"
  :description "S-expression GPU kernel DSL with Metal, CUDA and OpenCL backends"
  :author "George Watson <gigolo@hotmail.co.uk>"
  :license "GPL-3.0-only"
  :version "0.0.0"
  :depends-on ("eclector" "eclector-concrete-syntax-tree" "concrete-syntax-tree"
               "trivial-garbage" "float-features" "bordeaux-threads")
  :serial t
  :components ((:file "package")
               (:file "conditions")
               (:file "types")
               (:file "ast")
               (:file "reader")
               (:file "source")
               (:file "env")
               (:file "walker")
               (:file "macros")
               (:file "ir")
               (:file "intrinsics")
               (:file "lower")
               (:file "printer")
               (:file "msl")
               (:file "device")
               (:file "array")
               (:file "view")
               (:file "kernel") (:file "parts"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/tests"))))

(asdf:defsystem "cl-gpu/tests"
  :depends-on ("cl-gpu" "fiveam")
  :serial t
  :components ((:file "tests/package")
               (:file "tests/smoke")
               (:file "tests/reader")
               (:file "tests/walker")
               (:file "tests/types")
               (:file "tests/intrinsics")
               (:file "tests/lower")
               (:file "tests/printer")
               (:file "tests/msl")
               (:file "tests/kernel") (:file "tests/parts"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/tests :run-tests)
               (error "cl-gpu tests failed"))))

(asdf:defsystem "cl-gpu/metal"
  :description "Metal runtime for cl-gpu"
  :license "GPL-3.0-only"
  :depends-on ("cl-gpu" "cffi" "float-features" "bordeaux-threads")
  :serial t
  :components ((:module "metal"
                :serial t
                :components ((:file "package")
                             (:objc-dylib "shim")
                             (:file "ffi")
                             (:file "runtime")
                             (:file "backend")
                             (:file "mps"))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/metal/tests"))))

(asdf:defsystem "cl-gpu/metal/tests"
  :depends-on ("cl-gpu/metal" "fiveam")
  :components ((:file "tests/metal"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/metal/tests :run-tests)
               (error "cl-gpu/metal tests failed"))))

(asdf:defsystem "cl-gpu/blas"
  :description "BLAS on the GPU for cl-gpu"
  :license "GPL-3.0-only"
  :depends-on ("cl-gpu")
  :serial t
  :components ((:module "blas"
                :serial t
                :components ((:file "package")
                             (:file "matrix-view")
                             (:file "validate")
                             (:file "level1")
                             (:file "level2")
                             (:file "level3"))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/blas/tests"))))

(asdf:defsystem "cl-gpu/reductions"
  :description "Strided axis reductions and normalization on GPU arrays"
  :license "GPL-3.0-only"
  :depends-on ("cl-gpu")
  :serial t
  :components ((:module "reductions"
                :serial t
                :components ((:file "package") (:file "view")
                             (:file "kernels") (:file "routines"))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/reductions/tests"))))

(asdf:defsystem "cl-gpu/reductions/tests"
  :depends-on ("cl-gpu/reductions" "fiveam")
  :components ((:file "tests/reductions"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/reductions/tests :run-tests)
               (error "cl-gpu reductions tests failed"))))

(asdf:defsystem "cl-gpu/reductions/metal/tests"
  :depends-on ("cl-gpu/reductions/tests" "cl-gpu/metal")
  :components ((:file "tests/reductions-metal"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/reductions/tests :run-metal-tests)
               (error "cl-gpu Metal reductions tests failed"))))

(asdf:defsystem "cl-gpu/blas/tests"
  :depends-on ("cl-gpu/blas" "cl-gpu/metal" "trivial-simd/blas" "float-features" "fiveam")
  :components ((:file "tests/blas"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/blas/tests :run-tests)
               (error "cl-gpu/blas tests failed"))))

(asdf:defsystem "cl-gpu/elementwise"
  :description "Broadcast elementwise operations and device type conversion"
  :license "GPL-3.0-only"
  :depends-on ("cl-gpu")
  :serial t
  :components ((:module "elementwise" :serial t
                :components ((:file "package") (:file "conversion")
                             (:file "kernels") (:file "routines"))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/elementwise/tests"))))

(asdf:defsystem "cl-gpu/elementwise/tests"
  :depends-on ("cl-gpu/elementwise" "fiveam")
  :components ((:file "tests/elementwise"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/elementwise/tests :run-tests)
               (error "cl-gpu elementwise tests failed"))))

(asdf:defsystem "cl-gpu/elementwise/metal/tests"
  :depends-on ("cl-gpu/elementwise/tests" "cl-gpu/metal")
  :components ((:file "tests/elementwise-metal"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/elementwise/tests :run-metal-tests)
               (error "cl-gpu Metal elementwise tests failed"))))

(asdf:defsystem "cl-gpu/expressions"
  :description "Reusable fused expressions over strided device views"
  :license "GPL-3.0-only"
  :depends-on ("cl-gpu/elementwise" "cl-gpu/reductions")
  :serial t
  :components ((:module "expressions" :serial t
                :components ((:file "package") (:file "plan") (:file "kernels") (:file "execute"))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-gpu/expressions/tests"))))

(asdf:defsystem "cl-gpu/expressions/tests"
  :depends-on ("cl-gpu/expressions" "fiveam")
  :components ((:file "tests/expressions"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/expressions/tests :run-tests)
               (error "cl-gpu expression tests failed"))))

(asdf:defsystem "cl-gpu/expressions/metal/tests"
  :depends-on ("cl-gpu/expressions/tests" "cl-gpu/metal")
  :components ((:file "tests/expressions-metal"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-gpu/expressions/tests :run-metal-tests)
               (error "cl-gpu Metal expression tests failed"))))
