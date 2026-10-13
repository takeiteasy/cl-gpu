(in-package #:cl-gpu)

;;; The protocol a backend implements for its device class.

(defgeneric device-target (device)
  (:documentation "The printer target keyword whose source DEVICE compiles."))

(defgeneric device-shared-memory-limit (device)
  (:documentation "Bytes of group-shared memory a kernel may use on DEVICE, or NIL if unchecked.")
  (:method (device)
    (declare (ignore device))
    nil))

(defgeneric device-allocate (device bytes)
  (:documentation "A zeroed backend buffer of BYTES bytes."))

(defgeneric device-free (device buffer)
  (:documentation "Release BUFFER. May run on a finalizer thread."))

(defgeneric device-store (device buffer sequence element-type)
  (:documentation "Store the numbers in SEQUENCE from the start of BUFFER as ELEMENT-TYPE, a scalar or vector type.
The components of vector elements are consecutive in SEQUENCE. A 3-vector takes the space of a 4-vector.
Launches that are still running finish first."))

(defgeneric device-load (device buffer sequence element-type)
  (:documentation "Fill SEQUENCE with the components of elements of ELEMENT-TYPE from the start of BUFFER.
ELEMENT-TYPE and the layout are those of DEVICE-STORE. Launches that are still running finish first."))

(defgeneric device-prepare (device source entry-name)
  (:documentation "Compile SOURCE and return the runnable kernel ENTRY-NAME."))

(defgeneric device-launch (device prepared arguments grid group)
  (:documentation "Enqueue PREPARED over GRID, a list of three thread counts, in groups of GROUP or NIL.
ARGUMENTS are (:buffer buffer), (:bytes type value) or (:shared bytes) in parameter order.
Launches on one device run in order, but may not have started when this returns: DEVICE-STORE,
DEVICE-LOAD and DEVICE-SYNCHRONIZE wait for them."))

(defgeneric device-synchronize (device)
  (:documentation "Block until every launch on DEVICE has finished. Signal the first error a launch
raised since the last call.")
  (:method (device)
    (declare (ignore device))
    nil))

(defgeneric device-gemm (device m n k alpha a b beta c count)
  (:documentation "Set COUNT M by N matrices C[i] to ALPHA*A[i]*B[i] + BETA*C[i], where A[i] is M by K and B[i] is K by N.
A, B and C are (array offset row-stride column-stride batch-stride) lists of f32 arrays; element
(i j) of product p is at offset + p*batch-stride + i*row-stride + j*column-stride. M, N, K and COUNT
are positive, ALPHA is not zero and C addresses each element once. Provided by cl-gpu/blas, which
backends may specialize."))

(defgeneric device-gemv (device m n alpha a x beta y)
  (:documentation "Set the M elements of Y to ALPHA*A*X + BETA*Y, where A is M by N and X has N elements.
A is as in DEVICE-GEMM; X and Y are (array start increment) lists. M and N are positive and ALPHA is
not zero. Provided by cl-gpu/blas, which backends may specialize."))

;;; Devices

(defvar *device* nil
  "The device that arrays and launches use unless given :device.")

(defvar *backends* '()
  "Registered backends as (name . function), where the function returns the backend's default device.")

(defun register-backend (name default-device-function)
  "Register the backend NAME. DEFAULT-DEVICE-FUNCTION returns its default device."
  (setf *backends* (append (remove name *backends* :key #'car)
                           (list (cons name default-device-function))))
  name)

(defun current-device ()
  "*DEVICE*, or the default device of the first registered backend."
  (or *device*
      (if *backends*
          (funcall (cdr (first *backends*)))
          (gpu-fail "no GPU backend is loaded; load cl-gpu/metal"))))

(defun synchronize (&optional (device (current-device)))
  "Block until every launch on DEVICE, by default the current device, has finished.
Signal the first error a launch raised since the last synchronization."
  (device-synchronize device)
  nil)
