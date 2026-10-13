(in-package #:cl-gpu)

;;; The protocol a backend implements for its device class.

(defgeneric device-target (device)
  (:documentation "The printer target keyword whose source DEVICE compiles."))

(defgeneric device-allocate (device bytes)
  (:documentation "A zeroed backend buffer of BYTES bytes."))

(defgeneric device-free (device buffer)
  (:documentation "Release BUFFER. May run on a finalizer thread."))

(defgeneric device-store (device buffer sequence element-type)
  (:documentation "Store the numbers in SEQUENCE from the start of BUFFER as the scalar ELEMENT-TYPE."))

(defgeneric device-load (device buffer sequence element-type)
  (:documentation "Fill SEQUENCE with elements of the scalar ELEMENT-TYPE from the start of BUFFER."))

(defgeneric device-prepare (device source entry-name)
  (:documentation "Compile SOURCE and return the runnable kernel ENTRY-NAME."))

(defgeneric device-launch (device prepared arguments grid group wait)
  (:documentation "Run PREPARED over GRID, a list of three thread counts, in groups of GROUP or NIL.
ARGUMENTS are (:buffer buffer), (:bytes type value) or (:shared bytes) in parameter order.
Return the backend's command, which has finished when WAIT."))

(defgeneric wait (command)
  (:documentation "Block until the command that DEVICE-LAUNCH returned finishes."))

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
