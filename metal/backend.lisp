(in-package #:cl-gpu/metal)

(defmethod gpu:device-target ((device metal-device))
  :metal)

(defmethod gpu:device-shared-memory-limit ((device metal-device))
  (device-max-threadgroup-memory device))

(defmethod gpu:device-allocate ((device metal-device) bytes)
  (make-buffer device bytes))

(defmethod gpu:device-free ((device metal-device) buffer)
  (release buffer))

(defmethod gpu:device-store ((device metal-device) buffer sequence element-type)
  (copy-to-buffer buffer sequence :element-type element-type))

(defmethod gpu:device-load ((device metal-device) buffer sequence element-type)
  (copy-from-buffer buffer sequence :element-type element-type))

(defmethod gpu:device-prepare ((device metal-device) source entry-name)
  (let ((library (compile-library device source)))
    (unwind-protect (make-pipeline library entry-name)
      (release library))))

(defun metal-argument (argument)
  (ecase (first argument)
    (:buffer (second argument))
    (:bytes argument)
    (:shared (list :threadgroup (* 16 (ceiling (second argument) 16))))))

(defmethod gpu:device-launch ((device metal-device) pipeline arguments grid group)
  (dispatch pipeline (mapcar #'metal-argument arguments) grid :group group))

(defmethod gpu:device-synchronize ((device metal-device))
  (synchronize-device device))

(gpu:register-backend :metal #'default-device)
