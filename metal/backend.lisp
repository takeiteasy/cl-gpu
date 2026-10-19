(in-package #:cl-gpu/metal)

(defmethod gpu:device-target ((device metal-device))
  :metal)

(defmethod gpu:device-shared-memory-limit ((device metal-device))
  (device-max-threadgroup-memory device))

(defmethod gpu:device-allocate ((device metal-device) bytes)
  (make-buffer device bytes))

(defmethod gpu:device-free ((device metal-device) buffer)
  (release buffer))

;; TODO: a fixed safe value; the real minimum depends on the GPU family (#72)
(defmethod gpu:device-offset-alignment ((device metal-device) address-space)
  (if (eq address-space :constant) 256 1))

(defmethod gpu:device-store ((device metal-device) buffer sequence element-type offset)
  (copy-to-buffer buffer sequence :element-type element-type :offset offset))

(defmethod gpu:device-load ((device metal-device) buffer sequence element-type offset)
  (copy-from-buffer buffer sequence :element-type element-type :offset offset))

(defmethod gpu:device-prepare ((device metal-device) source entry-name &key precise-math)
  (let ((library (compile-library device source :precise-math precise-math)))
    (unwind-protect (make-pipeline library entry-name)
      (release library))))

(defun metal-argument (argument)
  (ecase (first argument)
    (:buffer argument)
    (:bytes argument)
    (:shared (list :threadgroup (* 16 (ceiling (second argument) 16))))))

(defmethod gpu:device-launch ((device metal-device) pipeline arguments grid group)
  (dispatch pipeline (mapcar #'metal-argument arguments) grid :group group))

(defmethod gpu:device-synchronize ((device metal-device))
  (synchronize-device device))

(gpu:register-backend :metal #'default-device)

(defmethod gpu:device-native-type-p ((device metal-device) element-type)
  (let ((type (gpu:parse-type element-type)))
    (and (typep type 'gpu:scalar-type)
         (member (gpu:scalar-type-name type)
                 '(gpu:f16 gpu:bf16 gpu:f32 gpu:i8 gpu:u8 gpu:i16 gpu:u16 gpu:i32 gpu:u32 gpu:i64 gpu:u64))
         t)))
