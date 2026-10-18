(in-package #:cl-gpu/gather)

(defvar *kernels* (make-hash-table :test #'equal))
(defvar *kernel-lock* (bordeaux-threads:make-lock "gather kernels"))

(gpu::define-intrinsic %bounds-error (:typed ((gpu:ptr gpu:u32 :global)) nil)
  :impure t :metal (:helper "cl_gpu_gather_error"))

(defmethod gpu::helper-definition :around ((printer gpu:msl-printer) name)
  (if (string= name "cl_gpu_gather_error")
      "inline void cl_gpu_gather_error(device uint* status) {
    atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(status), 1u, memory_order_relaxed);
}"
      (call-next-method)))

(defun decode-offsets (offsets)
  ;; TODO: O(rank) decoding per element; coalesce axes and specialize contiguous layouts (#84).
  `(let ((q position))
     (gpu:for (axis (gpu:convert gpu:u32 0) (< axis rank) (incf axis))
       (let* ((row (* (gpu:convert gpu:i64 axis) ,(1+ (length offsets))))
              (dimension (aref metadata row))
              (coordinate (rem q dimension)))
         ,@(loop for offset in offsets for column from 1
                 collect `(incf ,offset (* coordinate (aref metadata (+ row ,column)))))
         (setf q (/ q dimension))))))

(defun normalization-kernel (element)
  (let* ((signed (gpu::signed-type-p (gpu:parse-type element)))
         (wide (if signed 'gpu:i64 'gpu:u64)))
    (gpu::make-gpu-kernel
     'gather-indices
     `((indices (gpu:ptr ,element :global)) (normalized (gpu:ptr gpu:i64 :global))
       (metadata (gpu:ptr gpu:i64 :global)) (status (gpu:ptr gpu:u32 :global))
       (rank gpu:u32) (base gpu:i64) (extent gpu:i64) (count gpu:i64) (start gpu:i64))
     `(let ((position (+ start (gpu:convert gpu:i64 (gpu:global-id 0)))))
        (when (< position count)
          (let ((at base))
            ,(decode-offsets '(at))
            (let ((value (gpu:convert ,wide (aref indices at))))
              (if ,(if signed `(and (>= value (- extent)) (< value extent))
                       `(< value (gpu:convert gpu:u64 extent)))
                  (setf (aref normalized position)
                        ,(if signed `(if (< value 0) (+ extent value) value)
                             `(gpu:convert gpu:i64 value)))
                  (progn (setf (aref normalized position) (gpu:convert gpu:i64 -1))
                         (%bounds-error status)))))))
     :precise-math t)))

(defun copy-kernel (element gatherp)
  (gpu::make-gpu-kernel
   (if gatherp 'gather-values 'gather-copy)
   `((input (gpu:ptr ,element :global)) (out (gpu:ptr ,element :global))
     (metadata (gpu:ptr gpu:i64 :global)) (rank gpu:u32)
     (base gpu:i64) (obase gpu:i64) (count gpu:i64) (start gpu:i64)
     ,@(when gatherp '((normalized (gpu:ptr gpu:i64 :global)) (stride gpu:i64))))
   `(let ((position (+ start (gpu:convert gpu:i64 (gpu:global-id 0)))))
      (when (< position count)
        (let ((at base) (oi obase) ,@(when gatherp '((ii (gpu:convert gpu:i64 0)))))
          ,(decode-offsets (if gatherp '(at oi ii) '(at oi)))
          ,(if gatherp
               `(let ((index (aref normalized ii)))
                  (when (>= index 0)
                    (setf (aref out oi) (aref input (+ at (* index stride))))))
               `(setf (aref out oi) (aref input at))))))
   :precise-math t))

(defun cached-kernel (operation element)
  (bordeaux-threads:with-lock-held (*kernel-lock*)
    (or (gethash (list operation element) *kernels*)
        (setf (gethash (list operation element) *kernels*)
              (ecase operation
                (:indices (normalization-kernel element))
                (:gather (copy-kernel element t))
                (:copy (copy-kernel element nil)))))))
