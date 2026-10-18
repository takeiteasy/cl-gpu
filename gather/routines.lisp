(in-package #:cl-gpu/gather)

(defparameter *elements* '(gpu:f16 gpu:bf16 gpu:f32 gpu:i8 gpu:u8 gpu:i16 gpu:u16
                           gpu:i32 gpu:u32 gpu:i64 gpu:u64))
(defparameter *index-elements* '(gpu:i8 gpu:u8 gpu:i16 gpu:u16 gpu:i32 gpu:u32 gpu:i64 gpu:u64))

(defun checked-element (view elements)
  (gpu::check-view view)
  (let ((type (gpu:gpu-array-element-type (gpu:strided-view-data view))))
    (unless (and (typep type 'gpu:scalar-type) (member (gpu:scalar-type-name type) elements))
      (gpu::gpu-fail "gather requires supported scalar storage types"))
    (gpu:scalar-type-name type)))

(defun prepare-gather (input indices out axis)
  (let* ((element (checked-element input *elements*))
         (index-element (checked-element indices *index-elements*))
         (source-shape (gpu:strided-view-shape input))
         (rank (length source-shape))
         (device (gpu:gpu-array-device (gpu:strided-view-data input))))
    (unless (and (integerp axis) (<= (- rank) axis) (< axis rank))
      (gpu::gpu-fail "gather axis ~s is outside rank ~d" axis rank))
    (setf axis (mod axis rank))
    (let ((shape (concatenate 'vector (subseq source-shape 0 axis)
                              (gpu:strided-view-shape indices) (subseq source-shape (1+ axis)))))
      (unless (and (typep (length shape) '(unsigned-byte 32))
                   (typep (gpu::shape-size shape) '(signed-byte 64))
                   (every (lambda (s) (typep s '(signed-byte 64))) (gpu::row-strides shape)))
        (gpu::gpu-fail "gather result exceeds signed 64-bit indexing"))
      (unless (eq device (gpu:gpu-array-device (gpu:strided-view-data indices)))
        (gpu::gpu-fail "gather operands require one device"))
      (when out
        (unless (and (eq element (checked-element out *elements*))
                     (equalp shape (gpu:strided-view-shape out))
                     (eq device (gpu:gpu-array-device (gpu:strided-view-data out)))
                     (gpu::unique-view-p out))
          (gpu::gpu-fail "gather destination requires matching shape, type, device and unique addresses")))
      (values shape axis element index-element device))))

(defun allocate-result (shape element device)
  (let ((data (gpu:make-gpu-array (gpu::shape-size shape) :element-type element :device device)))
    (handler-case (gpu:make-strided-view data shape)
      (error (e) (gpu:free data) (error e)))))

(defun layout-metadata (shape &rest strides)
  (let* ((width (1+ (length strides)))
         (data (make-array (* width (length shape)) :element-type '(signed-byte 64))))
    (loop for axis downfrom (1- (length shape)) to 0 for row from 0
          do (setf (aref data (* row width)) (aref shape axis))
          (loop for layout in strides for column from 1
                do (setf (aref data (+ (* row width) column)) (aref layout axis))))
    data))

(defun gather-strides (input indices axis shape)
  (let* ((index-rank (length (gpu:strided-view-shape indices)))
         (source (make-array (length shape) :initial-element 0))
         (index (make-array (length shape) :initial-element 0))
         (strides (gpu:strided-view-strides input)))
    (dotimes (i axis) (setf (aref source i) (aref strides i)))
    (loop for i from (1+ axis) below (length strides)
          do (setf (aref source (+ i index-rank -1)) (aref strides i)))
    (loop for stride across (gpu::row-strides (gpu:strided-view-shape indices))
          for i from axis do (setf (aref index i) stride))
    (values source index)))

(defun launch-chunks (kernel count arguments device)
  (loop for start from 0 below count by 16776960
        do (gpu::launch kernel (funcall arguments start)
                        :grid (min 16776960 (- count start)) :group 256 :device device)))

(defun execute-gather (input indices out axis)
  (multiple-value-bind (shape axis element index-element device)
      (prepare-gather input indices out axis)
    (let ((result nil) (normalized nil) (status nil) (metadata nil)
          (gather-metadata nil) (copy-metadata nil) (success nil)
          (count (gpu::shape-size shape))
          (index-count (gpu::shape-size (gpu:strided-view-shape indices))))
      (unwind-protect
           (progn
             (setf result (allocate-result shape element device))
             (when (plusp index-count)
               (setf normalized (gpu:make-gpu-array index-count :element-type 'gpu:i64 :device device)
                     status (gpu:make-gpu-array 1 :element-type 'gpu:u32 :device device)
                     metadata (gpu:to-device
                               (layout-metadata (gpu:strided-view-shape indices)
                                                (gpu:strided-view-strides indices))
                               :element-type 'gpu:i64 :device device))
               (when (plusp count)
                 (multiple-value-bind (source-strides index-strides) (gather-strides input indices axis shape)
                   (setf gather-metadata
                         (gpu:to-device (layout-metadata shape source-strides (gpu::row-strides shape) index-strides)
                                        :element-type 'gpu:i64 :device device))))
               (launch-chunks
                (cached-kernel :indices index-element) index-count
                (lambda (start)
                  (list (gpu::view-root indices) normalized metadata status
                        (length (gpu:strided-view-shape indices)) (gpu::view-base indices)
                        (aref (gpu:strided-view-shape input) axis) index-count start)) device)
               (when (plusp count)
                 (launch-chunks
                  (cached-kernel :gather element) count
                  (lambda (start)
                    (list (gpu::view-root input) (gpu:strided-view-data result) gather-metadata
                          (length shape) (gpu::view-base input) 0 count start normalized
                          (aref (gpu:strided-view-strides input) axis))) device))
               (unless (zerop (aref (gpu:from-device status) 0))
                 (gpu::gpu-fail "gather index outside axis ~d of size ~d"
                                axis (aref (gpu:strided-view-shape input) axis))))
             (when (and out (plusp count))
               (setf copy-metadata
                     (gpu:to-device (layout-metadata shape (gpu::row-strides shape)
                                                     (gpu:strided-view-strides out))
                                    :element-type 'gpu:i64 :device device))
               (launch-chunks
                (cached-kernel :copy element) count
                (lambda (start)
                  (list (gpu:strided-view-data result) (gpu::view-root out) copy-metadata
                        (length shape) 0 (gpu::view-base out) count start)) device))
             (setf success t)
             (or out result))
        (dolist (array (list normalized status metadata gather-metadata copy-metadata))
          (when array (gpu:free array)))
        (when (and result (or out (not success)))
          (gpu:free (gpu:strided-view-data result)))))))

(defun gather (input indices &key (axis 0))
  "Gather slices along AXIS into independent storage using device INDICES."
  (execute-gather input indices nil axis))

(defun gather! (out input indices &key (axis 0))
  "Gather slices along AXIS, check bounds, then queue a copy into OUT and return it."
  (gpu::check-view out)
  (execute-gather input indices out axis))
