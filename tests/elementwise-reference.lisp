(in-package #:cl-gpu/elementwise/tests)

(defun values-of (view)
  (let ((data (gpu:from-device (gpu:strided-view-data view))) (values nil))
    (gpu::map-view-offsets (lambda (at) (push (aref data at) values)) view)
    (coerce (nreverse values) 'vector)))

(defun close-p (expected actual &optional (tolerance 1d-4))
  (and (= (length expected) (length actual))
       (every (lambda (a b) (<= (abs (- a b)) (* tolerance (max 1d0 (abs a) (abs b))))) expected actual)))

(defun raw-type (element)
  (ecase (gpu:scalar-type-bits (gpu:parse-type element))
    (16 'gpu:u16) (32 'gpu:u32) (64 'gpu:u64)))

(defun store-raw (data values)
  (gpu:device-store (gpu:gpu-array-device data) (gpu::live-buffer data) values
                    (gpu:parse-type (raw-type (gpu:scalar-type-name (gpu:gpu-array-element-type data)))) 0))

(defun load-raw (data)
  (let ((values (make-array (gpu:gpu-array-length data))))
    (gpu:device-load (gpu:gpu-array-device data) (gpu::live-buffer data) values
                     (gpu:parse-type (raw-type (gpu:scalar-type-name (gpu:gpu-array-element-type data)))) 0)
    values))

(defun float-format (element)
  (ecase element (gpu:f16 (values 10 5 15)) (gpu:bf16 (values 7 8 127)) (gpu:f32 (values 23 8 127)) (gpu:f64 (values 52 11 1023))))

(defun reference-float-conversion (raw source destination mode)
  (when (eq source destination) (return-from reference-float-conversion raw))
  (multiple-value-bind (sf se sb) (float-format source)
    (multiple-value-bind (df de db) (float-format destination)
      (let* ((negative (logbitp (+ sf se) raw))
             (sign (if negative (ash 1 (+ df de)) 0))
             (field (ldb (byte se sf) raw)) (fraction (ldb (byte sf 0) raw))
             (inf (ash (1- (ash 1 de)) df)))
        (when (= field (1- (ash 1 se)))
          (return-from reference-float-conversion
            (logior sign inf (if (zerop fraction) 0
                                (logior (ash 1 (1- df)) (ash fraction (- df sf)))))))
        (let* ((value (* (+ fraction (if (zerop field) 0 (ash 1 sf)))
                         (expt 2 (- (if (zerop field) 1 field) sb sf)))))
          (when (zerop value) (return-from reference-float-conversion sign))
          (let ((e (- (integer-length (numerator value)) (integer-length (denominator value)))))
            (when (< value (expt 2 e)) (decf e))
            (let* ((unit (expt 2 (- (max e (- 1 db)) df)))
                   (scaled (/ (if negative (- value) value) unit))
                   (rounded (abs (funcall (ecase mode
                                            (:nearest-even #'round) (:truncate #'truncate)
                                            (:floor #'floor) (:ceiling #'ceiling)) scaled))))
              (when (< e (- 1 db))
                (return-from reference-float-conversion (logior sign rounded)))
              (when (>= rounded (ash 1 (1+ df))) (setf rounded (ash rounded -1)) (incf e))
              (if (> e db)
                  (logior sign (if (or (eq mode :nearest-even) (and negative (eq mode :floor))
                                      (and (not negative) (eq mode :ceiling))) inf (1- inf)))
                  (logior sign (ash (+ e db) df) (ldb (byte df 0) rounded))))))))))

(defun integer-bounds (element)
  (let* ((type (gpu:parse-type element)) (bits (gpu:scalar-type-bits type))
         (signed (gpu::signed-type-p type)))
    (list (if signed (- (expt 2 (1- bits))) 0) (1- (expt 2 (if signed (1- bits) bits))))))

