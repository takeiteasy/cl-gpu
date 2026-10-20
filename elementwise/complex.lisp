(in-package #:cl-gpu/elementwise)

(defun complex-conversion-code (source destination value)
  (let* ((sc (component-element source)) (dc (component-element destination))
         (real (if (complex-element-p source) `(aref ,value 0) value)))
    (if (complex-element-p destination)
        `(gpu:vec ,dc ,(converted-code sc dc real 'rounding)
                  ,(if (complex-element-p source)
                       (converted-code sc dc `(aref ,value 1) 'rounding)
                       `(gpu:convert ,dc 0)))
        (converted-code sc dc real 'rounding))))

(defun conversion-error-code (source destination value)
  (let ((real (if (complex-element-p source) `(aref ,value 0) value)))
    `(or ,@(when (complex-element-p source) `((/= (aref ,value 1) 0.0)))
         ,@(when (not (or (float-element-p destination) (complex-element-p destination)))
             `((/= ,real ,real))))))

(defun complex-operation-code (operation element values operator)
  (destructuring-bind (a &optional b c) values
    (let* ((component (component-element element))
           (r `(aref ,a 0)) (i `(aref ,a 1))
           (s `(aref ,b 0)) (j `(aref ,b 1)))
      (case operation
        (:copy a)
        (:convert (complex-conversion-code element operator a))
        (:select `(if (/= ,a 0) ,b ,c))
        (:realpart r)
        (:imagpart i)
        (:conjugate
         `(gpu:vec ,component ,r
                   (gpu:bitcast ,component
                     (logxor (gpu:bitcast ,(unsigned-element component) ,i)
                             (gpu:convert ,(unsigned-element component)
                                          ,(ash 1 (1- (gpu:scalar-type-bits (gpu:parse-type component)))))))))
        (:negate `(- ,a))
        ((:add :subtract) `(,(if (eq operation :add) '+ '-) ,a ,b))
        (:multiply `(gpu:vec ,component (- (* ,r ,s) (* ,i ,j)) (+ (* ,r ,j) (* ,i ,s))))
        (:divide `(,(if (eq element 'gpu:c32) '%divide-c32 '%divide-c64) ,a ,b))
        (:abs `(,(if (eq element 'gpu:c32) '%abs-c32 '%abs-c64) ,a))
        (:compare
         `(if ,(let ((equal `(and (= ,r ,s) (= ,i ,j))))
                 (if (eq operator :eq) equal `(not ,equal)))
              (gpu:convert gpu:u8 1) (gpu:convert gpu:u8 0)))
        (otherwise (gpu-fail "unsupported complex operation ~s" operation))))))

(defparameter *complex-math-template*
"inline REAL cx_abs_SUFFIX(PAIR a) {
    REAL x = abs(a.x), y = abs(a.y);
    if (isinf(x) || isinf(y)) return (REAL)INFINITY;
    if (isnan(x) || isnan(y)) return (REAL)NAN;
    REAL m = max(x, y);
    if (m == (REAL)0) return (REAL)0;
    x /= m; y /= m;
    return m * sqrt(x*x + y*y);
}
inline PAIR cx_div_SUFFIX(PAIR a, PAIR b) {
    // TODO: shared numerator scaling loses extreme component ratios; track each term exponent (#93).
    REAL am = max(abs(a.x), abs(a.y)), bm = max(abs(b.x), abs(b.y));
    if (isfinite(am) && isfinite(bm) && bm != (REAL)0) {
        int ae = am == (REAL)0 ? 0 : ilogb(am), be = ilogb(bm);
        REAL ar = ldexp(a.x, -ae), ai = ldexp(a.y, -ae);
        REAL br = ldexp(b.x, -be), bi = ldexp(b.y, -be);
        REAL d = br*br + bi*bi;
        return PAIR(ldexp((ar*br + ai*bi)/d, ae-be),
                    ldexp((ai*br - ar*bi)/d, ae-be));
    }
    REAL d = b.x*b.x + b.y*b.y;
    return PAIR((a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d);
}")

(defun complex-helper (real pair suffix target)
  (labels ((replace-all (text old new)
             (with-output-to-string (out)
               (loop with start = 0 for at = (search old text :start2 start)
                     do (write-string text out :start start :end at)
                     while at do (write-string new out) (setf start (+ at (length old)))))))
    (let ((text *complex-math-template*))
      (when (eq target :opencl)
        (setf text (replace-all text "abs(" "fabs("))
        (setf text (replace-all text "return PAIR(" "return (PAIR)(")))
      (setf text (replace-all text "REAL" real)
            text (replace-all text "PAIR" pair))
      (replace-all text "SUFFIX" suffix))))

(gpu::define-intrinsic %abs-c32 (:typed ((gpu:vec gpu:f32 2)) gpu:f32)
  :metal (:helper "cx_abs_c32") :opencl (:helper "cx_abs_c32"))
(gpu::define-intrinsic %abs-c64 (:typed ((gpu:vec gpu:f64 2)) gpu:f64)
  :opencl (:helper "cx_abs_c64"))
(gpu::define-intrinsic %divide-c32 (:typed ((gpu:vec gpu:f32 2) (gpu:vec gpu:f32 2)) (gpu:vec gpu:f32 2))
  :metal (:helper "cx_div_c32") :opencl (:helper "cx_div_c32"))
(gpu::define-intrinsic %divide-c64 (:typed ((gpu:vec gpu:f64 2) (gpu:vec gpu:f64 2)) (gpu:vec gpu:f64 2))
  :opencl (:helper "cx_div_c64"))

(dolist (target '(:metal :opencl))
  (dolist (element (if (eq target :metal) '(gpu:c32) '(gpu:c32 gpu:c64)))
    (let* ((suffix (if (eq element 'gpu:c32) "c32" "c64"))
           (real (if (eq element 'gpu:c32) "float" "double"))
           (text (complex-helper real (concatenate 'string real "2") suffix target)))
      (dolist (name (list (concatenate 'string "cx_abs_" suffix) (concatenate 'string "cx_div_" suffix)))
        (let* ((split (search (concatenate 'string "inline " real "2") text))
               (source (if (search "cx_abs_" name) (subseq text 0 split) (subseq text split))))
          (if (eq target :metal) (gpu::register-msl-helper name source)
              (gpu::register-opencl-helper name source)))))))
