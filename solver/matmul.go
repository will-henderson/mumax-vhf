package solver

/*
#cgo CFLAGS: -I /home/will/amd/aocl/4.0/include_LP64
#cgo LDFLAGS: -lgfortran
#cgo LDFLAGS: -L /usr/local/lib -l:libarpack.a
#cgo LDFLAGS: -L /home/will/amd/aocl/4.0/lib_LP64 -l:libflame.a -lblas -lm
#include <cblas.h>
*/
import "C"

func matvecmul(m, n int, mat, vec, res []float64) {
	C.cblas_dgemv(C.CblasRowMajor, C.CblasNoTrans, C.int(m), C.int(n), C.double(1.), (*C.double)(&mat[0]),
		C.int(m), (*C.double)(&vec[0]), C.int(1), C.double(0.), (*C.double)(&res[0]), C.int(1))
}
