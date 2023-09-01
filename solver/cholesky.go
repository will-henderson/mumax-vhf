package solver

import (
	"fmt"
)

//#cgo CFLAGS: -I /home/will/amd/aocl/4.0/include_LP64
//#cgo LDFLAGS: -L /home/will/amd/aocl/4.0/lib_LP64 -l:libflame.a -lblis -lm
//#cgo LDFLAGS: -L${SRCDIR}
//#include "cholesky.h"
import "C"

// Cholesky returns the cholesky decomposition of a matrix mat, and returns the lower triangular part by overwriting the input.
func Cholesky(mat []float64) {
	info := C.cholesky(C.int(len(mat)), (*C.double)(&mat[0]))
	if info != 0 {
		panic(fmt.Sprintf("dpotrf failed: info = %d", info))
	}
}

func TriInv(mat []float64) {
	info := C.triinv(C.int(len(mat)), (*C.double)(&mat[0]))
	if info != 0 {
		panic(fmt.Sprintf("dtrtri failed: info = %d", info))
	}
}
