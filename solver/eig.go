package solver

import (
	"fmt"
)

//#cgo CFLAGS: -I /home/will/amd/aocl/4.0/include_LP64
//#cgo LDFLAGS: -L /home/will/amd/aocl/4.0/lib_LP64 -l:libflame.a -lblis -lm
//#cgo LDFLAGS: -L${SRCDIR}
//#include <lapacke.h>
import "C"

// Eig returns the eigenvalues and eigenvectors of a matrix mat, which passed in a flattened to row major form.
// Note that mat is overwritten during the routine.
func Eig(n int, mat []float64) ([]complex128, [][]complex128) {
	wr := make([]float64, n)
	wi := make([]float64, n)
	V := make([]float64, n*n)

	nC := C.lapack_int(n)
	//charN := C.CString("N")
	//charV := C.CString("V")
	//defer C.free(unsafe.Pointer(charN))
	//defer C.free(unsafe.Pointer(charV))

	info := C.LAPACKE_dgeev(C.LAPACK_ROW_MAJOR, C.char([]rune("N")[0]), C.char([]rune("V")[0]), nC,
		(*C.double)(&mat[0]), nC, (*C.double)(&wr[0]), (*C.double)(&wi[0]), nil, nC, (*C.double)(&V[0]), nC)

	if info != 0 {
		panic(fmt.Sprintf("dgeev failed: info = %d", info))
	}

	return GeevToCmplx(n, wr, wi, V)

}

func GeevToCmplx(n int, wr, wi, V []float64) ([]complex128, [][]complex128) {
	values := make([]complex128, n)
	vectors := make([][]complex128, n)
	for i := 0; i < n; i++ {
		values[i] = complex(wr[i], wi[i])
		vectors[i] = make([]complex128, n)
	}

	i := 0
	for i < n {
		if wi[i] == 0 {
			for j := 0; j < n; j++ {
				vectors[i][j] = complex(V[n*j+i], 0)
			}
			i++
		} else {
			for j := 0; j < n; j++ {
				vectors[i][j] = complex(V[n*j+i], V[n*j+i+1])
				vectors[i+1][j] = complex(V[n*j+i], -V[n*j+i+1])
			}
			i += 2
		}
	}

	return values, vectors
}
