package quickdisp

import (
	"math"
	"math/cmplx"

	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"
)

/*
#cgo CFLAGS: -I /home/will/amd/aocl/4.0/include_LP64
#cgo LDFLAGS: -L /home/will/amd/aocl/4.0/lib_LP64 -l:libflame.a -lblis -lm -fopenmp
#include <lapacke.h>
*/
import "C"

func UniformModesMatrix(samplePoints [3][]int32) (w [][3]complex128, V [][3][3]complex128) {

	ept := mag.EigenProblemTensor()

	w = make([][3]complex128, len(samplePoints[0]))
	V = make([][3][3]complex128, len(samplePoints[0]))
	for i := 0; i < len(samplePoints); i++ {
		w[i], V[i] = uniformModeMatrix([3]int32{samplePoints[0][i], samplePoints[1][i], samplePoints[2][i]}, ept)
	}

	return w, V

}

func FourierMode(idx, size [3]int32, cellsize [3]float64) []complex128 {

	var singles [3][]complex128
	for c := 0; c < 3; c++ {
		singles[c] = make([]complex128, size[c])
		factor := 2. * math.Pi * float64(idx[c]) / (float64(size[c]) * cellsize[c])
		for i := int32(0); i < size[c]; i++ {
			singles[c][i] = cmplx.Rect(1, factor*float64(i))
		}
	}

	//then make this into a flattened array
	mode := make([]complex128, size[2]*size[1]*size[0])
	for k := int32(0); k < size[2]; k++ {
		for j := int32(0); j < size[1]; j++ {
			for i := int32(0); i < size[0]; i++ {
				mode[(size[1]*k+j)*size[0]+i] = singles[0][i] * singles[1][j] * singles[2][k]
			}
		}
	}

	return mode

}

func uniformModeMatrix(idx [3]int32, ept Tensor) (w [3]complex128, V [3][3]complex128) {

	size := [3]int32{int32(ept.Size[0]), int32(ept.Size[1]), int32(ept.Size[2])}

	mode := FourierMode(idx, size, en.Mesh().CellSize())
	length := ept.Length()
	mat := ept.To4D()

	var Hk [3][3]complex128
	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			for i := 0; i < length; i++ {
				for i_ := 0; i_ < length; i_++ {

					Hk[c][c_] += complex(0., mat[c][c_][i][i_]) * cmplx.Conj(mode[i]) * mode[i_]

				}
			}
		}
	}

	li3 := C.lapack_int(3)
	HkFlat := make([]complex128, 9)
	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			HkFlat[3*c+c_] = Hk[c][c_]
		}
	}

	VFlat := make([]complex128, 9)

	C.LAPACKE_zgeev(C.LAPACK_ROW_MAJOR, C.char([]rune("N")[0]), C.char([]rune("V")[0]),
		C.lapack_int(3), (*C.complexdouble)(&HkFlat[0]), C.lapack_int(3),
		(*C.complexdouble)(&w[0]), nil, li3, (*C.complexdouble)(&VFlat[0]), li3)

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			V[c][c_] = VFlat[3*c_+c]
		}
	}

	return

}
