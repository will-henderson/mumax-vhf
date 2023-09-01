package solver

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/cuda/curand"

	"math"
)

//#cgo CFLAGS: -I /home/will/amd/aocl/4.0/include_LP64
//#cgo LDFLAGS: -lgfortran
//#cgo LDFLAGS: -L /usr/local/lib -l:libarpack.a
//#cgo LDFLAGS: -L /home/will/amd/aocl/4.0/lib_LP64 -l:libflame.a -lblis -lm
//#include <arpack/arpack.h>
//#include <lapacke.h>
import "C"

type ArnoldiGPU struct {
	genTriMat, ritz, compEB, Q, workspace *cuda.Bytes
	generator                             curand.Generator
	rnorm                                 float32
}

func NewArnoldiGPU(n, nev, ncv int, bmat, which string, tol float64, maxIter int, v0 []float32) ArnoldiGPU {

	return ArnoldiGPU{}
}

func (ar *ArnoldiGPU) Init(ncv int) {
	//workl is an array of floats that lives on gpu of size 3*necv**2 + 6 *ncv
	//but probably more sensible to just split these up
	ar.genTriMat, ar.ritz, ar.compEB, ar.Q, ar.workspace = makeWorkl(ncv)
	eps23 := machineConst()

	ar.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)

	_ = math.Pow(eps23, 2./3.)

}

func makeWorkl(ncv int) (genTriMat, ritz, compEB, Q, workspace *cuda.Bytes) {

	genTriMat = cuda.NewBytes(2 * ncv * cu.SIZEOF_FLOAT32)
	ritz = cuda.NewBytes(ncv * cu.SIZEOF_FLOAT32)
	compEB = cuda.NewBytes(ncv * cu.SIZEOF_FLOAT32)
	Q = cuda.NewBytes(ncv * ncv * cu.SIZEOF_FLOAT32)
	workspace = cuda.NewBytes(3 * ncv * cu.SIZEOF_FLOAT32)

	return genTriMat, ritz, compEB, Q, workspace
}

func machineConst() float64 {

	return float64(C.LAPACKE_slamch('E'))
}

// getV0 generates a random initial residual vector for the Arnoldi process.
// Force the residual vector to be in the range of the operator OP.
func (ar *ArnoldiGPU) getInitV0(n int) *cuda.Bytes {

	v0 := cuda.NewBytes(2 * n * cu.SIZEOF_FLOAT32)
	ar.generator.GenerateUniform(uintptr(v0.Ptr), int64(n))

	//need to scale to between 1 and - 1, and also free v0
	return v0

}

func snaitr()

func (ar *ArnoldiGPU) Iterate() {

}
