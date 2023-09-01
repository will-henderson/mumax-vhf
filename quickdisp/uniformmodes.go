package quickdisp

import (
	"fmt"
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/field"
)

func UniformModes(samplePoints [3][]int32) {

	le := field.NewLinearEvolution()

	for i := 0; i < len(samplePoints[0]); i++ {
		for j := 0; j < len(samplePoints[1]); j++ {
			for k := 0; k < len(samplePoints[2]); k++ {
				uniformMode([3]int32{samplePoints[0][i], samplePoints[1][j], samplePoints[2][k]}, le)
			}
		}
	}

}

func uniformMode(idx [3]int32, le *field.LinearEvolution) {

	mSingle := NewCBuffer(1, en.MeshSize())
	defer mSingle.Recycle()

	cuda.FourierMode(mSingle.Real(), mSingle.Imag(), idx, en.Mesh().CellSize())

	// ok, now we apply the eigenvalue tensor to this mode.

	zeroSlice := cuda.Buffer(1, en.MeshSize())
	B := NewCBuffer(3, en.MeshSize())
	defer cuda.Recycle(zeroSlice)
	defer B.Recycle()

	Hk := make([]complex64, 9)

	for c := 0; c < 3; c++ {
		ptrsReal := make([]unsafe.Pointer, 3)
		ptrsImag := make([]unsafe.Pointer, 3)
		for c_ := 0; c_ < 3; c_++ {
			if c_ == c {
				ptrsReal[c_] = mSingle.Real().DevPtr(0)
				ptrsImag[c_] = mSingle.Real().DevPtr(0)
			} else {
				ptrsReal[c_] = zeroSlice.DevPtr(0)
				ptrsImag[c_] = zeroSlice.DevPtr(0)
			}
		}
		mReal := data.SliceFromPtrs(en.MeshSize(), data.GPUMemory, ptrsReal)
		mImag := data.SliceFromPtrs(en.MeshSize(), data.GPUMemory, ptrsImag)
		m := CSliceFromParts(mReal, mImag)
		le.OperateComplex(&B, m)

		for c_ := 0; c_ < 3; c_++ {
			Hk[3*c_+c] = Dotc(mSingle, B.Comp(c_))
		}

	}

	fmt.Println("///////////////")
	fmt.Println(Hk[0], Hk[1], Hk[2])
	fmt.Println(Hk[3], Hk[4], Hk[5])
	fmt.Println(Hk[6], Hk[7], Hk[8])
	fmt.Println("///////////////")

}
