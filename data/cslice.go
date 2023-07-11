package data

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"

	"math"
	"math/rand"
)

type CSlice struct {
	real, imag *data.Slice
}

func NewCSlice(nComp int, size [3]int) CSlice {

	return CSlice{
		real: cuda.NewSlice(nComp, size),
		imag: cuda.NewSlice(nComp, size),
	}

}

func NewCSliceCPU(nComp int, size [3]int) CSlice {
	return CSlice{
		real: data.NewSlice(nComp, size),
		imag: data.NewSlice(nComp, size),
	}
}

func (cs CSlice) NComp() int {
	return cs.real.NComp()
}

func (cs CSlice) Size() [3]int {
	return cs.real.Size()
}

func (cs CSlice) Len() int {
	return cs.real.Len()
}

// Real returns the real part of the CSlice
func (cs CSlice) Real() *data.Slice {
	return cs.real
}

// Imag returns the imaginary part of the CSlice
func (cs CSlice) Imag() *data.Slice {
	return cs.imag
}

func (cs *CSlice) SwitchParts() {
	temp := cs.real
	cs.real = cs.imag
	cs.imag = temp
}

// Zero sets all elements of all components of both the real and imaginary parts of a CSlice to zero.
func Zero(cs CSlice) {
	cuda.Zero(cs.real)
	cuda.Zero(cs.imag)
}

func Copy(dst, src CSlice) {
	data.Copy(dst.real, src.real)
	data.Copy(dst.imag, src.imag)
}

func (cs CSlice) Free() {

	cs.real.Free()
	cs.imag.Free()

}

func (cs CSlice) CPUAccess() bool {
	r := cs.real.CPUAccess()
	i := cs.imag.CPUAccess()

	if r && i {
		return true
	} else if !r && !i {
		return false
	} else {
		panic("the real and imaginary parts do not live in the same memory")
	}
}

func (cs CSlice) HostCopy() CSlice {
	return CSlice{
		real: cs.real.HostCopy(),
		imag: cs.imag.HostCopy(),
	}
}

func (cs CSlice) DevCopy() CSlice {
	ret := NewCSlice(cs.NComp(), cs.Size())
	Copy(ret, cs)
	return ret
}

// RandomPerpendicular creates a CSlice with random directions perpendicular to the magnetisation,
// normalised over all space and all components to one.
func RandomPerpendicular(m *data.Slice) CSlice {
	//create a random vector on the CPU
	seed := 0
	rng := rand.New(rand.NewSource(int64(seed)))
	size := m.Size()

	rndRealCPU := data.NewSlice(3, size)
	rndRealVectors := rndRealCPU.Vectors()
	rndImagCPU := data.NewSlice(3, size)
	rndImagVectors := rndImagCPU.Vectors()
	for c := 0; c < 3; c++ {
		for k := 0; k < size[2]; k++ {
			for j := 0; j < size[1]; j++ {
				for i := 0; i < size[0]; i++ {
					rndRealVectors[c][k][j][i] = rng.Float32()
					rndImagVectors[c][k][j][i] = rng.Float32()
				}
			}
		}
	}

	rndRealGPU := cuda.NewSlice(3, size)
	data.Copy(rndRealGPU, rndRealCPU)
	rndImagGPU := cuda.NewSlice(3, size)
	data.Copy(rndImagGPU, rndImagCPU)

	paraReal := cuda.NewSlice(1, size)
	defer paraReal.Free()
	cuda.Zero(paraReal)
	cuda.AddDotProduct(paraReal, -1, m, rndRealGPU)
	cuda.AddMul1D(rndRealGPU, paraReal, m)

	paraImag := cuda.NewSlice(1, size)
	defer paraImag.Free()
	cuda.Zero(paraImag)
	cuda.AddDotProduct(paraImag, -1, m, rndImagGPU)
	cuda.AddMul1D(rndImagGPU, paraImag, m)

	//now we need to normalise
	rndCSlice := CSlice{rndRealGPU, rndImagGPU}
	magnitude := math.Sqrt(float64(real(Dotc(rndCSlice, rndCSlice))))
	SScal(rndCSlice, rndCSlice, float32(1/magnitude))

	return rndCSlice
}

// Dot c computes the dot product of two CSlices (over all space, all components) with complex conjugation
func Dotc(a, b CSlice) complex64 {

	resReal := cuda.Dot(a.real, b.real) + cuda.Dot(a.imag, b.imag)
	resImag := cuda.Dot(a.real, b.imag) - cuda.Dot(a.imag, b.real)

	return complex(resReal, resImag)

}

func SScal(dst, src CSlice, factor float32) {
	cuda.Scale(dst.real, src.real, factor)
	cuda.Scale(dst.imag, src.imag, factor)
}

func SMadd2(dst, src1, src2 CSlice, factor1, factor2 float32) {
	cuda.Madd2(dst.real, src1.real, src2.real, factor1, factor2)
	cuda.Madd2(dst.imag, src1.imag, src2.imag, factor1, factor2)
}

func CMadd2(dst, src1, src2 CSlice, factor1, factor2 complex64) {
	cuda.Madd4(dst.real, src1.real, src1.imag, src2.real, src2.imag, real(factor1), -imag(factor1), real(factor2), -imag(factor2))
	cuda.Madd4(dst.imag, src1.real, src1.imag, src2.real, src2.imag, imag(factor1), real(factor1), imag(factor2), real(factor2))
}
