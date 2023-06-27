package field

import (
	"reflect"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

var (
	magnetisationBuffer **data.Slice //address of en.M.buffer_ in memory
)

// SetFieldComplex sets the the slices beff_real and beff_imag to, repectively, the real and imaginary components of the field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func SetFieldComplex(s, b CSlice) {

	SetDemagComplex(s, b)
	AddExchangeComplex(s, b)
	AddAnisotropyComplex(s, b)

	en.B_ext.AddTo(b.Real)

}

// SelfInteractionOperate sets res to the value Σ_r' H_rr' s_r',
// that is, the operation of the self-interaction tensor on a complex magnetisation s
func SelfInteractionOperate(s, res CSlice) {

	SIField := NewCSlice(3, en.Mesh().Size())
	defer SIField.Free()

	SetSIFieldComplex(s, SIField)

	msat, rM := en.Msat.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}

	//Msat is real so can just multiply real and imag parts individually
	for c := 0; c < 3; c++ {
		cuda.Mul(res.Real, SIField.Real, msat)
		cuda.Mul(res.Imag, SIField.Imag, msat)
	}

}

func SetSIFieldComplex(s, b CSlice) {
	SetDemagComplex(s, b)
	AddExchangeComplex(s, b)
	AddAnisotropyComplex(s, b)
}

// getMagnetisationBuffer returns the address of en.M.buffer_, a pointer to a slice on the GPU storing the magnetisation.
// Cached on initial call.
func getMagnetisationBuffer() **data.Slice {
	if magnetisationBuffer == nil {

		//could check that M has actually been initialised first

		M := reflect.ValueOf(&en.M).Elem()
		buffer := M.Field(0)
		magnetisationBuffer = (**data.Slice)(buffer.Addr().UnsafePointer())

	}
	return magnetisationBuffer
}
