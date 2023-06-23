package field

import (
	"reflect"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

var (
	magnetisationBuffer **data.Slice //address of en.M.buffer_ in memory
)

type CSlice struct {
	Real, Imag *data.Slice
}

// SetFieldComplex sets the the slices beff_real and beff_imag to, repectively, the real and imaginary components of the field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func SetFieldComplex(s, b CSlice) {

	SetDemagComplex(s, b)
	AddExchangeComplex(s, b)
	AddAnisotropyComplex(s, b)

	en.B_ext.AddTo(b.Real)

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
