package field

import (
	"reflect"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

var (
	magnetisationBuffer **data.Slice //address of en.M.buffer_ in memory
)

// SetFieldComplex sets the the slices beff_real and beff_imag to, repectively, the real and imaginary components of the field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func SetFieldComplex(s_real, s_imag, beff_real, beff_imag *data.Slice) {

	SetDemagComplex(s_real, s_imag, beff_real, beff_imag)
	AddExchangeComplex(s_real, s_imag, beff_real, beff_imag)
	AddAnisotropyComplex(s_real, s_imag, beff_real, beff_imag)

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
