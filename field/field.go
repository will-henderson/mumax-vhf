package field

import (
	"reflect"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

var (
	magnetisationBuffer **data.Slice //address of en.M.buffer_ in memory
)

// SetFieldComplex sets the the slices beff_real and beff_imag to, repectively, the real and imaginary components of the field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func SetFieldComplex(b, s CSlice) {

	SetDemagComplex(b, s)
	AddExchangeComplex(b, s)
	AddAnisotropyComplex(b, s)

	en.B_ext.AddTo(b.Real())

}

// SetSIFieldComplex sets res to the value -(1/Ms) * Σ_r' H_rr' s_r' ,
// that is, the operation of the self-interaction tensor on a complex magnetisation s.
// It is the self-interaction field (i.e. without the external field Bext) created by the complex magnetisation s.
func SetSIFieldComplex(b, s CSlice) {
	SetDemagComplex(b, s)
	AddExchangeComplex(b, s)
	AddAnisotropyComplex(b, s)
}

func SetSIField(b *data.Slice, s *data.Slice) {
	SetDemagField(b, s)
	AddExchangeField(b, s)
	AddAnisotropyField(b, s)
}

// getMagnetisationBuffer returns the address of en.M.buffer_, a pointer to a slice on the GPU storing the magnetisation.
// Cached on initial call.
func GetMagnetisationBuffer() **data.Slice {
	if magnetisationBuffer == nil {

		//could check that M has actually been initialised first

		M := reflect.ValueOf(&en.M).Elem()
		buffer := M.Field(0)
		magnetisationBuffer = (**data.Slice)(buffer.Addr().UnsafePointer())

	}
	return magnetisationBuffer
}

// returns the ground state magnetic field, but as a scalar for each position
// (it's direction is equal to the direction of the ground state magnetisation)
func GroundStateField() *data.Slice {

	dst := cuda.NewSlice(3, en.Mesh().Size())
	defer dst.Free()

	en.SetDemagField(dst)
	en.AddExchangeField(dst)
	en.AddAnisotropyField(dst)
	en.B_ext.AddTo(dst)

	//dst now holds the ground state field.
	result := cuda.NewSlice(1, en.Mesh().Size())
	cuda.Zero(result)
	cuda.AddDotProduct(result, 1, en.M.Buffer(), dst)

	return result
}
