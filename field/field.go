package field

import (
	"reflect"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

var (
	magnetisationBuffer **data.Slice
)

//ok so we want to find the effective field for complex vectors.

func SetEffectiveField(s_real, s_imag, beff_real, beff_imag *data.Slice) {

	// we assume that all the inputs live on the GPU.
	// we then just want m.buffer_ to point to something else.

	SetDemagField(s_real, beff_real)
	SetDemagField(s_real, beff_imag)

}

func getMagnetisationBuffer() **data.Slice {
	if magnetisationBuffer == nil {

		//could check that M has actually been initialised first

		M := reflect.ValueOf(&en.M).Elem()
		buffer := M.Field(0) //but i really want the address of this field
		magnetisationBuffer = (**data.Slice)(buffer.Addr().UnsafePointer())

	}
	return magnetisationBuffer
}
