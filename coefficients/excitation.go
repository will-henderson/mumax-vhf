package coefficients

import (
	"reflect"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

type mulmask struct {
	mul  func() float64
	mask *data.Slice
}

var (
	mulmasks []mulmask
)

//this is stored in the en.B_ext variable.

func getExcitation() []mulmask {
	if mulmasks == nil {

		//could check that M has actually been initialised first

		excitation := reflect.ValueOf(&en.B_ext).Elem()
		mulmasks := excitation.Field(2)

		magnetisationBuffer = (**data.Slice)(buffer.Addr().UnsafePointer())

	}
	return magnetisationBuffer
}

func E1(a int) {

}
