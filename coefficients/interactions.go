// Package coefficients returns interaction coefficients for a system, given the eigenmodes and eigenvectors.
package coefficients

import (
	"github.com/will-henderson/mumax-vhf/field"
)

//i don't want to pass modes to this.
//i want to pass a set solver then have some global solve method which deals with this.

var (
	Modes []field.CSlice
	//we may have some problem about whether all of these can fit in the GPU memory... Hope we have big GPU.

	//but it is same amount of memory to save the eigenFields as well then.
	eigenFields []field.CSlice
)

func EigenFields(i int) field.CSlice {
	if eigenFields == nil {

		eigenFields = make([]field.CSlice, len(Modes))

		for i := 0; i < len(Modes); i++ {
			field.SetSIFieldComplex(Modes[i], eigenFields[i])
		}

	}
	return eigenFields[i]
}

func V3(a, b, c int) {

}

func v3_(a, b, c int) {
	- Ms *

}
