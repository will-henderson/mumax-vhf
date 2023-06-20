package field

import (
	_ "unsafe"

	"github.com/mumax/3/data"
)

//go:linkname magnetisationBuffer github.com/mumax/3/engine.M.buffer_
var magnetisationBuffer *data.Slice

//ok so we want to find the effective field for complex vectors.

func SetEffectiveField(s_real, s_imag, beff_real, beff_imag *data.Slice) {

	// we assume that all the inputs live on the GPU.
	// we then just want m.buffer_ to point to something else.

	SetDemagField(s_real, beff_real)
	SetDemagField(s_real, beff_imag)

}
