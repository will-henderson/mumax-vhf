package field

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"
)

// SetDemagComplex sets the the slices beff_real and beff_imag to, repectively, the real and imaginary components of the demagnetising field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func SetDemagComplex(b, s CSlice) {
	SetDemagField(b.Real(), s.Real())
	SetDemagField(b.Imag(), s.Imag())
}

// SetDemagField adds the anisotropy field B(s) for a real magnetisation s to dst.
// Note that this assumes that the inputs live on the GPU.
func SetDemagField(dst, s *data.Slice) {

	magnetisationBuffer := getMagnetisationBuffer()
	m0 := *magnetisationBuffer

	*magnetisationBuffer = s

	en.SetDemagField(dst)

	*magnetisationBuffer = m0

}
