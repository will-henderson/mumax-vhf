package field

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"
)

// AddAnisotropyComplex adds to beff_real and beff_imag, repectively, the real and imaginary components of the anisotropy field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func AddAnisotropyComplex(b, s CSlice) {

	AddAnisotropyField(b.Real(), s.Real())
	AddAnisotropyField(b.Imag(), s.Imag())
}

// AddAnisotropyField adds the anisotropy field B(s) for a real magnetisation s to dst.
// Note that this assumes that the inputs live on the GPU.
func AddAnisotropyField(dst, s *data.Slice) {

	magnetisationBuffer := GetMagnetisationBuffer()
	m0 := *magnetisationBuffer
	*magnetisationBuffer = s

	en.AddAnisotropyField(dst)

	*magnetisationBuffer = m0

}
