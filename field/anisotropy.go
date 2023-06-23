package field

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

// AddAnisotropyComplex adds to beff_real and beff_imag, repectively, the real and imaginary components of the anisotropy field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func AddAnisotropyComplex(s, b CSlice) {

	AddAnisotropyField(s.Real, b.Real)
	AddAnisotropyField(s.Imag, b.Imag)
}

// AddAnisotropyField adds the anisotropy field B(s) for a real magnetisation s to dst.
// Note that this assumes that the inputs live on the GPU.
func AddAnisotropyField(s, dst *data.Slice) {

	magnetisationBuffer := getMagnetisationBuffer()
	m0 := *magnetisationBuffer
	*magnetisationBuffer = s

	en.AddAnisotropyField(dst)

	*magnetisationBuffer = m0

}
