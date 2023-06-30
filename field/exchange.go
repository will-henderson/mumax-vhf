package field

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"
)

// AddExchangeComplex adds to beff_real and beff_imag, repectively, the real and imaginary components of the exchange field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func AddExchangeComplex(b, s CSlice) {

	AddExchangeField(b.Real(), s.Real())
	AddExchangeField(b.Imag(), s.Imag())
}

// AddExchangeField adds the exchange field B(s) for a real magnetisation s to dst.
// Note that this assumes that the inputs live on the GPU.
func AddExchangeField(dst, s *data.Slice) {

	magnetisationBuffer := getMagnetisationBuffer()
	m0 := *magnetisationBuffer
	*magnetisationBuffer = s

	en.AddExchangeField(dst)

	*magnetisationBuffer = m0

}
