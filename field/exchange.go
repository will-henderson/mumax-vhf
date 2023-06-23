package field

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

// AddExchangeComplex adds to beff_real and beff_imag, repectively, the real and imaginary components of the exchange field
// B(s) for a complex magnetisation s, with real component s_real and imaginary component s_imag.
// Note that this assumes that the inputs live on the GPU.
func AddExchangeComplex(s, b CSlice) {

	AddExchangeField(s.Real, b.Real)
	AddExchangeField(s.Imag, b.Imag)
}

// AddExchangeField adds the exchange field B(s) for a real magnetisation s to dst.
// Note that this assumes that the inputs live on the GPU.
func AddExchangeField(s, dst *data.Slice) {

	magnetisationBuffer := getMagnetisationBuffer()
	m0 := *magnetisationBuffer
	*magnetisationBuffer = s

	en.AddExchangeField(dst)

	*magnetisationBuffer = m0

}
