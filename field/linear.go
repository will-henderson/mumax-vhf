package field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

type LinearEvolution struct {
	groundStateField *data.Slice
}

func NewLinearEvolution() *LinearEvolution {
	return &LinearEvolution{groundStateField: GroundStateField()}
}

func (l LinearEvolution) Operate(s, res CSlice) {

	SetSIFieldComplex(s, res)
	SScal(res, res, -1)
	cuda.AddMul1D(res.Real(), l.groundStateField, s.Real())
	cuda.AddMul1D(res.Imag(), l.groundStateField, s.Imag())
	cuda.CrossProduct(res.Real(), en.M.Buffer(), res.Real())
	cuda.CrossProduct(res.Imag(), en.M.Buffer(), res.Imag())

	//now multiply by the dynamic factor: i * γ
	cuda.Scale(res.Real(), res.Real(), float32(en.GammaLL))
	cuda.Scale(res.Imag(), res.Imag(), -float32(en.GammaLL))

	res.SwitchParts()

}
