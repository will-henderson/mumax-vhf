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

// this returns the operation divided by i. such that it is real.
func (l LinearEvolution) Operate(res *data.Slice, s *data.Slice) {

	SetSIField(res, s)
	cuda.Scale(res, res, -1)
	cuda.AddMul1D(res, l.groundStateField, s)
	cuda.CrossProduct(res, en.M.Buffer(), res)

	//now multiply by the dynamic factor: i * γ
	cuda.Scale(res, res, float32(en.GammaLL))

}

// we pass the return by reference
func (l LinearEvolution) OperateComplex(res *CSlice, s CSlice) {

	SetSIFieldComplex(*res, s)
	SScal(*res, *res, -1)
	cuda.AddMul1D(res.Real(), l.groundStateField, s.Real())
	cuda.AddMul1D(res.Imag(), l.groundStateField, s.Imag())
	cuda.CrossProduct(res.Real(), en.M.Buffer(), res.Real())
	cuda.CrossProduct(res.Imag(), en.M.Buffer(), res.Imag())

	//now multiply by the dynamic factor: i * γ
	cuda.Scale(res.Real(), res.Real(), float32(en.GammaLL))
	cuda.Scale(res.Imag(), res.Imag(), -float32(en.GammaLL))

	res.SwitchParts()

}
