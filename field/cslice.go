package field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type CSlice struct {
	Real, Imag *data.Slice
}

func NewCSlice(nComp int, size [3]int) CSlice {

	return CSlice{
		Real: cuda.NewSlice(nComp, size),
		Imag: cuda.NewSlice(nComp, size),
	}

}

func (cs CSlice) Free() {

	cs.Real.Free()
	cs.Imag.Free()

}
