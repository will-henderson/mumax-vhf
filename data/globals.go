package data

import (
	"github.com/mumax/3/data"
)

var (
	TensorFieldFactor_ *data.Slice
)

func ResetGlobals() {
	TensorFieldFactor_.Free()
	TensorFieldFactor_ = nil
}
