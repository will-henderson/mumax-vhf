package tests

import (
	"math"

	. "github.com/will-henderson/mumax-vhf/data"
)

func Normality(vecs []CSlice, maxErr float64) int {

	n := 0
	for _, v := range vecs {
		norm := dotc(v, v)
		if math.Abs(1-real(norm)) > maxErr {
			n++
		}
	}
	return n
}
