package tests

import (
	"github.com/will-henderson/mumax-vhf/setup"
)

func EqualScalars(A, B float64, maxErr float64) int {
	frac := A / B
	n := 0
	if frac > 1.01 || frac < .99 {
		n = 1
	}
	return n
}

func EqualFields(A, B [3][][][]float32, maxErr float32) int {

	n := 0
	for c := 0; c < 3; c++ {
		for i := 0; i < setup.Nx; i++ {
			for j := 0; j < setup.Ny; j++ {
				for k := 0; k < setup.Nz; k++ {
					frac := A[c][k][j][i] / B[c][k][j][i]
					if frac > 1+maxErr || frac < 1-maxErr {
						n++
					}
				}
			}
		}
	}
	return n
}
