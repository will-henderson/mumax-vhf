package mag

import (
	en "github.com/mumax/3/engine"
)

var (
	uniAnisTensor *Tensor
)

func UniAnisTensor() Tensor {
	if uniAnisTensor == nil {
		return UpdateUniAnisTensor()
	} else {
		return *uniAnisTensor
	}
}

func UpdateUniAnisTensor() Tensor {

	unianis_factor := -2 * en.Ku1.GetRegion(0) * Dx * Dy * Dz
	direction := en.AnisU.GetRegion(0)

	var op_direction [3][3]float64

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			op_direction[c][c_] = direction[c] * direction[c_] * unianis_factor
		}
	}

	t := Zeros()

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			for i := 0; i < Nx; i++ {
				for j := 0; j < Ny; j++ {
					for k := 0; k < Nz; k++ {
						t.SetIdx(c, c_, i, j, k, i, j, k, op_direction[c][c_])
					}
				}
			}
		}
	}

	uniAnisTensor = &t
	return t

}