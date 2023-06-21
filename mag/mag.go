package mag

import (
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/setup"
)

func SystemTensor() Tensor {
	return AddTensors(DemagTensor(), ExchangeTensor(), UniAnisTensor())
}

func UpdateSystemTensor() Tensor {
	return AddTensors(UpdateDemagTensor(), UpdateExchangeTensor(), UpdateUniAnisTensor())
}

func LinearTensor() Tensor {

	systemTensor := SystemTensor()

	mSl := en.M.Buffer().HostCopy()
	m := mSl.Vectors() //order is Z, Y, X

	B_extSl, _ := en.B_ext.Slice()
	B_ext := B_extSl.Vectors()

	linearTensor := systemTensor.Copy()

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				zeeTerm := 0.
				for c := 0; c < 3; c++ {
					zeeTerm += float64(B_ext[c][k][j][i] * m[c][k][j][i])
				}

				gsTerm := 0.
				for i_ := 0; i_ < Nx; i_++ {
					for j_ := 0; j_ < Ny; j_++ {
						for k_ := 0; k_ < Nz; k_++ {
							for c := 0; c < 3; c++ {
								for c_ := 0; c_ < 3; c_++ {
									gsTerm += float64(m[c][k][j][i]*m[c_][k_][j_][i_]) * systemTensor.GetIdx(c, c_, i, j, k, i_, j_, k_)
								}
							}

						}
					}
				}

				for c := 0; c < 3; c++ {
					linearTensor.AddIdx(c, i, j, k, c, i, j, k, zeeTerm-gsTerm)
				}

			}
		}
	}

	return linearTensor
}
