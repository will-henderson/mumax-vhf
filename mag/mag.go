package mag

import (
	"fmt"

	en "github.com/mumax/3/engine"
)

var (
	DynamicFactor float64
)

type SystemParameters struct {
	Nx, Ny, Nz     int
	Dx, Dy, Dz     float64
	Msat, Aex, Ku1 float64
	AnisU          [3]float64
	B_ext          [3]float64
}

func (p SystemParameters) Setup() {

	Nx = p.Nx
	Ny = p.Ny
	Nz = p.Nz
	en.SetGridSize(Nx, Ny, Nz)

	Dx = p.Dx
	Dy = p.Dy
	Dz = p.Dz
	en.SetCellSize(p.Dx, p.Dy, p.Dz)

	en.Msat.Set(p.Msat)
	en.Aex.Set(p.Aex)
	en.Ku1.Set(p.Ku1)
	en.Eval(fmt.Sprintf("AnisU = vector(%f, %f, %f)", p.AnisU[0], p.AnisU[1], p.AnisU[2]))
	en.B_ext.Set(en.Vector(p.B_ext[0], p.B_ext[1], p.B_ext[2]))

	DynamicFactor = en.GammaLL / p.Msat

}

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
