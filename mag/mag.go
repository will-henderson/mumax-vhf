package mag

import (
	"fmt"

	en "github.com/mumax/3/engine"
)

type SystemParameters struct {
	nx, ny, nz     int
	dx, dy, dz     float64
	Msat, Aex, Ku1 float64
	AnisU          [3]float64
	B_ext          [3]float64
}

func (p SystemParameters) Setup() {

	Nx = p.nx
	Ny = p.ny
	Nz = p.nz
	en.SetGridSize(Nx, Ny, Nz)

	Dx = p.dx
	Dy = p.dy
	Dz = p.dz
	en.SetCellSize(p.dx, p.dy, p.dz)

	en.Msat.Set(p.Msat)
	en.Aex.Set(p.Aex)
	en.Ku1.Set(p.Ku1)
	en.Eval(fmt.Sprintf("AnisU = vector(%f, %f, %f)", p.AnisU[0], p.AnisU[1], p.AnisU[2]))
	en.B_ext.Set(en.Vector(p.B_ext[0], p.B_ext[1], p.B_ext[2]))

}

func SystemTensor() Tensor {
	return AddTensors(DemagTensor(), ExchangeTensor(), UniAnisTensor())
}

func UpdateSystemTensor() Tensor {
	return AddTensors(UpdateDemagTensor(), UpdateExchangeTensor(), UpdateUniAnisTensor())
}
