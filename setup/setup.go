package setup

import (
	"fmt"

	en "github.com/mumax/3/engine"
)

var (
	Nx, Ny, Nz int
	Dx, Dy, Dz float64

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
