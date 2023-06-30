// Package setup declares system parameters and tells them to mumax
package data

import (
	"fmt"

	en "github.com/mumax/3/engine"
)

var (
	Nx, Ny, Nz int
	Dx, Dy, Dz float64

	DynamicFactor float64
)

// A SystemParameters holds the variables defining the micromagnetic model
type SystemParameters struct {
	Nx, Ny, Nz     int
	Dx, Dy, Dz     float64
	Msat, Aex, Ku1 float64
	AnisU          [3]float64
	B_ext          [3]float64
}

// Setup assigns the variables to global variables, and tells them to mumax.
// This should be called after "defer en.InitAndClose()()"
func (p SystemParameters) Setup() {

	ResetGlobals()

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
