// Package solver contains structs with a Modes function to calculate the eigenmodes and corresponding eigenfrequencies of the system.
package solver

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
	. "github.com/will-henderson/mumax-vhf/data"
)

// EigenSolver is an interface which wraps the Modes method.
// Modes returns the eigenfrequencies and corresponding eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// and assumes that the ground state magnetisation is currently stored in en.M.
// It returns 2 * Nx * Ny * Nz eigenpairs (zero eigenfrequencies are ignored)
type EigenSolver interface {
	Modes() ([]float64, []CSlice) //returns real eigenvalues and the eigenvectors
}

// Types implementing the EigenSolver interface should extend the eigenSolver struct.
// This will allow convience methods to be added to all solvers in the future.
type eigenSolver struct{}

var (
	Solver EigenSolver
)

func SetSolver(solverName string) {
	switch solverName {
	case "StraightGonum":
		Solver = new(StraightGonum)
	case "RotatedToZ":
		Solver = new(RotatedToZ)
	default:
		panic("solver not known")
	}
}

func Modes() ([]float64, []CSlice) {
	return Solver.Modes()
}

func WriteModes(frequencies []float64, modes []CSlice, name string) {
	//probably want to make a directory

	os.Mkdir(name, os.ModePerm)

	for i := 0; i < len(frequencies); i++ {

		//write the real part
		info := data.Meta{Time: frequencies[i], Name: "Real Eigenmode", Unit: fmt.Sprint(frequencies[i]),
			CellSize: en.Mesh().CellSize()}

		fname := filepath.Join(name, fmt.Sprintf("%d_real.ovf", i))
		f, err := httpfs.Create(fname)
		util.FatalErr(err)
		oommf.WriteOVF2(f, modes[i].Real(), info, "binary 4")
		f.Close()

		//write the real part
		info = data.Meta{Time: frequencies[i], Name: "Imag Eigenmode", Unit: "1",
			CellSize: en.Mesh().CellSize()}

		fname = filepath.Join(name, fmt.Sprintf("%d_imag.ovf", i))
		f, err = httpfs.Create(fname)
		util.FatalErr(err)
		oommf.WriteOVF2(f, modes[i].Imag(), info, "binary 4")
		f.Close()
	}

}
