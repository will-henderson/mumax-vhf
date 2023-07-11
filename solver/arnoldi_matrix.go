package solver

import (
	"fmt"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"
	. "github.com/will-henderson/mumax-vhf/mag"

	"github.com/mumax/3/util"
)

// A RotatedToZ solver returns the modes of the system by first rotating the system
// such that the z direction at each point coincides with the ground state direction at that point.
// As a result (and because non-null eigenmodes are perpendicular to the ground state), the zero eigenmodes can be eliminated from the system
// and hence a smaller matrix can be diagonalised.
// The time complexity is O((2*Nx*Ny*Nz)^3)
type ArnoldiMatrix struct {
	eigenSolver
}

// Modes returns the eigenfrequencies and corresponding eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// and assumes that the ground state magnetisation is currently stored in en.M.
// It returns 2 * Nx * Ny * Nz eigenpairs (zero eigenfrequencies are ignored)
func (solver ArnoldiMatrix) Modes() ([]float64, []CSlice) {
	t := EigenProblemTensor()
	return solver.Solve(t)
}

// Solve returns the non-null eigenpairs of a particular input Tensor after taking cross product with the system magnetisation.
func (solver ArnoldiMatrix) Solve(t Tensor) ([]float64, []CSlice) {

	rot := new(mag.RotationToZ)
	rot.InitRotation()
	rotated := rot.RotateTensor(t)
	twoD := rotated.XY()
	arr := twoD.To1D()

	totalSize := 2 * twoD.Length()

	arn := newArnoldiD(totalSize, totalSize-2, -1, "I", "SM", 0, 100*totalSize, nil)

	ido, x, y := arn.iterate()
	for ido == 1 || ido == -1 {
		matvecmul(totalSize, totalSize, arr, x, y)
		ido, x, y = arn.iterate()
	}

	info, infoString := arn.iterateInfo()
	util.AssertMsg(info == 0, infoString)

	values, vectors := arn.extract(true, nil)
	info, infoString = arn.extractInfo()
	util.AssertMsg(info == 0, infoString)

	nevReturned := len(values)
	iterations := arn.iparam[2]
	util.Log(fmt.Sprintf("Found %d eigenvalues (out of total dimension %d) in %d iterations.", nevReturned, totalSize, iterations))

	freq := make([]float64, nevReturned)
	modes := make([]CSlice, nevReturned)

	Nx := t.Size[0]
	Ny := t.Size[1]
	Nz := t.Size[2]

	for p := 0; p < nevReturned; p++ {

		freq[p] = imag(values[p])

		mode := NewCSliceCPU(2, t.Size)
		modeReal := mode.Real().Tensors()
		modeImag := mode.Imag().Tensors()

		for c := 0; c < 2; c++ {
			for i := 0; i < Nx; i++ {
				for j := 0; j < Ny; j++ {
					for k := 0; k < Nz; k++ {
						modeReal[c][k][j][i] = float32(real(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
						modeImag[c][k][j][i] = float32(imag(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
					}
				}
			}
		}

		modes[p] = rot.DerotateMode(mode)
	}

	return freq, modes

}
