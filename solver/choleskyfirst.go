package solver

import (
	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"
	. "github.com/will-henderson/mumax-vhf/mag"
)

// A RotatedToZ solver returns the modes of the system by first rotating the system
// such that the z direction at each point coincides with the ground state direction at that point.
// As a result (and because non-null eigenmodes are perpendicular to the ground state), the zero eigenmodes can be eliminated from the system
// and hence a smaller matrix can be diagonalised.
// The time complexity is O((2*Nx*Ny*Nz)^3)
type CholeskyFirst struct {
	eigenSolver
}

// Modes returns the eigenfrequencies and corresponding eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// and assumes that the ground state magnetisation is currently stored in en.M.
// It returns 2 * Nx * Ny * Nz eigenpairs (zero eigenfrequencies are ignored)
func (solver CholeskyFirst) Modes() ([]float64, []CSlice) {
	t := LinearHamiltonianTensor()
	return solver.Solve(t)
}

// Solve returns the non-null eigenpairs of a particular input Tensor after taking cross product with the system magnetisation.
func (solver CholeskyFirst) Solve(t Tensor) ([]float64, []CSlice) {

	rot := new(mag.RotationToZ)
	rot.InitRotation()
	rotated := rot.RotateTensor(t)
	twoD := rotated.XY()
	arr := twoD.To1D()

	Cholesky(arr)
	TriInv(arr)
	LT := From1D(arr, 2, twoD.Size)
	DynamicOperateRotated(LT)

	values, vectors := Eig(3*twoD.Length(), arr)

	totalSize := 2 * twoD.Length()
	freq := make([]float64, totalSize)
	modes := make([]CSlice, totalSize)

	Nx := t.Size[0]
	Ny := t.Size[1]
	Nz := t.Size[2]

	for p := 0; p < totalSize; p++ {

		freq[p] = imag(values[p])

		mode := NewCSliceCPU(2, t.Size)
		modeReal := modes[p].Real().Vectors()
		modeImag := modes[p].Imag().Vectors()

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
