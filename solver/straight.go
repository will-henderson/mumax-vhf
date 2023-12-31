package solver

import (
	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"
)

// A Straight solver returns the modes in the simplest way possible. By diagonalising the LinearTensor of the system directly.
// The time complexity is O((3*Nx*Ny*Nz)^3)
type Straight struct {
	eigenSolver
}

// Modes returns the eigenfrequencies and corresponding eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// and assumes that the ground state magnetisation is currently stored in en.M.
// It returns 2 * Nx * Ny * Nz eigenpairs (zero eigenfrequencies are ignored)
func (solver Straight) Modes() ([]float64, []CSlice) {
	t := mag.EigenProblemTensor()
	return solver.Solve(t)
}

// Solve returns the non-null eigenpairs of a particular input Tensor after taking cross product with the system magnetisation.
func (solver Straight) Solve(t Tensor) ([]float64, []CSlice) {

	arr := t.To1D()

	values, vectors := Eig(3*t.Length(), arr)

	return processStraight(values, vectors, t.Size)

}

func processStraight(values []complex128, vectors [][]complex128, size [3]int) ([]float64, []CSlice) {

	Nx := size[0]
	Ny := size[1]
	Nz := size[2]

	totalSize := 2 * Nx * Ny * Nz
	freqs := make([]float64, totalSize)
	modes := make([]CSlice, totalSize)

	q := 0
	for p := 0; p < totalSize; p++ {

		freq := imag(values[p])

		if freq != 0 {

			freqs[q] = freq
			modes[q] = NewCSliceCPU(3, size)
			modeReal := modes[q].Real().Vectors()
			modeImag := modes[q].Imag().Vectors()

			for c := 0; c < 3; c++ {
				for k := 0; k < size[2]; k++ {
					for j := 0; j < size[1]; j++ {
						for i := 0; i < size[0]; i++ {
							modeReal[c][k][j][i] = float32(real(vectors[q][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
							modeImag[c][k][j][i] = float32(imag(vectors[q][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
						}
					}
				}
			}

			q++

		}
	}

	return freqs, modes
}
