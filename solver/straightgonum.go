package solver

import (
	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"

	"gonum.org/v1/gonum/mat"
)

// A Straight solver returns the modes in the simplest way possible. By diagonalising the LinearTensor of the system directly.
// The time complexity is O((3*Nx*Ny*Nz)^3)
type StraightGonum struct {
	eigenSolver
}

// Modes returns the eigenfrequencies and corresponding eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// and assumes that the ground state magnetisation is currently stored in en.M.
// It returns 2 * Nx * Ny * Nz eigenpairs (zero eigenfrequencies are ignored)
func (solver StraightGonum) Modes() ([]float64, []CSlice) {
	t := mag.EigenProblemTensor()
	return solver.Solve(t)
}

// Solve returns the non-null eigenpairs of a particular input Tensor after taking cross product with the system magnetisation.
func (solver StraightGonum) Solve(t Tensor) ([]float64, []CSlice) {

	arr := t.To1D()

	matrix := mat.NewDense(3*t.Length(), 3*t.Length(), arr)

	var eig mat.Eigen
	eig.Factorize(matrix, mat.EigenRight)

	values := eig.Values(nil)
	vectors := mat.NewCDense(3*t.Length(), 3*t.Length(), nil)
	eig.VectorsTo(vectors)

	totalSize := 2 * t.Length()
	freqs := make([]float64, totalSize)
	modes := make([]CSlice, totalSize)

	Nx := t.Size[0]
	Ny := t.Size[1]
	Nz := t.Size[2]

	q := 0
	for p := 0; p < totalSize; p++ {

		freq := imag(values[p])

		if freq != 0 {

			freqs[q] = freq

			modes[q] = NewCSliceCPU(3, t.Size)
			modeReal := modes[q].Real().Vectors()
			modeImag := modes[q].Imag().Vectors()

			for c := 0; c < 3; c++ {
				for k := 0; k < t.Size[2]; k++ {
					for j := 0; j < t.Size[1]; j++ {
						for i := 0; i < t.Size[0]; i++ {
							modeReal[c][k][j][i] = float32(real(vectors.At(c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i, q)))
							modeImag[c][k][j][i] = float32(imag(vectors.At(c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i, q)))
						}
					}
				}
			}

			q++

		}
	}

	return freqs, modes

}
