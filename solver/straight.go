package solver

import (
	. "github.com/will-henderson/mumax-vhf/mag"
	. "github.com/will-henderson/mumax-vhf/setup"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	"gonum.org/v1/gonum/mat"
)

type Straight struct {
	eigenSolver
}

func (solver Straight) Modes() ([]float64, [][3][][][]complex128) {
	t := LinearTensor()
	return solver.Solve(t)
}

func (solver Straight) Solve(t Tensor) ([]float64, [][3][][][]complex128) {

	toSolve := solver.magCross(t)
	arr := toSolve.To1D()
	matrix := mat.NewDense(3*Nx*Ny*Nz, 3*Nx*Ny*Nz, arr)

	var eig mat.Eigen
	eig.Factorize(matrix, mat.EigenRight)

	values := eig.Values(nil)
	vectors := mat.NewCDense(3*Nx*Ny*Nz, 3*Nx*Ny*Nz, nil)
	eig.VectorsTo(vectors)

	totalSize := 3 * Nx * Ny * Nz
	freqs := make([]float64, totalSize)
	modes := make([][3][][][]complex128, 3*Nx*Ny*Nz)

	q := 0
	for p := 0; p < totalSize; p++ {

		freq := imag(values[p])

		if freq != 0 {

			freqs[q] = freq * DynamicFactor

			for c := 0; c < 3; c++ {
				modes[q][c] = make([][][]complex128, Nx)
				for i := 0; i < Nx; i++ {
					modes[q][c][i] = make([][]complex128, Ny)
					for j := 0; j < Ny; j++ {
						modes[q][c][i][j] = make([]complex128, Nz)
						for k := 0; k < Nz; k++ {
							modes[q][c][i][j][k] = vectors.At(q, c*Nx*Ny*Nz+i*Nz*Ny+j*Nz+k)
						}
					}
				}
			}

			q++

		}
	}

	return freqs, modes

}

func (solver Straight) magCross(t Tensor) Tensor {

	mSl := data.NewSlice(3, en.Mesh().Size())
	en.M.EvalTo(mSl)
	m := mSl.Vectors()

	result := Zeros()

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				mx := float64(m[0][k][j][i])
				my := float64(m[1][k][j][i])
				mz := float64(m[2][k][j][i])

				m_cross := [3][3]float64{
					{0, -mz, my},
					{mz, 0, -mx},
					{-my, mx, 0},
				}

				for i_ := 0; i_ < Nx; i_++ {
					for j_ := 0; j_ < Ny; j_++ {
						for k_ := 0; k_ < Nz; k_++ {

							// do the matrix multiplication

							for p := 0; p < 3; p++ {
								for q := 0; q < 3; q++ {
									for r := 0; r < 3; r++ {
										result.AddIdx(p, q, i, j, k, i_, j_, k_, m_cross[p][r]*t.GetIdx(r, q, i, j, k, i_, j_, k_))
									}
								}
							}
						}
					}
				}

			}
		}
	}

	return result
}
