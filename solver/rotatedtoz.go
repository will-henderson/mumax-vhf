package solver

import (
	"math"

	. "github.com/will-henderson/mumax-vhf/mag"
	. "github.com/will-henderson/mumax-vhf/setup"

	en "github.com/mumax/3/engine"

	"gonum.org/v1/gonum/mat"
)

// A RotatedToZ solver returns the modes of the system by first rotating the system
// such that the z direction at each point coincides with the ground state direction at that point.
// As a result (and because non-null eigenmodes are perpendicular to the ground state), the zero eigenmodes can be eliminated from the system
// and hence a smaller matrix can be diagonalised.
// The time complexity is O((2*Nx*Ny*Nz)^3)
type RotatedToZ struct {
	eigenSolver
	R [][][][3][3]float64
}

// Modes returns the eigenfrequencies and corresponding eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// and assumes that the ground state magnetisation is currently stored in en.M.
// It returns 2 * Nx * Ny * Nz eigenpairs (zero eigenfrequencies are ignored)
func (solver RotatedToZ) Modes() ([]float64, [][3][][][]complex128) {
	t := LinearHamiltonianTensor()
	return solver.Solve(t)
}

// Solve returns the non-null eigenpairs of a particular input Tensor after taking cross product with the system magnetisation.
func (solver RotatedToZ) Solve(t Tensor) ([]float64, [][3][][][]complex128) {

	solver.initRotation()

	rotated := solver.rotateToZ(t)
	toSolve := solver.magCross(rotated)
	arr := toSolve.To1D()
	matrix := mat.NewDense(2*Nx*Ny*Nz, 2*Nx*Ny*Nz, arr)

	var eig mat.Eigen
	eig.Factorize(matrix, mat.EigenRight)

	values := eig.Values(nil)
	vectors := mat.NewCDense(2*Nx*Ny*Nz, 2*Nx*Ny*Nz, nil)
	eig.VectorsTo(vectors)
	totalSize := 3 * Nx * Ny * Nz
	freq := make([]float64, totalSize)
	modes := make([][3][][][]complex128, 3*Nx*Ny*Nz)

	for p := 0; p < totalSize; p++ {

		freq[p] = imag(values[p]) * DynamicFactor

		var mode [2][][][]complex128

		for c := 0; c < 2; c++ {
			mode[c] = make([][][]complex128, Nx)
			for i := 0; i < Nx; i++ {
				mode[c][i] = make([][]complex128, Ny)
				for j := 0; j < Ny; j++ {
					mode[c][i][j] = make([]complex128, Nz)
					for k := 0; k < Nz; k++ {
						mode[c][i][j][k] = vectors.At(p, c*Nx*Ny*Nz+i*Nz*Ny+j*Nz+k)
					}
				}
			}
		}

		modes[p] = solver.derotateMode(mode)
	}

	return freq, modes

}

// initRotation initialises the variable R, the 3D slice of pointwise rotation matrices used for rotating to z.
// It satisfies R_r m_r := (0, 0, 1) where m_r is the (supposedly) ground state magnetisation
func (solver RotatedToZ) initRotation() {

	mSl := en.M.Buffer().HostCopy()
	m := mSl.Vectors() //order is Z, Y, X

	R := make([][][][3][3]float64, Nx)

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				mx := float64(m[0][k][j][i])
				my := float64(m[1][k][j][i])
				mz := float64(m[2][k][j][i])

				Cth := mz
				Sth := math.Sqrt(1 - mz*mz)
				Cph := mx / Sth
				Sph := my / Sth

				//edge case where Sth == 0!!!

				R[i][j][k] = [3][3]float64{
					{Cth * Cph, Cth * Sph, -Sth},
					{-Sph, Cph, 0},
					{Sth * Cph, Sth * Sph, Cth},
				}
			}
		}
	}
}

// rotateToZ applies the pointwise rotation matrix to the Tensor t
// It returns R_r t_rr' R_r.T
func (solver RotatedToZ) rotateToZ(t Tensor) Tensor {

	copy := t.Copy()

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				R := solver.R[i][j][k]

				for i_ := 0; i_ < Nx; i_++ {
					for j_ := 0; j_ < Ny; j_++ {
						for k_ := 0; k_ < Nz; k_++ {

							// do the matrix multiplication

							var temp [3][3]float64

							for p := 0; p < 3; p++ {
								for q := 0; q < 3; q++ {
									temp[p][q] = 0
									for r := 0; r < 3; r++ {
										temp[p][q] += R[p][r] * copy.GetIdx(r, q, i, j, k, i_, j_, k_)
									}
								}
							}

							for p := 0; p < 3; p++ {
								for q := 0; q < 3; q++ {
									copy.SetIdx(p, q, i, j, k, i_, j_, k_, temp[p][q])
								}
							}

							// and do it for the other side with the transpose of the rotation matrix.
							for p := 0; p < 3; p++ {
								for q := 0; q < 3; q++ {
									temp[p][q] = 0
									for r := 0; r < 3; r++ {
										temp[p][q] += R[q][r] * copy.GetIdx(p, r, i_, j_, k_, i, j, k)
									}
								}
							}

							for p := 0; p < 3; p++ {
								for q := 0; q < 3; q++ {
									copy.SetIdx(p, q, i_, j_, k_, i, j, k, temp[p][q])
								}
							}
						}
					}
				}

			}
		}
	}

	return copy

}

// magCross performs the cross product of the input tensor with the system magnetisation,
// working in the rotated-to-z basis.
func (solver RotatedToZ) magCross(t Tensor) Tensor {

	result := Zeros()

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {
				for i_ := 0; i_ < Nx; i_++ {
					for j_ := 0; j_ < Ny; j_++ {
						for k_ := 0; k_ < Nz; k_++ {

							for q := 0; q < 3; q++ {
								result.SetIdx(0, q, i, j, k, i_, j_, k_, -t.GetIdx(1, q, i, j, k, i_, j_, k_))
								result.SetIdx(1, q, i, j, k, i_, j_, k_, t.GetIdx(0, q, i, j, k, i_, j_, k_))
							}

						}
					}
				}

			}
		}
	}

	return result
}

// derotateMode rotates a mode back to the original basis.
// It returns R_r.T (mode_r)
func (solver RotatedToZ) derotateMode(mode [2][][][]complex128) [3][][][]complex128 {

	var derotated [3][][][]complex128

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				R := solver.R[i][j][k]

				for p := 0; p < 3; p++ {
					derotated[p][i][j][k] = 0
					for q := 0; q < 2; q++ {
						derotated[p][i][j][k] += mode[q][i][j][k] * complex(R[q][p], 0)
					}
				}

			}
		}
	}

	return derotated

}
