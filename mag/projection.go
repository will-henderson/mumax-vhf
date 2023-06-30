package mag

import (
	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"

	"math"
)

type RotationToZ struct {
	R [][][][3][3]float64
}

// InitRotation initialises the variable R, the 3D slice of pointwise rotation matrices used for rotating to z.
// It satisfies R_r m_r := (0, 0, 1) where m_r is the (supposedly) ground state magnetisation
func (rtz RotationToZ) InitRotation() {

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
func (rtz RotationToZ) RotateTensor(t Tensor) Tensor {

	copy := t.Copy()

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				R := rtz.R[i][j][k]

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

// derotateMode rotates a mode back to the original basis.
// It returns R_r.T (mode_r)
func (rtz RotationToZ) DerotateMode(mode [2][][][]complex128) [3][][][]complex128 {

	var derotated [3][][][]complex128

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				R := rtz.R[i][j][k]

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
