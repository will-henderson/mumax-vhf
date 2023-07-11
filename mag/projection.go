package mag

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"

	"math"
)

// R is stored in order z, y, x
type RotationToZ struct {
	R [][][][3][3]float64
}

// InitRotation initialises the variable R, the 3D slice of pointwise rotation matrices used for rotating to z.
// It satisfies R_r m_r := (0, 0, 1) where m_r is the (supposedly) ground state magnetisation
func (rtz *RotationToZ) InitRotation() {

	mSl := en.M.Buffer().HostCopy()
	m := mSl.Vectors() //order is Z, Y, X

	rtz.R = make([][][][3][3]float64, mSl.Size()[2])

	for k := 0; k < mSl.Size()[2]; k++ {
		rtz.R[k] = make([][][3][3]float64, mSl.Size()[1])
		for j := 0; j < mSl.Size()[1]; j++ {
			rtz.R[k][j] = make([][3][3]float64, mSl.Size()[0])
			for i := 0; i < mSl.Size()[0]; i++ {

				mx := float64(m[0][k][j][i])
				my := float64(m[1][k][j][i])
				mz := float64(m[2][k][j][i])

				Cth := mz
				Sth := math.Sqrt(1 - mz*mz)
				Cph := mx / Sth
				Sph := my / Sth

				//edge case where Sth == 0!!!

				rtz.R[k][j][i] = [3][3]float64{
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
	Nx := t.Size[0]
	Ny := t.Size[1]
	Nz := t.Size[2]

	for k := 0; k < Nz; k++ {
		for j := 0; j < Ny; j++ {
			for i := 0; i < Nx; i++ {

				R := rtz.R[k][j][i]

				for k_ := 0; k_ < Nz; k_++ {
					for j_ := 0; j_ < Ny; j_++ {
						for i_ := 0; i_ < Nx; i_++ {

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
// It returns R_r.T (mode_r).
// It assumes the input CSlice lives on the CPU, and two dimensional, i.e. living in the space perpendicular to the ground state magnetisation.
func (rtz RotationToZ) DerotateMode(mode CSlice) CSlice {

	// the input slice is assumed to be 2
	modeReal := mode.Real().Tensors()
	modeImag := mode.Imag().Tensors()

	derotated := NewCSliceCPU(3, mode.Size())
	derotatedReal := derotated.Real().Vectors()
	derotatedImag := derotated.Imag().Vectors()

	for k := 0; k < mode.Size()[2]; k++ {
		for j := 0; j < mode.Size()[1]; j++ {
			for i := 0; i < mode.Size()[0]; i++ {

				R := rtz.R[k][j][i]

				for p := 0; p < 3; p++ {
					derotatedReal[p][k][j][i] = 0
					derotatedImag[p][k][j][i] = 0
					for q := 0; q < 2; q++ {
						derotatedReal[p][k][j][i] += modeReal[q][k][j][i] * float32(R[q][p])
						derotatedImag[p][k][j][i] += modeImag[q][k][j][i] * float32(R[q][p])
					}
				}

			}
		}
	}

	return derotated

}

func (rtz RotationToZ) RotateMode(mode CSlice) CSlice {

	// the input slice is assumed to have 3 components. The output slice will have two
	modeReal := mode.Real().Vectors()
	modeImag := mode.Imag().Vectors()

	rotated := NewCSliceCPU(2, mode.Size())
	rotatedReal := rotated.Real().Tensors()
	rotatedImag := rotated.Imag().Tensors()

	for k := 0; k < mode.Size()[2]; k++ {
		for j := 0; j < mode.Size()[1]; j++ {
			for i := 0; i < mode.Size()[0]; i++ {

				R := rtz.R[k][j][i]

				for p := 0; p < 2; p++ {
					rotatedReal[p][k][j][i] = 0
					rotatedImag[p][k][j][i] = 0
					for q := 0; q < 3; q++ {
						rotatedReal[p][k][j][i] += modeReal[q][k][j][i] * float32(R[p][q])
						rotatedImag[p][k][j][i] += modeImag[q][k][j][i] * float32(R[p][q])
					}
				}

			}
		}
	}

	return rotated

}

func (rtz RotationToZ) DerotateModeReal(mode *data.Slice) *data.Slice {

	// the input slice is assumed to be 2
	modet := mode.Tensors()

	derotated := data.NewSlice(3, mode.Size())
	derotatedt := derotated.Vectors()

	for k := 0; k < mode.Size()[2]; k++ {
		for j := 0; j < mode.Size()[1]; j++ {
			for i := 0; i < mode.Size()[0]; i++ {

				R := rtz.R[k][j][i]

				for p := 0; p < 3; p++ {
					derotatedt[p][k][j][i] = 0
					for q := 0; q < 2; q++ {
						derotatedt[p][k][j][i] += modet[q][k][j][i] * float32(R[q][p])
					}
				}

			}
		}
	}

	return derotated

}

func (rtz RotationToZ) RotateModeReal(mode *data.Slice) *data.Slice {

	// the input slice is assumed to have 3 components. The output slice will have two
	modet := mode.Vectors()

	rotated := data.NewSlice(2, mode.Size())
	rotatedt := rotated.Tensors()

	for k := 0; k < mode.Size()[2]; k++ {
		for j := 0; j < mode.Size()[1]; j++ {
			for i := 0; i < mode.Size()[0]; i++ {

				R := rtz.R[k][j][i]

				for p := 0; p < 2; p++ {
					for q := 0; q < 3; q++ {
						rotatedt[p][k][j][i] += modet[q][k][j][i] * float32(R[p][q])
					}
				}

			}
		}
	}

	return rotated

}
