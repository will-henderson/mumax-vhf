package mag

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

// Energy returns the energy of a magnetisation state. Calculated as .5 * Σ_rr' m_r t_rr' m_r'.
func Energy(t Tensor, mSl *data.Slice) float64 {

	if !mSl.CPUAccess() {
		mSl = mSl.HostCopy()
	}

	m := mSl.Vectors() //order is Z, Y, X

	E := 0.
	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			for k := 0; k < Nz; k++ {
				for j := 0; j < Ny; j++ {
					for i := 0; i < Nx; i++ {
						for k_ := 0; k_ < Nz; k_++ {
							for j_ := 0; j_ < Ny; j_++ {
								for i_ := 0; i_ < Nx; i_++ {
									E += t.GetIdx(c, c_, i, j, k, i_, j_, k_) * float64(m[c][k][j][i]*m[c_][k_][j_][i_])
								}
							}
						}
					}
				}
			}
		}
	}

	return .5 * E * Dx * Dy * Dz

}

// SIField returns the self-interaction field for magnetisation mSl, calculated for a given self-interaction tensor t.
// The returned slice lives on the CPU
func SIField(t Tensor, mSl *data.Slice) *data.Slice {

	if !mSl.CPUAccess() {
		mSl = mSl.HostCopy()
	}

	v := TSP(t, mSl)

	msat, rM := en.Msat.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	msat = msat.HostCopy()

	vVec := v.Vectors()
	ms := msat.Scalars()

	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					if ms[k][j][i] == 0 {
						vVec[c][k][j][i] = 0
					} else {
						vVec[c][k][j][i] = -vVec[c][k][j][i] / ms[k][j][i]
					}
				}
			}
		}
	}

	return v

}

// SIFieldComplex returns the self-interaction field for complex magnetisation mSl, calculated for a given self-interaction tensor t.
// The returned slice lives on the CPU
func SIFieldComplex(t Tensor, mSl CSlice) CSlice {

	if !mSl.CPUAccess() {
		mSl = mSl.HostCopy()
	}

	v := TCSP(t, mSl)

	msat, rM := en.Msat.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	msat = msat.HostCopy()

	vVecReal := v.Real().Vectors()
	vVecImag := v.Imag().Vectors()
	ms := msat.Scalars()

	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					if ms[k][j][i] == 0 {
						vVecReal[c][k][j][i] = 0
						vVecImag[c][k][j][i] = 0
					} else {
						vVecReal[c][k][j][i] = -vVecReal[c][k][j][i] / ms[k][j][i]
						vVecImag[c][k][j][i] = -vVecImag[c][k][j][i] / ms[k][j][i]
					}
				}
			}
		}
	}

	return v

}
