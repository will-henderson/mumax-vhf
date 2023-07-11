// Package mag calculates self-interaction tensors corresponding to exchange, demagnetising and uniaxial anisotropy interactions.
// Additionally it calculates the linear Hamiltonian tensor that results from these.
package mag

import (
	"github.com/mumax/3/cuda"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

// SelfInteractionTensor returns the self-interaction tensor with contibutions from the Demagnetising, Exchange, and Uniaxial Anisotropy interactions.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
func SelfInteractionTensor() Tensor {
	return AddTensors(DemagTensor(), ExchangeTensor(), UniAnisTensor())
}

// LinearHamiltonianTensor returns the tensor representation of the linear Hamiltonian of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
func LinearHamiltonianTensor() Tensor {

	systemTensor := SelfInteractionTensor()

	Nx := systemTensor.Size[0]
	Ny := systemTensor.Size[1]
	Nz := systemTensor.Size[2]

	mSl := en.M.Buffer().HostCopy()
	m := mSl.Vectors() //order is Z, Y, X

	B_extGPU, rM := en.B_ext.Slice()
	B_extSl := B_extGPU.HostCopy()
	if rM {
		cuda.Recycle(B_extGPU)
	}

	msatGPU, rM := en.Msat.Slice()
	msat := msatGPU.HostCopy()
	if rM {
		cuda.Recycle(msatGPU)
	}
	ms := msat.Scalars()

	B_ext := B_extSl.Vectors()

	ret := systemTensor.Copy()

	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {

				zeeTerm := 0.
				for c := 0; c < 3; c++ {
					zeeTerm += float64(B_ext[c][k][j][i] * m[c][k][j][i])
				}

				gsTerm := 0.
				for i_ := 0; i_ < Nx; i_++ {
					for j_ := 0; j_ < Ny; j_++ {
						for k_ := 0; k_ < Nz; k_++ {
							for c := 0; c < 3; c++ {
								for c_ := 0; c_ < 3; c_++ {
									gsTerm += float64(m[c][k][j][i]*m[c_][k_][j_][i_]) * systemTensor.GetIdx(c, c_, i, j, k, i_, j_, k_)
								}
							}

						}
					}
				}

				for c := 0; c < 3; c++ {
					ret.AddIdx(c, c, i, j, k, i, j, k, zeeTerm*float64(ms[k][j][i])-gsTerm)
				}

			}
		}
	}
	return ret
}

// EigenProblemTensor returns the tensor representation of the matrix (divided by i, so real)
// which is diagonalised to find the eigenfrequencies and eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
func EigenProblemTensor() Tensor {

	lht := LinearHamiltonianTensor()
	ept := DynamicOperate(lht)
	return ept

}

func DynamicOperate(t Tensor) Tensor {

	m := en.M.Buffer().HostCopy().Vectors()

	msatGPU, rM := en.Msat.Slice()
	msat := msatGPU.HostCopy()
	if rM {
		cuda.Recycle(msatGPU)
	}
	ms := msat.Scalars()

	γ := en.GammaLL

	result := ZeroTensor(t.NComp, t.Size)

	for k := 0; k < t.Size[2]; k++ {
		for j := 0; j < t.Size[1]; j++ {
			for i := 0; i < t.Size[0]; i++ {

				mx := float64(m[0][k][j][i])
				my := float64(m[1][k][j][i])
				mz := float64(m[2][k][j][i])

				m_cross := [3][3]float64{
					{0, -mz, my},
					{mz, 0, -mx},
					{-my, mx, 0},
				}

				for k_ := 0; k_ < t.Size[2]; k_++ {
					for j_ := 0; j_ < t.Size[1]; j_++ {
						for i_ := 0; i_ < t.Size[0]; i_++ {

							// do the matrix multiplication

							for p := 0; p < 3; p++ {
								for q := 0; q < 3; q++ {
									for r := 0; r < 3; r++ {
										result.AddIdx(p, q, i, j, k, i_, j_, k_, m_cross[p][r]*t.GetIdx(r, q, i, j, k, i_, j_, k_))
									}

									// multiply by the dynamic factor.
									result.SetIdx(p, q, i, j, k, i_, j_, k_,
										result.GetIdx(p, q, i, j, k, i_, j_, k_)*γ/float64(ms[k][j][i]))

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

func DynamicOperateRotated(t Tensor) Tensor {

	msatGPU, rM := en.Msat.Slice()
	msat := msatGPU.HostCopy()
	if rM {
		cuda.Recycle(msatGPU)
	}
	ms := msat.Scalars()

	γ := en.GammaLL

	result := ZeroTensor(t.NComp, t.Size)

	for k := 0; k < t.Size[2]; k++ {
		for j := 0; j < t.Size[1]; j++ {
			for i := 0; i < t.Size[0]; i++ {
				for k_ := 0; k_ < t.Size[2]; k_++ {
					for j_ := 0; j_ < t.Size[1]; j_++ {
						for i_ := 0; i_ < t.Size[0]; i_++ {
							for q := 0; q < 3; q++ {
								result.SetIdx(0, q, i, j, k, i_, j_, k_, -t.GetIdx(1, q, i, j, k, i_, j_, k_)*γ/float64(ms[k][j][i]))
								result.SetIdx(1, q, i, j, k, i_, j_, k_, t.GetIdx(0, q, i, j, k, i_, j_, k_)*γ/float64(ms[k][j][i]))
							}
						}
					}
				}

			}
		}
	}

	return result

}
