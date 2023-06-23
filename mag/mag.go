// Package mag calculates self-interaction tensors corresponding to exchange, demagnetising and uniaxial anisotropy interactions.
// Additionally it calculates the linear Hamiltonian tensor that results from these.
package mag

import (
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/setup"
)

var (
	linearTensor *Tensor
)

// SystemTensor returns the self-interaction tensor with contibutions from the Demagnetising, Exchange, and Uniaxial Anisotropy interactions.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
func SystemTensor() Tensor {
	return AddTensors(DemagTensor(), ExchangeTensor(), UniAnisTensor())
}

// UpdateSystemTensor returns the self-interaction tensor with contibutions from the Demagnetising, Exchange, and Uniaxial Anisotropy interactions.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// This forces the updating of the self-interaction tensor, rather than potentially returning a cached value.
func UpdateSystemTensor() Tensor {
	return AddTensors(UpdateDemagTensor(), UpdateExchangeTensor(), UpdateUniAnisTensor())
}

// LinearTensor returns the tensor representation of the linear Hamiltonian of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
func LinearTensor() Tensor {
	if linearTensor == nil {
		return UpdateLinearTensor()
	} else {
		return *linearTensor
	}
}

// UpdateLinearTensor returns the tensor representation of the linear Hamiltonian of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// This forces updating of self-interaction tensors and zeeman energy, as well as accounting for possibly changed ground state.
func UpdateLinearTensor() Tensor {

	systemTensor := UpdateSystemTensor()

	mSl := en.M.Buffer().HostCopy()
	m := mSl.Vectors() //order is Z, Y, X

	B_extSl, _ := en.B_ext.Slice()
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
					ret.AddIdx(c, i, j, k, c, i, j, k, zeeTerm-gsTerm)
				}

			}
		}
	}

	linearTensor = &ret
	return ret
}
