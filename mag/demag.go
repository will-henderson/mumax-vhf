package mag

import (
	en "github.com/mumax/3/engine"
	"github.com/mumax/3/mag"

	. "github.com/will-henderson/mumax-vhf/data"
)

// DemagTensor returns the self-interaction tensor for the Demagnetising interaction.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
func DemagTensor() Tensor {

	kernel := mag.DemagKernel(en.Mesh().Size(), en.Mesh().PBC(), en.Mesh().CellSize(), en.DemagAccuracy, *en.Flag_cachedir)
	t := Zeros()

	Msat := en.Msat.GetRegion(0)
	demag_factor := -Msat * Msat * mag.Mu0

	size := kernel[0][0].Size()

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {

			if Nz == 1 {
				if (c == 2 && c_ != 2) || (c_ == 2 && c != 2) {
					for i := 0; i < Nx; i++ {
						for j := 0; j < Ny; j++ {
							for k := 0; k < Nz; k++ {
								for i_ := 0; i_ < Nx; i_++ {
									for j_ := 0; j_ < Ny; j_++ {
										for k_ := 0; k_ < Nz; k_++ {
											t.SetIdx(c, c_, i, j, k, i_, j_, k, 0)
										}
									}
								}
							}
						}
					}
				} else {
					array := kernel[c][c_].Scalars()
					for i := 0; i < Nx; i++ {
						for j := 0; j < Ny; j++ {
							for k := 0; k < Nz; k++ {
								for i_ := 0; i_ < Nx; i_++ {
									for j_ := 0; j_ < Ny; j_++ {
										for k_ := 0; k_ < Nz; k_++ {
											t.SetIdx(c, c_, i, j, k, i_, j_, k_, float64(array[0][mod(j-j_, size[1])][mod(i-i_, size[0])])*demag_factor)
										}
									}
								}
							}
						}
					}
				}
			} else {
				array := kernel[c][c_].Scalars()
				for i := 0; i < Nx; i++ {
					for j := 0; j < Ny; j++ {
						for k := 0; k < Nz; k++ {
							for i_ := 0; i_ < Nx; i_++ {
								for j_ := 0; j_ < Ny; j_++ {
									for k_ := 0; k_ < Nz; k_++ {

										t.SetIdx(c, c_, i, j, k, i_, j_, k_, float64(array[mod(k-k_, size[2])][mod(j-j_, size[1])][mod(i-i_, size[0])])*demag_factor)

									}
								}
							}
						}
					}
				}
			}
		}
	}

	return t

}

// mod returns the remainder of the division of a by b
func mod(a, b int) int {
	return (a + b) % b
}
