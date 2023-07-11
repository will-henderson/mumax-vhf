package mag

import (
	"github.com/mumax/3/cuda"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

// UniAnisTensor returns the self-interaction tensor for the uniaxial anisotropy interaction.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// This forces the updating of the tensor, rather than potentially returning a cached value.
func UniAnisTensor() Tensor {

	Ku1GPU, rM := en.Ku1.Slice()
	Ku1 := Ku1GPU.HostCopy().Scalars()
	if rM {
		cuda.Recycle(Ku1GPU)
	}

	AnisUGPU, rM := en.AnisU.Slice()
	AnisU := AnisUGPU.HostCopy().Vectors()
	if rM {
		cuda.Recycle(AnisUGPU)
	}

	t := ZeroTensor(3, en.MeshSize())
	Nx := en.MeshSize()[0]
	Ny := en.MeshSize()[1]
	Nz := en.MeshSize()[2]

	for k := 0; k < Nz; k++ {
		for j := 0; j < Ny; j++ {
			for i := 0; i < Nx; i++ {
				for c := 0; c < 3; c++ {
					for c_ := 0; c_ < 3; c_++ {
						t.SetIdx(c, c_, i, j, k, i, j, k, float64(-2*Ku1[k][j][i]*AnisU[c][k][j][i]*AnisU[c_][k][j][i]))
					}
				}
			}
		}
	}

	return t

}
