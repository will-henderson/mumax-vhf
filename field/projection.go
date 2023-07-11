package field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

type RotationToZ struct {
	R [3][3]*data.Slice
}

func (rtz *RotationToZ) InitRotation() {
	m := en.M.Buffer()

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			rtz.R[c][c_] = cuda.NewSlice(1, en.MeshSize())
		}
	}

	cuda.InitRotation(m, rtz.R)
}

// it returns a two component slice living in the space perpendicular to the ground state magnetisation.
func (rtz RotationToZ) RotateMode(dst, mode *data.Slice) {
	cuda.RotateMode(dst, mode, rtz.R)
}

func (rtz RotationToZ) DerotateMode(dst, mode *data.Slice) {
	cuda.DerotateMode(dst, mode, rtz.R)
}

func (rtz RotationToZ) Free() {
	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			rtz.R[c][c_].Free()
		}
	}
}
