package field

import (
	"math/rand"
	"testing"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"
	"github.com/will-henderson/mumax-vhf/tests"
)

func TestProjection(t *testing.T) {

	defer en.InitAndClose()()
	testcases := tests.Load()

	for test_idx, s := range testcases {

		seed := 0
		rng := rand.New(rand.NewSource(int64(seed)))

		Setup(s)
		en.Relax()

		rotCPU := new(mag.RotationToZ)
		rotCPU.InitRotation()

		rotGPU := new(RotationToZ)
		rotGPU.InitRotation()

		numTests := 10
		for i := 0; i < numTests; i++ {
			rnd := tests.RandomSlice(2, en.MeshSize(), rng)

			rndGPU := cuda.NewSlice(2, en.Mesh().Size())
			data.Copy(rndGPU, rnd)

			derotatedCPU := rotCPU.DerotateModeReal(rnd)
			derotatedGPU := cuda.NewSlice(3, en.MeshSize())
			rotGPU.DerotateMode(derotatedGPU, rndGPU)

			err := tests.EqualSlices(derotatedCPU, derotatedGPU, 1e-3)
			if err > 0 {
				t.Errorf("%d: Derotated Fields are not equal: %d%% error", test_idx, 100*err/(3*derotatedCPU.Len()))
			}

			rotatedCPU := rotCPU.RotateModeReal(derotatedCPU)
			rotatedGPU := cuda.NewSlice(2, en.MeshSize())
			rotGPU.RotateMode(rotatedGPU, derotatedGPU)

			err = tests.EqualSlices(rotatedCPU, rotatedGPU, 1e-3)
			if err > 0 {
				t.Errorf("%d: Rotated Fields are not equal: %d%% error", test_idx, 100*err/(3*rotatedCPU.Len()))
			}

			rotatedGPU.Free()
			derotatedGPU.Free()
			rndGPU.Free()

		}

		rotGPU.Free()

	}

}
