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

// TestSIFieldReal checks that SI fields computed for real slices by mumax and via the explicit self-interaction tensor are the same.
func TestSIFieldReal(t *testing.T) {

	defer en.InitAndClose()()
	testcases := tests.Load()

	for test_idx, p := range testcases {

		//just test with random
		seed := 0
		rng := rand.New(rand.NewSource(int64(seed)))

		p.Setup()
		SITensor := mag.SelfInteractionTensor()

		numTests := 10
		for i := 0; i < numTests; i++ {

			//create a random tensor for testing
			rnd := randomSlice(rng)
			SIField_tens := mag.SIField(SITensor, rnd)

			rndGPU := cuda.NewSlice(3, en.Mesh().Size())
			data.Copy(rndGPU, rnd)

			SIField_mumax := cuda.NewSlice(3, en.Mesh().Size())

			SetDemagField(SIField_mumax, rndGPU)
			AddExchangeField(SIField_mumax, rndGPU)
			AddAnisotropyField(SIField_mumax, rndGPU)

			err := tests.EqualSlices(SIField_mumax, SIField_tens, 1e-2)
			if err > 0 {
				t.Errorf("%d: Fields are not equal: %d%% error", test_idx, 100*err/(3*SIField_mumax.Len()))
			}

			rndGPU.Free()
			SIField_mumax.Free()

		}

	}
}

func randomSlice(rng *rand.Rand) *data.Slice {
	//create a random tensor for testing
	rnd := data.NewSlice(3, en.MeshSize())
	rndVectors := rnd.Vectors()
	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					rndVectors[c][k][j][i] = rng.Float32()
				}
			}
		}
	}

	return rnd
}

// TestSIFieldComplex checks that SI fields computed for complex slices by mumax and via the explicit self-interaction tensor are the same.
func TestSIFieldComplex(t *testing.T) {
	//compare field of directly from mumax and from the exported functions.

	defer en.InitAndClose()()
	testcases := tests.Load()

	for test_idx, p := range testcases {

		//just test with random vectors.
		seed := 0
		rng := rand.New(rand.NewSource(int64(seed)))

		p.Setup()
		SITensor := mag.SelfInteractionTensor()

		numTests := 1
		for i := 0; i < numTests; i++ {

			//create a random tensor for testing
			rnd := randomCSlice(rng)
			SIField_tens := mag.SIFieldComplex(SITensor, rnd)

			rndGPU := rnd.DevCopy()

			SIField_mumax := NewCSlice(3, en.Mesh().Size())
			SetSIFieldComplex(SIField_mumax, rndGPU)

			err := tests.EqualCSlices(SIField_mumax, SIField_tens, 1e-3)
			if err > 0 {
				t.Errorf("%d: Fields are not equal: %d%% error", test_idx, 100*err/(3*SIField_mumax.Len()))
			}

			rndGPU.Free()
			SIField_mumax.Free()

		}

	}
}

func randomCSlice(rng *rand.Rand) CSlice {
	//create a random tensor for testing
	rnd := NewCSliceCPU(3, en.MeshSize())
	rndRealVectors := rnd.Real().Vectors()
	rndImagVectors := rnd.Imag().Vectors()
	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					rndRealVectors[c][k][j][i] = rng.Float32()
					rndImagVectors[c][k][j][i] = rng.Float32()
				}
			}
		}
	}

	return rnd
}
