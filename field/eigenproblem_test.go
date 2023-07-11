package field

import (
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"
	"github.com/will-henderson/mumax-vhf/tests"

	"math/rand"
	"testing"
)

// TestEigenMatrix checks that the operation of the eigenmatrix that is solved to find the harmonic modes of the system
// is the same for the explicit calculation of magnetisation and for that obtained via mumax effective fields
func TestEigenMatrix(t *testing.T) {

	defer en.InitAndClose()()
	testcases := tests.Load()

	for test_idx, s := range testcases {

		seed := 0
		rng := rand.New(rand.NewSource(int64(seed)))

		Setup(s)

		en.Relax()

		EigenProblem := mag.EigenProblemTensor()

		numTests := 10
		for i := 0; i < numTests; i++ {
			rnd := tests.RandomCSlice(3, en.MeshSize(), rng)
			ep_tens := EigenProblem.ITCSP(rnd)

			rndGPU := rnd.DevCopy()

			le := NewLinearEvolution()
			ep_mumax := NewCSlice(3, en.MeshSize())
			le.OperateComplex(&ep_mumax, rndGPU)

			err := tests.EqualCSlices(ep_mumax, ep_tens, 1e-3)
			if err > 0 {
				t.Errorf("%d: Fields are not equal: %d%% error", test_idx, 100*err/(3*ep_mumax.Len()))
			}

			rndGPU.Free()
			ep_mumax.Free()

		}

	}

}
