package quickdisp

import (
	"fmt"
	"math"
	"testing"

	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/tests"
)

func TestUniformRelax(t *testing.T) {

	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, s := range testcases {

		Setup(s)

		m0 := UniformGroundState(AVERAGE_FIELD)
		mmm := UniformGroundState(AVERAGE_MAGNETISATION)
		//average over all space.
		fmt.Println(test_idx, m0, mmm)

		var diffnorm float32
		for c := 0; c < 3; c++ {
			diff := (m0[c] - mmm[c])
			diffnorm += diff * diff
		}
		if math.Sqrt(float64(diffnorm)) > 1e-4 {
			t.Errorf("%d: Components of the ground state magnetisations not equal", test_idx)
		}

	}
}
