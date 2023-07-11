package solver

import (
	"fmt"
	"testing"
	"time"

	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/tests"
)

func TestNormality(t *testing.T) {
	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, s := range testcases {

		Setup(s)
		en.Relax()

		Solver = new(ArnoldiField)
		_, vecs := Modes()

		err := tests.Normality(vecs, 1e-5)
		if err > 0 {
			t.Errorf("%d: The returned eigenvectors do not have norm equal to one: %d%% error", test_idx, 100*err/len(vecs))
		}
	}
}

func TestArnoldiUnrotated(t *testing.T) {
	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, s := range testcases {
		Setup(s)
		en.Relax()

		Solver = new(RotatedToZ)
		valsA, vecsA := Modes()

		Solver = new(ArnoldiFieldUnrotated)
		valsB, vecsB := Modes()

		err := tests.EqualSubDecompositions(vecsA, vecsB, valsA, valsB, 1e-5, 1e-3)

		//for i, v := range valsB {
		//		fmt.Println(v, valsA[i])
		//}

		if err > 0 {
			t.Errorf("%d: Decompositions are not equal: %d%% error", test_idx, 100*err/len(valsA))
		}
	}
}

func TestArnoldiField(t *testing.T) {
	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, s := range testcases {
		Setup(s)
		en.Relax()

		startA := time.Now()
		Solver = new(RotatedToZ)
		valsA, vecsA := Modes()
		endA := time.Now()

		Solver = new(ArnoldiField)
		valsB, vecsB := Modes()
		endB := time.Now()

		fmt.Println(endA.Sub(startA), endB.Sub(endA))

		err := tests.EqualSubDecompositions(vecsA, vecsB, valsA, valsB, 1e-3, 1e-3)

		//for i, v := range valsB {
		//	fmt.Println(v, valsA[i])
		//}

		if err > 0 {
			t.Errorf("%d: Decompositions are not equal: %d%% error", test_idx, 100*err/len(valsA))
		}
	}
}
