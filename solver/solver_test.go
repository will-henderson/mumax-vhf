package solver

import (
	"fmt"
	"testing"
	"time"

	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/tests"
)

func TestStraight(t *testing.T) {
	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, s := range testcases {

		Setup(s)

		en.Relax()

		startA := time.Now()
		Solver = new(StraightGonum)
		valsA, vecsA := Modes()

		endA := time.Now()

		Solver = new(Straight)
		valsB, vecsB := Modes()

		endB := time.Now()

		fmt.Println(endA.Sub(startA), endB.Sub(endA))

		err := tests.EqualDecompositions(vecsA, vecsB, valsA, valsB, 1e-5, 1e-3)

		//for i, v := range valsA {
		//	fmt.Println(v, valsB[i])
		//}

		if err > 0 {
			t.Errorf("%d: Decompositions are not equal: %d%% error", test_idx, 100*err/len(valsA))
		}

	}
}

func TestRotatedToZ(t *testing.T) {
	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, s := range testcases {

		Setup(s)

		en.Relax()

		startA := time.Now()
		Solver = new(Straight)
		valsA, vecsA := Modes()

		endA := time.Now()

		Solver = new(RotatedToZ)
		valsB, vecsB := Modes()

		endB := time.Now()

		fmt.Println(endA.Sub(startA), endB.Sub(endA))

		err := tests.EqualDecompositions(vecsA, vecsB, valsA, valsB, 1e-4, 1e-3)

		//for i, v := range valsA {
		//	fmt.Println(v, valsB[i])
		//}

		if err > 0 {
			t.Errorf("%d: Decompositions are not equal: %d%% error", test_idx, 100*err/len(valsA))
		}

	}
}
