package quickdisp

import (
	"fmt"
	"testing"

	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/tests"
)

func TestModes(t *testing.T) {

	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, s := range testcases {

		Setup(s)
		en.Relax()

		samplePoints := ModeSample([3]float64{-1e10, -1e10, -1e10},
			[3]float64{1e10, 1e10, 1e10},
			[3]int{3, 3, 3})

		//UniformModes(samplePoints)
		w, _ := UniformModesMatrix(samplePoints)

		for i := 0; i < len(samplePoints[0]); i++ {
			for j := 0; j < len(samplePoints[1]); j++ {
				for k := 0; k < len(samplePoints[2]); k++ {
					fmt.Println(w[i])
				}
			}
		}

		_ = test_idx

	}

}
