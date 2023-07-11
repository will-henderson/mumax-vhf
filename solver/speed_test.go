package solver

import (
	"fmt"
	"testing"
	"time"

	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

var bigTest = `
SetGridSize(50, 10, 4)
SetCellSize(20e-9, 20e-9, 20e-9)

Msat  = 16074.649
Aex   = 1.3e-12

Ku1 = 756.3
AnisU = vector(0, 0, 1)

B_ext = vector(0.5, 0, 0)
`

func TestSpeed(t *testing.T) {
	defer en.InitAndClose()()
	Setup(bigTest)
	en.Relax()

	t0 := time.Now()
	Solver = new(ArnoldiField)
	_, _ = Modes()
	tAF := time.Now()

	Solver = new(RotatedToZ)
	_, _ = Modes()
	tRtZ := time.Now()

	fmt.Println("Arnoldi Field ", tAF.Sub(t0),
		", RotatedToZ", tRtZ.Sub(tAF))

}

func TestArnoldiTime(t *testing.T) {
	defer en.InitAndClose()()
	Setup(bigTest)
	en.Relax()

	aft := new(ArnoldiFieldTimes)
	_, _ = aft.Modes()

	fmt.Println("Linear Operation Time:", aft.lopTime,
		", Arpack Time ", aft.arpackTime,
		", Extracting the final eigenvectors", aft.finishingTime)

}
