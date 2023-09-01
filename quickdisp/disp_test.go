package quickdisp

import (
	"fmt"
	"os"
	"testing"

	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

var testmx3sm = `
SetGridSize(64, 16, 4)
SetCellSize(40e-9, 40e-9, 10e-9)

Msat = 16074.649
Aex = 1.3e-12
Ku1 = 0.0202 * 0.0941 / (2 * mu0)
AnisU = vector(0, 0, 1)
B_ext = vector(0, 0.08576, 0)
`

func TestNumericalDispersion(t *testing.T) {

	os.Mkdir("plots.out", os.ModePerm)
	defer en.InitAndClose()()

	Setup(testmx3sm)
	en.Relax()

	//UniformModes(samplePoints)
	direction := [3]float64{1, 0, 0}
	ks := AlongDirection(direction)
	fmt.Println(len(ks[0]))

	magnitudes, frequencies := NumericalDispersion(1e11, 5e7, ks)

	outname := fmt.Sprintf("plots.out/fmr.dat")
	DispImageDat(ks, frequencies, magnitudes[2], direction, outname)
	GnuplotColorMap(outname)

}
