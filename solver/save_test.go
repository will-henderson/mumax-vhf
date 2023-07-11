package solver

import (
	"testing"

	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/mag"
)

func TestSave(t *testing.T) {

	defer en.InitAndClose()()
	Setup(fmrSPTest)
	en.M.Set(en.Uniform(0, 0, 1))
	en.Relax()

	en.SaveAs(&en.M, "groundstate")

	Solver = new(RotatedToZ)
	vals, vecs := Modes()

	WriteModes(vals, vecs, "direct.out")

	mag.SelfInteractionTensor().ToCSV("direct.out/H.csv")
	mag.LinearHamiltonianTensor().ToCSV("direct.out/H0.csv")
	mag.EigenProblemTensor().ToCSV("direct.out/M.csv")

}
