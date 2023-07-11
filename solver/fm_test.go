package solver

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"
)

var fmrSPTest = `
SetGridSize(24, 24, 2)
SetCellSize(5e-9, 5e-9, 5e-9)

Msat  = 800e3
Aex   = 1.3e-11
B_ext = vector(0.076360, 0.0545976, 0)
`

func TestFM(t *testing.T) {
	defer en.InitAndClose()()
	Setup(fmrSPTest)
	en.M.Set(en.Uniform(0, 0, 1))
	en.Relax()

	//now compute the actual vectors for comparison.
	Solver = new(RotatedToZ)
	vals, vecs := Modes()

	WriteModes(vals, vecs, "direct.out")

	en.SaveAs(&en.M, "gs")

	//en.Alpha.Set(0.008)
	en.Alpha.Set(0)

	timesteps := 4000
	δ := 5e-12

	//perturb the magnetisation.
	m := data.NewSlice(3, en.MeshSize())
	en.M.EvalTo(m)
	seed := 0
	rng := rand.New(rand.NewSource(int64(seed)))
	vectors := m.Tensors()
	for c := 0; c < 3; c++ {
		for k := 0; k < en.MeshSize()[2]; k++ {
			for j := 0; j < en.MeshSize()[1]; j++ {
				for i := 0; i < en.MeshSize()[0]; i++ {
					vectors[c][k][j][i] += (rng.Float32() - .5) * .01
				}
			}
		}
	}
	en.M.SetArray(m)

	//the standard problem proposal paper:
	//en.B_ext.Set(data.Vector{.076696, 0.053687, 0})

	for i := 0; i < timesteps; i++ {
		fmt.Println(i)
		en.Run(δ)
		en.Save(&en.M)
	}

}
