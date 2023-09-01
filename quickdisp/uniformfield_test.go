package quickdisp

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/field"
	"github.com/will-henderson/mumax-vhf/tests"
)

func TestUniformDemag(t *testing.T) {
	defer en.InitAndClose()()

	testcases := tests.Load()
	for test_idx, s := range testcases {

		seed := 0
		rng := rand.New(rand.NewSource(int64(seed)))

		Setup(s)
		as := NewAssumeUniform()

		numTests := 10
		for i := 0; i < numTests; i++ {

			m := tests.Random3Float(rng)

			asField := as.Demag(m)

			uniformSlice := expandToSlice(m)

			demagField := cuda.NewSlice(3, en.MeshSize())
			field.SetDemagField(demagField, uniformSlice)

			var mmField [3]float32
			for c := 0; c < 3; c++ {
				mmField[c] = cuda.Sum(demagField.Comp(c))
			}

			uniformSlice.Free()
			demagField.Free()

			err := 0
			for c := 0; c < 3; c++ {
				if math.Abs(float64((mmField[c]-float32(asField[c]))/mmField[c])) > 1e-4 {
					err++
				}
			}
			if err > 0 {
				t.Errorf("%d: %d components of the fields are not equal", test_idx, err)
			}
		}
	}

}

func TestUniformUniAnis(t *testing.T) {
	defer en.InitAndClose()()

	testcases := tests.Load()
	for test_idx, s := range testcases {

		seed := 0
		rng := rand.New(rand.NewSource(int64(seed)))

		Setup(s)
		as := NewAssumeUniform()

		numTests := 10
		for i := 0; i < numTests; i++ {

			m := tests.Random3Float(rng)

			asField := as.UniAnis(m)

			uniformSlice := expandToSlice(m)
			uniAnisField := cuda.NewSlice(3, en.MeshSize())
			field.AddAnisotropyField(uniAnisField, uniformSlice)

			var mmField [3]float32
			for c := 0; c < 3; c++ {
				mmField[c] = cuda.Sum(uniAnisField.Comp(c))
			}

			uniformSlice.Free()
			uniAnisField.Free()

			err := 0
			for c := 0; c < 3; c++ {
				if math.Abs(float64((mmField[c]-float32(asField[c]))/mmField[c])) > 1e-4 {
					err++
				}
			}
			if err > 0 {
				t.Errorf("%d: %d components of the fields are not equal", test_idx, err)
			}
		}
	}

}

// return a slice living on the gpu with magnetisation m at all spatial points.
func expandToSlice(m [3]float64) *data.Slice {

	arr := make([][]float32, 3)
	for c := 0; c < 3; c++ {
		arr[c] = make([]float32, en.Mesh().NCell())
		for j := 0; j < en.Mesh().NCell(); j++ {
			arr[c][j] = float32(m[c])
		}
	}
	sl := data.SliceFromArray(arr, en.Mesh().Size())
	slGPU := cuda.NewSlice(3, en.MeshSize())
	data.Copy(slGPU, sl)

	return slGPU

}
