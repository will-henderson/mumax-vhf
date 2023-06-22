package field

import (
	"testing"

	cuda "github.com/mumax/3/cuda"
	data "github.com/mumax/3/data"
	en "github.com/mumax/3/engine"

	"github.com/will-henderson/mumax-vhf/tests"
)

// TestDemagField checks that the demag field computed corresponds to that directly from Mumax.
// It also checks that the magnetisation is not changed during this computation.
func TestDemagField(t *testing.T) {
	//compare field of directly from mumax and from the exported functions.

	testcases := tests.Load()

	for _, p := range testcases {
		t.Run(tests.Name(p), func(t *testing.T) {

			defer en.InitAndClose()()
			p.Setup()

			en.M.Set(en.RandomMagSeed(0))

			//create a slice on GPU to assign field to
			BMumaxGPU := cuda.NewSlice(3, en.Mesh().Size())
			en.SetDemagField(BMumaxGPU)
			BMumax := BMumaxGPU.HostCopy().Vectors()
			BMumaxGPU.Free()

			//copy the old M, then set magnetisation to something else to test
			MCopy := cuda.NewSlice(3, en.Mesh().Size())
			data.Copy(MCopy, en.M.Buffer())

			en.M.Set(en.RandomMagSeed(1))
			MNew := en.M.Buffer()

			BGPU := cuda.NewSlice(3, en.Mesh().Size())
			SetDemagField(MCopy, BGPU)
			B := BGPU.HostCopy().Vectors()
			BGPU.Free()

			if tests.EqualFields(B, BMumax, 1e-3) > 0 {
				t.Error("Demag Fields are not Equal")
			}

			MWant := MNew.HostCopy().Vectors()
			MGot := en.M.Buffer().HostCopy().Vectors()

			if tests.EqualFields(MWant, MGot, 1e-3) > 0 {
				t.Error("Magnetisation has been changed")
			}

		})

	}
}

// TestExchangeField checks that the exchange field computed corresponds to that directly from Mumax.
// It also checks that the magnetisation is not changed during this computation.
func TestExchangeField(t *testing.T) {
	//compare field of directly from mumax and from the exported functions.

	testcases := tests.Load()

	for _, p := range testcases {
		t.Run(tests.Name(p), func(t *testing.T) {

			defer en.InitAndClose()()
			p.Setup()

			en.M.Set(en.RandomMagSeed(0))

			//create a slice on GPU to assign field to
			BMumaxGPU := cuda.NewSlice(3, en.Mesh().Size())
			cuda.Zero(BMumaxGPU)
			en.AddExchangeField(BMumaxGPU)
			BMumax := BMumaxGPU.HostCopy().Vectors()
			BMumaxGPU.Free()

			//copy the old M, then set magnetisation to something else to test
			MCopy := cuda.NewSlice(3, en.Mesh().Size())
			data.Copy(MCopy, en.M.Buffer())

			en.M.Set(en.RandomMagSeed(1))
			MNew := en.M.Buffer()

			BGPU := cuda.NewSlice(3, en.Mesh().Size())
			cuda.Zero(BGPU)
			AddExchangeField(MCopy, BGPU)
			B := BGPU.HostCopy().Vectors()
			BGPU.Free()

			if tests.EqualFields(B, BMumax, 1e-3) > 0 {
				t.Error("Exchange Fields are not Equal")
			}

			MWant := MNew.HostCopy().Vectors()
			MGot := en.M.Buffer().HostCopy().Vectors()

			if tests.EqualFields(MWant, MGot, 1e-3) > 0 {
				t.Error("Magnetisation has been changed")
			}

		})

	}
}

// TestAnisotropyField checks that the anisotropy field computed corresponds to that directly from Mumax.
// It also checks that the magnetisation is not changed during this computation.
func TestAnisotropyField(t *testing.T) {
	//compare field of directly from mumax and from the exported functions.

	testcases := tests.Load()

	for _, p := range testcases {
		t.Run(tests.Name(p), func(t *testing.T) {

			defer en.InitAndClose()()
			p.Setup()

			en.M.Set(en.RandomMagSeed(0))

			//create a slice on GPU to assign field to
			BMumaxGPU := cuda.NewSlice(3, en.Mesh().Size())
			cuda.Zero(BMumaxGPU)
			en.AddAnisotropyField(BMumaxGPU)
			BMumax := BMumaxGPU.HostCopy().Vectors()
			BMumaxGPU.Free()

			//copy the old M, then set magnetisation to something else to test
			MCopy := cuda.NewSlice(3, en.Mesh().Size())
			data.Copy(MCopy, en.M.Buffer())

			en.M.Set(en.RandomMagSeed(1))
			MNew := en.M.Buffer()

			BGPU := cuda.NewSlice(3, en.Mesh().Size())
			cuda.Zero(BGPU)
			AddAnisotropyField(MCopy, BGPU)
			B := BGPU.HostCopy().Vectors()
			BGPU.Free()

			if tests.EqualFields(B, BMumax, 1e-3) > 0 {
				t.Error("Demag Fields are not Equal")
			}

			MWant := MNew.HostCopy().Vectors()
			MGot := en.M.Buffer().HostCopy().Vectors()

			if tests.EqualFields(MWant, MGot, 1e-3) > 0 {
				t.Error("Magnetisation has been changed")
			}

		})

	}
}
