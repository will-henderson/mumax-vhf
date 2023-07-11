package field

import (
	"testing"

	cuda "github.com/mumax/3/cuda"
	data "github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
	. "github.com/will-henderson/mumax-vhf/data"

	"github.com/will-henderson/mumax-vhf/tests"
)

// TestDemagField checks that the demag field computed corresponds to that directly from Mumax.
// It also checks that the magnetisation is not changed during this computation.
func TestDemagField(t *testing.T) {
	//compare field of directly from mumax and from the exported functions.
	defer en.InitAndClose()()

	testcases := tests.Load()

	for _, s := range testcases {

		Setup(s)

		en.M.Set(en.RandomMagSeed(0))

		//create a slice on GPU to assign field to
		BMumax := cuda.NewSlice(3, en.Mesh().Size())
		en.SetDemagField(BMumax)

		//copy the old M, then set magnetisation to something else to test
		MCopy := cuda.NewSlice(3, en.Mesh().Size())
		data.Copy(MCopy, en.M.Buffer())

		en.M.Set(en.RandomMagSeed(1))
		MNew := en.M.Buffer()

		B := cuda.NewSlice(3, en.Mesh().Size())
		SetDemagField(B, MCopy)

		if tests.EqualSlices(B, BMumax, 1e-3) > 0 {
			t.Error("Demag Fields are not Equal")
		}

		if tests.EqualSlices(MNew, en.M.Buffer(), 1e-3) > 0 {
			t.Error("Magnetisation has been changed")
		}

		B.Free()
		BMumax.Free()
		MCopy.Free()
	}
}

// TestExchangeField checks that the exchange field computed corresponds to that directly from Mumax.
// It also checks that the magnetisation is not changed during this computation.
func TestExchangeField(t *testing.T) {
	//compare field of directly from mumax and from the exported functions.

	defer en.InitAndClose()()
	testcases := tests.Load()

	for _, s := range testcases {

		Setup(s)

		en.M.Set(en.RandomMagSeed(0))

		//create a slice on GPU to assign field to
		BMumax := cuda.NewSlice(3, en.Mesh().Size())
		cuda.Zero(BMumax)
		en.AddExchangeField(BMumax)

		//copy the old M, then set magnetisation to something else to test
		MCopy := cuda.NewSlice(3, en.Mesh().Size())
		data.Copy(MCopy, en.M.Buffer())

		en.M.Set(en.RandomMagSeed(1))
		MNew := en.M.Buffer()

		B := cuda.NewSlice(3, en.Mesh().Size())
		cuda.Zero(B)
		AddExchangeField(B, MCopy)

		if tests.EqualSlices(B, BMumax, 1e-3) > 0 {
			t.Error("Exchange Fields are not Equal")
		}

		if tests.EqualSlices(MNew, en.M.Buffer(), 1e-3) > 0 {
			t.Error("Magnetisation has been changed")
		}

		BMumax.Free()
		B.Free()
		MCopy.Free()

	}
}

// TestAnisotropyField checks that the anisotropy field computed corresponds to that directly from Mumax.
// It also checks that the magnetisation is not changed during this computation.
func TestAnisotropyField(t *testing.T) {

	defer en.InitAndClose()()
	testcases := tests.Load()

	for _, s := range testcases {

		Setup(s)

		en.M.Set(en.RandomMagSeed(0))

		//create a slice on GPU to assign field to
		BMumax := cuda.NewSlice(3, en.Mesh().Size())
		cuda.Zero(BMumax)
		en.AddAnisotropyField(BMumax)

		//copy the old M, then set magnetisation to something else to test
		MCopy := cuda.NewSlice(3, en.Mesh().Size())
		data.Copy(MCopy, en.M.Buffer())

		en.M.Set(en.RandomMagSeed(1))
		MNew := en.M.Buffer()

		B := cuda.NewSlice(3, en.Mesh().Size())
		cuda.Zero(B)
		AddAnisotropyField(B, MCopy)

		if tests.EqualSlices(B, BMumax, 1e-3) > 0 {
			t.Error("Demag Fields are not Equal")
		}

		if tests.EqualSlices(MNew, en.M.Buffer(), 1e-3) > 0 {
			t.Error("Magnetisation has been changed")
		}

		B.Free()
		BMumax.Free()
		MCopy.Free()

	}
}
