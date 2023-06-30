package mag

import (
	"testing"

	en "github.com/mumax/3/engine"

	"github.com/will-henderson/mumax-vhf/tests"
)

// TestEnergy tests equality between energies calculated by mumax and via the self-interaction tensor
// for the Demag, Exchange and Uniaxial Anisotropy energies.
// The energies are calculated for the ground state.
func TestEnergy(t *testing.T) {

	testcases := tests.Load()
	defer en.InitAndClose()()

	for test_idx, p := range testcases {

		p.Setup()

		en.Relax()

		want := en.GetDemagEnergy()
		got := Energy(DemagTensor(), en.M.Buffer())
		if tests.EqualScalars(want, got, 1e-2) > 0 {
			t.Errorf("%d: Demag Energy was %e; want %e", test_idx, got, want)
		}

		want = en.GetExchangeEnergy()
		got = Energy(ExchangeTensor(), en.M.Buffer())
		if tests.EqualScalars(want, got, 1e-2) > 0 {
			t.Errorf("%d: Exchange Energy was %e; want %e", test_idx, got, want)
		}

		want = en.GetAnisotropyEnergy()
		got = Energy(UniAnisTensor(), en.M.Buffer())
		if tests.EqualScalars(want, got, 1e-2) > 0 {
			t.Errorf("%d: Anisotropy Energy was %e; want %e", test_idx, got, want)
		}
	}
}
