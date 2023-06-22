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

	for _, p := range testcases {

		t.Run(tests.Name(p), func(t *testing.T) {

			defer en.InitAndClose()()

			p.Setup()

			en.Relax()

			want := en.GetDemagEnergy()
			got := DemagTensor().Energy()
			if tests.EqualScalars(want, got, 1e-2) > 0 {
				t.Errorf("Demag Energy was %e; want %e", got, want)
			}

			want = en.GetExchangeEnergy()
			got = ExchangeTensor().Energy()
			if tests.EqualScalars(want, got, 1e-2) > 0 {
				t.Errorf("Exchange Energy was %e; want %e", got, want)
			}

			want = en.GetAnisotropyEnergy()
			got = UniAnisTensor().Energy()
			if tests.EqualScalars(want, got, 1e-2) > 0 {
				t.Errorf("Anisotropy Energy was %e; want %e", got, want)
			}
		})
	}
}
