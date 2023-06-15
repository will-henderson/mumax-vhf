package mag

import (
	"testing"

	en "github.com/mumax/3/engine"
)

func TestEnergy(t *testing.T) {

	var tests = []SystemParameters{
		{10, 10, 10,
			20e-9, 20e-9, 20e-9,
			16074.649, 1.3e-12, 756.3,
			[3]float64{0, 0, 1},
			[3]float64{.5, 0, 0},
		},
	}

	for _, test := range tests {
		want, got := runner(test)

		frac := want[0] / got[0]
		if frac > 1.01 || frac < .99 {
			t.Errorf("Demag Energy was %e; want %e", got[0], want[0])
		}

		frac = want[1] / got[1]
		if frac > 1.01 || frac < .99 {
			t.Errorf("Exchange Energy was %e; want %e", got[1], want[1])
		}

		frac = want[2] / got[2]
		if frac > 1.01 || frac < .99 {
			t.Errorf("Demag Energy was %e; want %e", got[2], want[2])
		}
	}

}

func runner(p SystemParameters) ([3]float64, [3]float64) {
	defer en.InitAndClose()()

	p.Setup()

	en.Relax()

	want := [3]float64{en.GetDemagEnergy(), en.GetExchangeEnergy(), en.GetAnisotropyEnergy()}
	got := [3]float64{DemagTensor().Energy(), ExchangeTensor().Energy(), UniAnisTensor().Energy()}

	return want, got
}
