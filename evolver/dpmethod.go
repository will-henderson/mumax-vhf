package evolver

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	. "github.com/will-henderson/mumax-vhf/data"
)

//utilities to extract the coefficients from the magnetisation..

func ExtractModeAmplitudes(magnetisation, groundState *data.Slice, modeProfiles []CSlice) []complex64 {

	todot := orthy(magnetisation, groundState)

	nModes := len(modeProfiles)

	amplitudes := make([]complex64, nModes)

	for i := 0; i < nModes; i++ {
		amplitudeReal := cuda.Dot(modeProfiles[i].Real(), todot)
		amplitudeImag := cuda.Dot(modeProfiles[i].Imag(), todot)
		amplitudes[i] = complex(amplitudeReal, amplitudeImag)
	}

}

func orthy(magnetisation, groundState *data.Slice) *data.Slice {
	spinDev := mag2spinDeviation(magnetisation, groundState)
	cuda.CrossProduct(spinDev, groundState, spinDev)
	return spinDev
}

func mag2spinDeviation(magnetisation, groundState *data.Slice) *data.Slice {

	gsdp := cuda.Dot(magnetisation, groundState)
	factor := float32(math.Sqrt(2 / (1 + float64(gsdp))))

	mdotgs := cuda.NewSlice(1, magnetisation.Size())
	defer mdotgs.Free()
	cuda.AddDotProduct(mdotgs, -1, magnetisation, groundState)

	result := cuda.NewSlice(3, magnetisation.Size())
	defer result.Free()
	for c := 0; c < 3; c++ {
		cuda.Mul(result.Comp(c), mdotgs, groundState.Comp(c))
	}
	cuda.Add(result, result, magnetisation)
	cuda.Scale(result, result, factor)

	return result

}
