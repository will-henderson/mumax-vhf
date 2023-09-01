package quickdisp

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	en "github.com/mumax/3/engine"
)

var (
	DisturbanceSTD float32 = .003
)

func NumericalDispersion(freqMax, freqStep float64, ks [3][]int32) ([3][][]float32, []float64) {

	en.Relax() //might have been relaxed before, but might as well do this again here.
	en.Alpha.Set(0)
	DisturbMagnetisation()

	δ := 1 / (2 * freqMax)
	timeSteps := int(math.Ceil(1 / (freqStep * δ)))

	//we will fourier transform before transferring to ram and then pick out the points which
	//correspond to what we want as, selected by the inputs.
	//we assume that we have selected sufficiently few that we can keep the ones we are interested in
	//in gpu memory.

	dispFFT := cuda.NewDispersionFFT(en.MeshSize(), ks, timeSteps)

	for i := 0; i < timeSteps; i++ {
		en.Run(δ)
		//en.Save(en.M.Comp(0))

		//we then fourier transform to find the k components.
		dispFFT.FFT(en.M.Buffer())
	}

	return dispFFT.TimeFFT(), fftFreq(timeSteps, δ)
}

func DisturbMagnetisation() {

	generator := curand.CreateGenerator(curand.PSEUDO_DEFAULT)
	generator.SetSeed(0)

	perturbation := cuda.NewSlice(3, en.MeshSize())
	for c := 0; c < 3; c++ {
		generator.GenerateNormal(uintptr(perturbation.DevPtr(c)), int64(en.Mesh().NCell()), 0, DisturbanceSTD)
	}

	cuda.Add(en.M.Buffer(), en.M.Buffer(), perturbation)
	cuda.Normalize(en.M.Buffer(), en.Geometry().Gpu())

}

func fftFreq(timeSteps int, δ float64) []float64 {
	frequencies := make([]float64, timeSteps)

	var limPositive int
	if timeSteps%2 == 0 {
		limPositive = timeSteps / 2
	} else {
		limPositive = (timeSteps + 1) / 2
	}

	denom := 1 / (float64(timeSteps) * δ)
	for i := 0; i < limPositive; i++ {
		frequencies[i] = float64(i) * denom
	}
	for i := limPositive; i < timeSteps; i++ {
		frequencies[i] = float64(i-timeSteps) * denom
	}

	return frequencies
}
