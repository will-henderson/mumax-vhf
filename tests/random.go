package tests

import (
	"math"

	"github.com/mumax/3/data"

	. "github.com/will-henderson/mumax-vhf/data"

	"math/rand"
)

func RandomSlice(nComp int, size [3]int, rng *rand.Rand) *data.Slice {
	//create a random tensor for testing
	rnd := data.NewSlice(nComp, size)
	rndVectors := rnd.Tensors()
	for c := 0; c < nComp; c++ {
		for k := 0; k < size[2]; k++ {
			for j := 0; j < size[1]; j++ {
				for i := 0; i < size[0]; i++ {
					rndVectors[c][k][j][i] = rng.Float32()
				}
			}
		}
	}

	return rnd
}

func RandomCSlice(nComp int, size [3]int, rng *rand.Rand) CSlice {
	//create a random tensor for testing
	rnd := NewCSliceCPU(nComp, size)
	rndRealVectors := rnd.Real().Tensors()
	rndImagVectors := rnd.Imag().Tensors()
	for c := 0; c < nComp; c++ {
		for k := 0; k < size[2]; k++ {
			for j := 0; j < size[1]; j++ {
				for i := 0; i < size[0]; i++ {
					rndRealVectors[c][k][j][i] = rng.Float32()
					rndImagVectors[c][k][j][i] = rng.Float32()
				}
			}
		}
	}

	return rnd
}

func RandomFloats(length int, rng *rand.Rand) []float32 {

	m := make([]float32, length)
	for c := 0; c < length; c++ {
		m[c] = rng.Float32() - .5
	}

	norm := float32(0.)
	for c := 0; c < length; c++ {
		norm += m[c] * m[c]
	}
	norm = float32(math.Sqrt(float64(norm)))

	for c := 0; c < length; c++ {
		m[c] /= norm
	}

	return m

}

func Random3Float(rng *rand.Rand) (m [3]float32) {

	for c := 0; c < 3; c++ {
		m[c] = rng.Float32() - .5
	}

	norm := float32(0.)
	for c := 0; c < 3; c++ {
		norm += m[c] * m[c]
	}
	norm = float32(math.Sqrt(float64(norm)))

	for c := 0; c < 3; c++ {
		m[c] /= norm
	}

	return m

}
