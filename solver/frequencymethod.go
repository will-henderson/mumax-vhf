package solver

import (
	en "github.com/mumax/3/engine"
)

type FrequencyMethod struct {
	eigenSolver
}

func (fm FrequencyMethod) evolve(δ float64, timesteps int) {

	//set alpha equal to zero as we don't want to damp.
	en.Alpha.Set(0)

	//I think just evolving normally is the same as acting with my matrix.

	// need to slightly perturb the magnetisation to start.

	for i := 0; i < timesteps; i++ {
		en.Run(δ)
		en.Save(&en.M)
	}

}
