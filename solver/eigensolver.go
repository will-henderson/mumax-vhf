package solver

import (
	. "github.com/will-henderson/mumax-vhf/mag"
)

type EigenSolver interface {
	Solve(t Tensor) ([]float64, [][3][][][]complex128) //returns real eigenvalues and the eigenvectors
}

type eigenSolver struct{}

func Modes(solver EigenSolver) ([]float64, [][3][][][]complex128) {
	tensor := SystemTensor()
	return solver.Solve(tensor)
}
