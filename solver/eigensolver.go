package solver

type EigenSolver interface {
	Modes() ([]float64, [][3][][][]complex128) //returns real eigenvalues and the eigenvectors
}

type eigenSolver struct{}
