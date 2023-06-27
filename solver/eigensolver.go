// Package solver contains structs with a Modes function to calculate the eigenmodes and corresponding eigenfrequencies of the system.
package solver

// EigenSolver is an interface which wraps the Modes method.
// Modes returns the eigenfrequencies and corresponding eigenmodes of the system.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
// and assumes that the ground state magnetisation is currently stored in en.M.
// It returns 2 * Nx * Ny * Nz eigenpairs (zero eigenfrequencies are ignored)
type EigenSolver interface {
	Modes() ([]float64, [][3][][][]complex128) //returns real eigenvalues and the eigenvectors
}

// Types implementing the EigenSolver interface should extend the eigenSolver struct.
// This will allow convience methods to be added to all solvers in the future.
type eigenSolver struct{}

var (
	solver EigenSolver
)

var (
	eigenmodes       [][3][][][]complex128
	eigenfrequencies []float64
)

func Modes() ([]float64, [][3][][][]complex128) {
	if eigenfrequencies == nil {
		eigenfrequencies, eigenmodes = solver.Modes()
	}
	return eigenfrequencies, eigenmodes
}
