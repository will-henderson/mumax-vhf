package solver

import (
	"math"
	"math/rand"

	cblas "gonum.org/v1/gonum/blas/cblas128"
)

type LanczosMatrix struct {
	eigenSolver
	R [][][][3][3]float64
}

func tridiagonalMatrix(H cblas.Hermitian, m int) {

	n := 4

	Td := make([]complex128, m)
	Tu := make([]float64, m-1)

	//Define the Lanczos vectors
	V := make([]cblas.Vector, n)

	//Generate a random unit vector V[0]
	V[0] = cblas.Vector{n, 1, make([]complex128, n)}
	for j := 0; j < n; j++ {
		V[0].Data[j] = complex(rand.Float64(), rand.Float64())
	}
	normalise(V[0])

	//Initial iteration, u initially corresponds to w but is overwritten
	u := cblas.Vector{n, 1, make([]complex128, n)}

	//perform the matrix vector multiplication H * V[0] and store the result in u
	cblas.Hemv(1, H, V[0], 0, u)

	α := cblas.Dotc(u, V[0])
	Td[0] = α

	//I am overwriting the w value is overwritten with u here!
	cblas.Axpy(-α, V[0], u)

	//other iterations
	for j := 1; j < m; j++ {
		β := math.Sqrt(real(cblas.Dotc(u, u)))
		Tu[j-1] = β
		cblas.Dscal(1/β, u) //so u now represents the value of V[j]
		V[j] = u
		u = cblas.Vector{n, 1, make([]complex128, n)}
		cblas.Hemv(1, H, V[j], 0, u)
		α = cblas.Dotc(u, V[0])
		Td[j] = α
		cblas.Axpy(-α, V[j], u)
		cblas.Axpy(complex(-β, 0), V[j-1], u)

	}

}

func normalise(v cblas.Vector) {
	magnitude := math.Sqrt(real(cblas.Dotc(v, v)))
	cblas.Dscal(1/magnitude, v)
}
