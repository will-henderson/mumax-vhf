package solver

type tridiagonal struct {
	Td, To []complex64
}

/*
func tridiagonalField(m int) ([]CSlice, tridiagonal) {

	V := make([]CSlice, Nx*Ny*Nz*2) //this dimension because we enforce only working with vectors perpendicular to the ground state
	Td := make([]complex64, m)
	To := make([]complex64, m-1)

	operator := field.NewLinearEvolution()

	//generate a random initial slice and enforce that it is perpendicular to the ground state
	vPrev := RandomPerpendicular(en.M.Buffer())

	V[0] = NewCSliceCPU(3, en.Mesh().Size())
	Copy(V[0], vPrev)

	u := operator.Operate(vPrev)
	// α is real.
	α := Dotc(u, vPrev)
	CMadd2(u, u, vPrev, -α, 1)

	Td[0] = α

	v := NewCSlice(3, en.Mesh().Size())

	// other iterations
	for j := 1; j < m; j++ {
		β := float32(math.Sqrt(float64(real(Dotc(u, u)))))
		SScal(v, u, 1/β)

		V[j] = NewCSliceCPU(3, en.Mesh().Size())
		Copy(V[j], v)

		u = operator.Operate(v)
		α = Dotc(u, v)

		SMadd2(u, u, vPrev, 1, -β)
		CMadd2(u, u, v, 1, -α)

		temp := vPrev
		vPrev = v
		v = temp

		Td[j] = α
		To[j-1] = complex(β, 0)

	}

	return V, tridiagonal{Td, To}
}
*/
