package mag

import (
	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/data"
)

// ExchangeTensor returns the self-interaction tensor for the Exchange interaction.
// Note that it does not have any inputs. Rather it uses the geometry defined by the global variables.
func ExchangeTensor() Tensor {

	Nx := en.MeshSize()[0]
	Ny := en.MeshSize()[1]
	Nz := en.MeshSize()[2]
	Dx := en.Mesh().CellSize()[0]
	Dy := en.Mesh().CellSize()[1]
	Dz := en.Mesh().CellSize()[2]

	var prev, next float64

	t := ZeroTensor(3, en.MeshSize())

	xf := -2. * (1 / (Dx * Dx))
	for k := 0; k < Nz; k++ {
		for j := 0; j < Ny; j++ {
			// i = 0 has no left neighbour.
			next = xf * float64(en.ExchangeAtCell(0, j, k, 1, j, k))
			for c := 0; c < 3; c++ {
				t.SetIdx(c, c, 0, j, k, 0, j, k, -next)
				t.SetIdx(c, c, 0, j, k, 1, j, k, next)
			}
			for i := 1; i < Nx-1; i++ {
				next = xf * float64(en.ExchangeAtCell(i, j, k, i+1, j, k))
				prev = xf * float64(en.ExchangeAtCell(i, j, k, i-1, j, k))
				for c := 0; c < 3; c++ {
					t.SetIdx(c, c, i, j, k, i, j, k, -prev-next)
					t.SetIdx(c, c, i, j, k, i-1, j, k, prev)
					t.SetIdx(c, c, i, j, k, i+1, j, k, next)
				}
			}
			// i = Nx - 1 has no right neighbour
			prev = xf * float64(en.ExchangeAtCell(Nx-1, j, k, Nx-2, j, k))
			for c := 0; c < 3; c++ {
				t.SetIdx(c, c, Nx-1, j, k, Nx-1, j, k, -prev)
				t.SetIdx(c, c, Nx-1, j, k, Nx-2, j, k, prev)
			}
		}
	}

	yf := -2. * (1 / (Dy * Dy))
	for k := 0; k < Nz; k++ {
		for i := 0; i < Nx; i++ {
			// j = 0 has no back neighbour.
			next = yf * float64(en.ExchangeAtCell(i, 0, k, i, 1, k))
			for c := 0; c < 3; c++ {
				t.AddIdx(c, c, i, 0, k, i, 0, k, -next)
				t.SetIdx(c, c, i, 0, k, i, 1, k, next)
			}
			for j := 1; j < Ny-1; j++ {
				next = yf * float64(en.ExchangeAtCell(i, j, k, i, j+1, k))
				prev = yf * float64(en.ExchangeAtCell(i, j, k, i, j-1, k))
				for c := 0; c < 3; c++ {
					t.AddIdx(c, c, i, j, k, i, j, k, -prev-next)
					t.SetIdx(c, c, i, j, k, i, j-1, k, prev)
					t.SetIdx(c, c, i, j, k, i, j+1, k, next)
				}
			}
			// j = Ny - 1 has no front neighbour
			prev = yf * float64(en.ExchangeAtCell(i, Ny-1, k, i, Ny-2, k))
			for c := 0; c < 3; c++ {
				t.AddIdx(c, c, i, Ny-1, k, i, Ny-1, k, -prev)
				t.SetIdx(c, c, i, Ny-1, k, i, Ny-2, k, prev)
			}
		}
	}

	if Nz > 1 {
		zf := -2. * (1 / (Dz * Dz))
		for j := 0; j < Ny; j++ {
			for i := 0; i < Nx; i++ {
				// y = 0 has no bottom neighbour.
				next = zf * float64(en.ExchangeAtCell(i, j, 0, i, j, 1))
				for c := 0; c < 3; c++ {
					t.AddIdx(c, c, i, j, 0, i, j, 0, -next)
					t.SetIdx(c, c, i, j, 0, i, j, 1, next)
				}
				for k := 1; k < Nz-1; k++ {
					next = zf * float64(en.ExchangeAtCell(i, j, k, i, j, k+1))
					prev = zf * float64(en.ExchangeAtCell(i, j, k, i, j, k-1))
					for c := 0; c < 3; c++ {
						t.AddIdx(c, c, i, j, k, i, j, k, -prev-next)
						t.SetIdx(c, c, i, j, k, i, j, k-1, prev)
						t.SetIdx(c, c, i, j, k, i, j, k+1, next)
					}
				}
				// k = Nz - 1 has no top neighbour
				prev = zf * float64(en.ExchangeAtCell(i, j, Nz-1, i, j, Nz-2))
				for c := 0; c < 3; c++ {
					t.AddIdx(c, c, i, j, Nz-1, i, j, Nz-1, -prev)
					t.SetIdx(c, c, i, j, Nz-1, i, j, Nz-2, prev)
				}
			}
		}
	}

	return t

}
