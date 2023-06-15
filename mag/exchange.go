package mag

import (
	en "github.com/mumax/3/engine"
)

var (
	exchangeTensor *Tensor
)

func ExchangeTensor() Tensor {
	if exchangeTensor == nil {
		return UpdateExchangeTensor()
	} else {
		return *exchangeTensor
	}
}

func UpdateExchangeTensor() Tensor {

	exchange_factor := -2. * en.Aex.GetRegion(0)

	xf := (Dy * Dz / Dx) * exchange_factor
	yf := (Dx * Dz / Dy) * exchange_factor
	zf := (Dx * Dy / Dz) * exchange_factor

	t := Zeros()

	for c := 0; c < 3; c++ {

		if Nx != 1 {
			for j := 0; j < Ny; j++ {
				for k := 0; k < Nz; k++ {
					t.SetIdx(c, c, 0, j, k, 0, j, k, -1*xf)
					t.SetIdx(c, c, Nx-1, j, k, Nx-1, j, k, -1*xf)
					for i := 1; i < Nx-1; i++ {
						t.SetIdx(c, c, i, j, k, i, j, k, -2*xf)
					}
					for i := 1; i < Nx; i++ {
						t.SetIdx(c, c, i, j, k, i-1, j, k, xf)
						t.SetIdx(c, c, i-1, j, k, i, j, k, xf)
					}
				}
			}
		}

		if Ny != 1 {
			for i := 0; i < Nx; i++ {
				for k := 0; k < Nz; k++ {
					t.AddIdx(c, c, i, 0, k, i, 0, k, -1*yf)
					t.AddIdx(c, c, i, Ny-1, k, i, Ny-1, k, -1*yf)
					for j := 1; j < Ny-1; j++ {
						t.AddIdx(c, c, i, j, k, i, j, k, -2*yf)
					}
					for j := 1; j < Ny; j++ {
						t.AddIdx(c, c, i, j, k, i, j-1, k, yf)
						t.AddIdx(c, c, i, j-1, k, i, j, k, yf)
					}
				}
			}
		}

		if Nz != 1 {
			for i := 0; i < Nx; i++ {
				for j := 0; j < Ny; j++ {
					t.AddIdx(c, c, i, j, 0, i, j, 0, -1*zf)
					t.AddIdx(c, c, i, j, Nz-1, i, j, Nz-1, -1*zf)
					for k := 1; k < Nz-1; k++ {
						t.AddIdx(c, c, i, j, k, i, j, k, -2*zf)
					}
					for k := 1; k < Nz; k++ {
						t.AddIdx(c, c, i, j, k, i, j, k-1, zf)
						t.AddIdx(c, c, i, j, k-1, i, j, k, zf)
					}
				}
			}
		}
	}

	exchangeTensor = &t
	return t

}
