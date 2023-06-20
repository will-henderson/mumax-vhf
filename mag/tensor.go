package mag

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	en "github.com/mumax/3/engine"
)

var ( //need to actaully initialise this at some point
	Nx, Ny, Nz int
	Dx, Dy, Dz float64
)

type Tensor struct {
	N [3][3][][][][][][]float64
}

func (t Tensor) Get() [3][3][][][][][][]float64 {
	return t.N
}

func (t Tensor) GetIdx(c, c_, i, j, k, i_, j_, k_ int) float64 {
	return t.N[c][c_][i][j][k][i_][j_][k_]
}

func (t Tensor) SetIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.N[c][c_][i][j][k][i_][j_][k_] = val
}

func (t Tensor) AddIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.N[c][c_][i][j][k][i_][j_][k_] += val
}

func AddTensors(Ns ...Tensor) Tensor {

	var result [3][3][][][][][][]float64

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			result[c][c_] = make([][][][][][]float64, Nx)
			for i := 0; i < Nx; i++ {
				result[c][c_][i] = make([][][][][]float64, Ny)
				for j := 0; j < Ny; j++ {
					result[c][c_][i][j] = make([][][][]float64, Nz)
					for k := 0; k < Nz; k++ {
						result[c][c_][i][j][k] = make([][][]float64, Nx)
						for i_ := 0; i_ < Nx; i_++ {
							result[c][c_][i][j][k][i_] = make([][]float64, Ny)
							for j_ := 0; j_ < Ny; j_++ {
								result[c][c_][i][j][k][i_][j_] = make([]float64, Nz)
								for k_ := 0; k_ < Nz; k_++ {
									result[c][c_][i][j][k][i_][j_][k_] = 0
									for _, N := range Ns {
										result[c][c_][i][j][k][i_][j_][k_] += N.GetIdx(c, c_, i, j, k, i_, j_, k_)
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return Tensor{result}
}

func (t Tensor) To1D() []float64 {
	totalSize := 3 * Nx * Ny * Nz * 3 * Nx * Ny * Nz
	arr := make([]float64, totalSize)

	for c := 0; c < 3; c++ {
		for i := 0; i < Nx; i++ {
			for j := 0; j < Ny; j++ {
				for k := 0; k < Nz; k++ {
					// make a row
					p := 3 * Nx * Ny * Nz * (Nx*Ny*Nz*c + Ny*Nz*i + Nz*j + k)
					for c_ := 0; c_ < 3; c_++ {
						for i_ := 0; i_ < Nx; i_++ {
							for j_ := 0; j_ < Ny; j_++ {
								for k_ := 0; k_ < Nz; k_++ {
									arr[p+Nx*Ny*Nz*c_+Ny*Nz*i_+Nz*j_+k_] = t.GetIdx(c, c_, i, j, k, i_, j_, k_)
								}
							}
						}
					}
				}
			}
		}
	}
	return arr
}

func (t Tensor) To2D() [][]float64 {

	totalSize := 3 * Nx * Ny * Nz

	N := make([][]float64, totalSize)

	for c := 0; c < 3; c++ {
		for i := 0; i < Nx; i++ {
			for j := 0; j < Ny; j++ {
				for k := 0; k < Nz; k++ {
					// make a row
					row := make([]float64, totalSize)
					for c_ := 0; c_ < 3; c_++ {
						for i_ := 0; i_ < Nx; i_++ {
							for j_ := 0; j_ < Ny; j_++ {
								for k_ := 0; k_ < Nz; k_++ {
									row[Nx*Ny*Nz*c_+Ny*Nz*i_+Nz*j_+k_] = t.GetIdx(c, c_, i, j, k, i_, j_, k_)
								}
							}
						}
					}
					N[Nx*Ny*Nz*c+Ny*Nz*i+Nz*j+k] = row
				}
			}
		}
	}
	return N

}

func (t Tensor) ToCSV(name string) {

	f, _ := os.Create(name)
	defer f.Close()

	wr := csv.NewWriter(f)
	defer wr.Flush()

	N := t.To2D()

	for _, row := range N {
		wr.Write(strings.Fields(strings.Trim(fmt.Sprint(row), "[]")))
	}
}

func Zeros() Tensor {
	var N [3][3][][][][][][]float64

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			N[c][c_] = make([][][][][][]float64, Nx)
			for i := 0; i < Nx; i++ {
				N[c][c_][i] = make([][][][][]float64, Ny)
				for j := 0; j < Ny; j++ {
					N[c][c_][i][j] = make([][][][]float64, Nz)
					for k := 0; k < Nz; k++ {
						N[c][c_][i][j][k] = make([][][]float64, Nx)
						for i_ := 0; i_ < Nx; i_++ {
							N[c][c_][i][j][k][i_] = make([][]float64, Ny)
							for j_ := 0; j_ < Ny; j_++ {
								N[c][c_][i][j][k][i_][j_] = make([]float64, Nz)
								for k_ := 0; k_ < Nz; k_++ {
									N[c][c_][i][j][k][i_][j_][k_] = 0
								}
							}
						}
					}
				}
			}
		}
	}

	return Tensor{N}
}

func (t Tensor) Copy() Tensor {

	var N_ [3][3][][][][][][]float64

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			N_[c][c_] = make([][][][][][]float64, Nx)
			for i := 0; i < Nx; i++ {
				N_[c][c_][i] = make([][][][][]float64, Ny)
				for j := 0; j < Ny; j++ {
					N_[c][c_][i][j] = make([][][][]float64, Nz)
					for k := 0; k < Nz; k++ {
						N_[c][c_][i][j][k] = make([][][]float64, Nx)
						for i_ := 0; i_ < Nx; i_++ {
							N_[c][c_][i][j][k][i_] = make([][]float64, Ny)
							for j_ := 0; j_ < Ny; j_++ {
								N_[c][c_][i][j][k][i_][j_] = make([]float64, Nz)
								for k_ := 0; k_ < Nz; k_++ {
									N_[c][c_][i][j][k][i_][j_][k_] = t.GetIdx(c, c_, i, j, k, i_, j_, k_)
								}
							}
						}
					}
				}
			}
		}
	}

	return Tensor{N_}

}

func (t Tensor) Energy() float64 {

	mSl := en.M.Buffer().HostCopy()
	m := mSl.Vectors() //order is Z, Y, X

	E := 0.
	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {
				for i_ := 0; i_ < Nx; i_++ {
					for j_ := 0; j_ < Ny; j_++ {
						for k_ := 0; k_ < Nz; k_++ {
							for c := 0; c < 3; c++ {
								for c_ := 0; c_ < 3; c_++ {

									E += t.GetIdx(c, c_, i, j, k, i_, j_, k_) * float64(m[c][k][j][i]*m[c_][k_][j_][i_])
								}
							}
						}
					}
				}
			}
		}
	}

	return .5 * E

}
