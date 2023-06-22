package mag

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	en "github.com/mumax/3/engine"

	. "github.com/will-henderson/mumax-vhf/setup"
)

// A Tensor contains data in a [3][3][Nx][Ny][Nz][Nx][Ny][Nz] array.
type Tensor struct {
	N [3][3][][][][][][]float64
}

// Get returns the underlying array structure.
func (t Tensor) Get() [3][3][][][][][][]float64 {
	return t.N
}

// Get returns the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_).
func (t Tensor) GetIdx(c, c_, i, j, k, i_, j_, k_ int) float64 {
	return t.N[c][c_][i][j][k][i_][j_][k_]
}

// Set sets the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_), to value val.
func (t Tensor) SetIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.N[c][c_][i][j][k][i_][j_][k_] = val
}

// Add adds val to the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_).
func (t Tensor) AddIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.N[c][c_][i][j][k][i_][j_][k_] += val
}

// AddTensors returns a tensor corresponding to the elementwise addition of the inputs.
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

// To1D returns a [(3*Nx*Ny*Nz) * (3*Nx*Ny*Nz)]float64 of elements.
// The elements are ordered by magnetisation component, then x position, then y position, then z position for the r position,
// and then analogously for the r' position.
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

// To2D returns a [3*Nx*Ny*Nz][3*Nx*Ny*Nz]float64 of elements with the first dimension corresponding to the r position and second to the r' position.
// The elements are ordered by magnetisation component, then x position, then y position, then z position.
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

// ToCSV saves a tensor in CSV format in a file named name. The columns correspond to the r position and rows to the r' position.
// The elements are ordered by magnetisation component, then x position, then y position, then z position.
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

// Zeros returns a Tensor with all elements set to zero.
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

// Copy returns a deep copy of the tensor.
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

// Energy returns the energy of the tensor. Calculated as .5 * Σ_rr' m_r t_rr' m_r'.
// It depends on the magnetization, but this is not an input, rather the value of en.M is used.
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
