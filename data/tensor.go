package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// A Tensor contains data in a [3][3][Nz][Ny][Nx][Nz][Ny][Nx] array.
// This spatial ordering may seem odd but it corresponds to slices in mumax.
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
	return t.N[c][c_][k][j][i][k_][j_][i_]
}

// Set sets the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_), to value val.
func (t Tensor) SetIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.N[c][c_][k][j][i][k_][j_][i_] = val
}

// Add adds val to the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_).
func (t Tensor) AddIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.N[c][c_][k][j][i][k_][j_][i_] += val
}

// AddTensors returns a tensor corresponding to the elementwise addition of the inputs.
func AddTensors(Ns ...Tensor) Tensor {

	var result [3][3][][][][][][]float64

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			result[c][c_] = make([][][][][][]float64, Nz)
			for k := 0; k < Nz; k++ {
				result[c][c_][k] = make([][][][][]float64, Ny)
				for j := 0; j < Ny; j++ {
					result[c][c_][k][j] = make([][][][]float64, Nx)
					for i := 0; i < Nx; i++ {
						result[c][c_][k][j][i] = make([][][]float64, Nz)
						for k_ := 0; k_ < Nz; k_++ {
							result[c][c_][k][j][i][k_] = make([][]float64, Ny)
							for j_ := 0; j_ < Ny; j_++ {
								result[c][c_][k][j][i][k_][j_] = make([]float64, Nx)
								for i_ := 0; i_ < Nx; i_++ {
									result[c][c_][k][j][i][k_][j_][i_] = 0
									for _, N := range Ns {
										result[c][c_][k][j][i][k_][j_][i_] += N.GetIdx(c, c_, i, j, k, i_, j_, k_)
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

// To1D returns a [(3*Nz*Ny*Nx) * (3*Nz*Ny*Nx)]float64 of elements.
// The elements are ordered by magnetisation component, then z position, then y position, then x position for the r position,
// and then analogously for the r' position.
func (t Tensor) To1D() []float64 {
	totalSize := 3 * Nx * Ny * Nz * 3 * Nx * Ny * Nz
	arr := make([]float64, totalSize)

	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					// make a row
					p := 3 * Nx * Ny * Nz * (Nx*Ny*Nz*c + Ny*Nx*k + Nx*j + i)
					for c_ := 0; c_ < 3; c_++ {
						for k_ := 0; k_ < Nz; k_++ {
							for j_ := 0; j_ < Ny; j_++ {
								for i_ := 0; i_ < Nx; i_++ {
									arr[p+Nx*Ny*Nz*c_+Ny*Nx*k_+Nx*j_+i_] = t.GetIdx(c, c_, i, j, k, i_, j_, k_)
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
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					// make a row
					row := make([]float64, totalSize)
					for c_ := 0; c_ < 3; c_++ {
						for k_ := 0; k_ < Nz; k_++ {
							for j_ := 0; j_ < Ny; j_++ {
								for i_ := 0; i_ < Nx; i_++ {
									row[Nx*Ny*Nz*c_+Ny*Nx*k_+Nx*j_+i_] = t.GetIdx(c, c_, i, j, k, i_, j_, k_)
								}
							}
						}
					}
					N[Nx*Ny*Nz*c+Ny*Nx*k+Nx*j+i] = row
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
			N[c][c_] = make([][][][][][]float64, Nz)
			for k := 0; k < Nz; k++ {
				N[c][c_][k] = make([][][][][]float64, Ny)
				for j := 0; j < Ny; j++ {
					N[c][c_][k][j] = make([][][][]float64, Nx)
					for i := 0; i < Nx; i++ {
						N[c][c_][k][j][i] = make([][][]float64, Nz)
						for k_ := 0; k_ < Nz; k_++ {
							N[c][c_][k][j][i][k_] = make([][]float64, Ny)
							for j_ := 0; j_ < Ny; j_++ {
								N[c][c_][k][j][i][k_][j_] = make([]float64, Nx)
								for i_ := 0; i_ < Nx; i_++ {
									N[c][c_][k][j][i][k_][j_][i_] = 0
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
			N_[c][c_] = make([][][][][][]float64, Nz)
			for k := 0; k < Nz; k++ {
				N_[c][c_][k] = make([][][][][]float64, Ny)
				for j := 0; j < Ny; j++ {
					N_[c][c_][k][j] = make([][][][]float64, Nx)
					for i := 0; i < Nx; i++ {
						N_[c][c_][k][j][i] = make([][][]float64, Nz)
						for k_ := 0; k_ < Nz; k_++ {
							N_[c][c_][k][j][i][k_] = make([][]float64, Ny)
							for j_ := 0; j_ < Ny; j_++ {
								N_[c][c_][k][j][i][k_][j_] = make([]float64, Nx)
								for i_ := 0; i_ < Nx; i_++ {
									N_[c][c_][k][j][i][k_][j_][i_] = t.GetIdx(c, c_, i, j, k, i_, j_, k_)
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

// TSP (Tensor Slice Product) returns the operation of a tensor on a slice
func TSP(t Tensor, v *data.Slice) *data.Slice {

	//check whether v is on the cpu. if not copy it here
	cpu := v.CPUAccess()
	if !cpu {
		v = v.HostCopy()
	}

	vVectors := v.Vectors()

	//work with 64 bit for doing the tensor slice product.
	var resArr [3][][][]float64
	for c := 0; c < 3; c++ {
		resArr[c] = make([][][]float64, Nz)
		for k := 0; k < Nz; k++ {
			resArr[c][k] = make([][]float64, Ny)
			for j := 0; j < Ny; j++ {
				resArr[c][k][j] = make([]float64, Nx)
				for i := 0; i < Nx; i++ {
					resArr[c][k][j][i] = 0
				}
			}
		}
	}

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			for k := 0; k < Nz; k++ {
				for j := 0; j < Ny; j++ {
					for i := 0; i < Nx; i++ {
						for k_ := 0; k_ < Nz; k_++ {
							for j_ := 0; j_ < Ny; j_++ {
								for i_ := 0; i_ < Nx; i_++ {
									resArr[c][k][j][i] += t.GetIdx(c, c_, i, j, k, i_, j_, k_) * float64(vVectors[c_][k_][j_][i_])
								}
							}
						}
					}
				}
			}
		}
	}

	result := data.NewSlice(3, [3]int{Nx, Ny, Nz})
	resVectors := result.Vectors()

	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					resVectors[c][k][j][i] = float32(resArr[c][k][j][i])
				}
			}
		}
	}

	//put back on the gpu if that is where the input was from.
	if cpu {
		return result
	} else {
		resGPU := cuda.NewSlice(3, [3]int{Nx, Ny, Nz})
		data.Copy(resGPU, result)
		return resGPU
	}
}

func TCSP(t Tensor, v CSlice) CSlice {

	//don't need to check that it is on host because TSP does this
	return CSlice{
		real: TSP(t, v.Real()),
		imag: TSP(t, v.Imag()),
	}

}
