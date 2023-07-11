package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// A Tensor contains data in a [nComp][nComp][Nz * Ny *Nx][Nz * Ny * Nx] array.
// This spatial ordering may seem odd but it corresponds to slices in mumax.
type Tensor struct {
	n     [][][][]float64
	NComp int
	Size  [3]int
}

func (t Tensor) Length() int {
	length := 1
	for c := 0; c < 3; c++ {
		length *= t.Size[c]
	}
	return length
}

func (t Tensor) Idx(i, j, k int) int {
	return t.Size[0]*(t.Size[1]*k+j) + i
}

// Get returns the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_).
func (t Tensor) GetIdx(c, c_, i, j, k, i_, j_, k_ int) float64 {
	return t.n[c][c_][t.Idx(i, j, k)][t.Idx(i_, j_, k_)]
}

// Set sets the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_), to value val.
func (t Tensor) SetIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.n[c][c_][t.Idx(i, j, k)][t.Idx(i_, j_, k_)] = val
}

// Add adds val to the element corresponding to the c component of magnetisation at position (i, j, k)
// and the c_ component of magnetisation at position (i_, j_, k_).
func (t Tensor) AddIdx(c, c_, i, j, k, i_, j_, k_ int, val float64) {
	t.n[c][c_][t.Idx(i, j, k)][t.Idx(i_, j_, k_)] += val
}

// AddTensors returns a tensor corresponding to the elementwise addition of the inputs.
func AddTensors(Ns ...Tensor) Tensor {

	if len(Ns) == 0 {
		panic("can't determine size to return")
	}

	NComp := Ns[0].NComp
	length := Ns[0].Length()

	result := make([][][][]float64, NComp)
	for c := 0; c < NComp; c++ {
		result[c] = make([][][]float64, NComp)
		for c_ := 0; c_ < 3; c_++ {
			result[c][c_] = make([][]float64, length)
			for i := 0; i < length; i++ {
				result[c][c_][i] = make([]float64, length)
				for i_ := 0; i_ < length; i_++ {
					for _, N := range Ns {
						result[c][c_][i][i_] += N.n[c][c_][i][i_]
					}
				}
			}
		}
	}

	return Tensor{n: result, NComp: NComp, Size: Ns[0].Size}
}

// To1D returns a [(NComp*Nz*Ny*Nx) * (NComp*Nz*Ny*Nx)]float64 of elements.
// The elements are ordered by magnetisation component, then z position, then y position, then x position for the r position,
// and then analogously for the r' position.
func (t Tensor) To1D() []float64 {
	length := t.Length()
	totalSize := t.NComp * length * t.NComp * length
	arr := make([]float64, totalSize)

	for c := 0; c < t.NComp; c++ {
		for i := 0; i < length; i++ {
			p := t.NComp * length * (length*c + i)
			for c_ := 0; c_ < t.NComp; c_++ {
				for i_ := 0; i_ < length; i_++ {
					arr[p+length*c_+i_] = t.n[c][c_][i][i_]
				}

			}
		}
	}
	return arr
}

func From1D(arr []float64, nComp int, size [3]int) Tensor {

	length := size[0] * size[1] * size[2]
	t := ZeroTensor(nComp, size)

	for c := 0; c < nComp; c++ {
		for i := 0; i < length; i++ {
			p := nComp * length * (length*c + i)
			for c_ := 0; c_ < nComp; c_++ {
				for i_ := 0; i_ < length; i_++ {
					t.n[c][c_][i][i_] = arr[p+length*c_+i_]
				}

			}
		}
	}

	return t
}

// To2D returns a [NComp*Nx*Ny*Nz][NComp*Nx*Ny*Nz]float64 of elements with the first dimension corresponding to the r position and second to the r' position.
// The elements are ordered by magnetisation component, then x position, then y position, then z position.
func (t Tensor) To2D() [][]float64 {

	length := t.Length()
	totalSize := t.NComp * length

	twoD := make([][]float64, totalSize)

	for c := 0; c < t.NComp; c++ {
		for i := 0; i < length; i++ {
			row := make([]float64, totalSize)
			for c_ := 0; c_ < t.NComp; c_++ {
				for i_ := 0; i_ < length; i_++ {
					row[length*c_+i_] = t.n[c][c_][i][i_]
				}
			}
			twoD[length*c+i] = row
		}
	}
	return twoD
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

// ZeroTensor returns a Tensor with all elements set to zero.
func ZeroTensor(nComp int, size [3]int) Tensor {

	length := 1
	for c := 0; c < 3; c++ {
		length *= size[c]
	}

	n := make([][][][]float64, nComp)
	for c := 0; c < nComp; c++ {
		n[c] = make([][][]float64, nComp)
		for c_ := 0; c_ < nComp; c_++ {
			n[c][c_] = make([][]float64, length)
			for i := 0; i < length; i++ {
				n[c][c_][i] = make([]float64, length)
			}
		}
	}

	return Tensor{n: n, NComp: nComp, Size: size}
}

// Copy returns a deep copy of the tensor.
func (t Tensor) Copy() Tensor {

	length := t.Length()

	n_ := make([][][][]float64, t.NComp)
	for c := 0; c < t.NComp; c++ {
		n_[c] = make([][][]float64, t.NComp)
		for c_ := 0; c_ < t.NComp; c_++ {
			n_[c][c_] = make([][]float64, length)
			for i := 0; i < length; i++ {
				n_[c][c_][i] = make([]float64, length)
				for i_ := 0; i_ < length; i_++ {
					n_[c][c_][i][i_] = t.n[c][c_][i][i_]
				}
			}
		}
	}

	return Tensor{n: n_, NComp: t.NComp, Size: t.Size}
}

// TSP (Tensor Slice Product) returns the operation of a tensor on a real slice
func (t Tensor) TSP(v *data.Slice) *data.Slice {

	//check that the slice has the same number of components:
	if v.NComp() != t.NComp {
		panic("number of components are not the same")
	}
	if t.Size != v.Size() {
		panic("sizes do not match")
	}
	Nx := t.Size[0]
	Ny := t.Size[1]
	Nz := t.Size[2]

	//check whether v is on the cpu. if not copy it here
	cpu := v.CPUAccess()
	if !cpu {
		v = v.HostCopy()
	}

	vVectors := v.Tensors()

	//work with 64 bit for doing the tensor slice product.
	resArr := make([][][][]float64, t.NComp)
	for c := 0; c < t.NComp; c++ {
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

	for c := 0; c < t.NComp; c++ {
		for c_ := 0; c_ < t.NComp; c_++ {
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

	result := data.NewSlice(t.NComp, t.Size)
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
		resGPU := cuda.NewSlice(3, t.Size)
		data.Copy(resGPU, result)
		return resGPU
	}
}

// TCSP (Tensor Complex Slice Product) returns the operation of a tensor on a complex slice
func (t Tensor) TCSP(v CSlice) CSlice {

	//don't need to check that it is on host because TSP does this
	return CSlice{
		real: t.TSP(v.Real()),
		imag: t.TSP(v.Imag()),
	}

}

// ITCSP (Imagninary Tensor Complex Slice Product) returns the operation of a i * tensor on a complex.
// This is primarily used as it is part of the eigenvalue equation.
func (t Tensor) ITCSP(v CSlice) CSlice {

	cpu := v.CPUAccess()
	if !cpu {
		v = v.HostCopy()
	}

	w := t.TSP(v.Imag())
	wVec := w.Tensors()

	for c := 0; c < 3; c++ {
		for k := 0; k < t.Size[2]; k++ {
			for j := 0; j < t.Size[1]; j++ {
				for i := 0; i < t.Size[0]; i++ {
					wVec[c][k][j][i] = -wVec[c][k][j][i]
				}
			}
		}
	}

	result := CSlice{
		real: w,
		imag: t.TSP(v.Real()),
	}

	if cpu {
		return result
	} else {
		return result.DevCopy()
	}
}

// XY returns a copy of a tensor with the z component removed.
func (t Tensor) XY() Tensor {

	if t.NComp < 2 {
		panic("there is not an X and Y component")
	}

	length := t.Length()

	n_ := make([][][][]float64, 2)
	for c := 0; c < 2; c++ {
		n_[c] = make([][][]float64, 2)
		for c_ := 0; c_ < 2; c_++ {
			n_[c][c_] = make([][]float64, length)
			for i := 0; i < length; i++ {
				n_[c][c_][i] = make([]float64, length)
				for i_ := 0; i_ < length; i_++ {
					n_[c][c_][i][i_] = t.n[c][c_][i][i_]
				}
			}
		}
	}

	return Tensor{n: n_, NComp: 2, Size: t.Size}

}
