package tests

import (
	"math"
	"sort"

	"github.com/mumax/3/data"
	. "github.com/will-henderson/mumax-vhf/data"
)

// EqualScalars tests whether the fractional error between a and b is less than maxErr
// It returns an integer, 0 if less than maxErr, 1 if greater than maxError
// The return value is integer rather than boolean for consistency with the other equality functions in the package.
func EqualScalars(a, b float64, maxErr float64) int {
	n := 0
	if math.Abs((a-b)/a) > maxErr {
		n = 1
	}
	return n
}

// EqualSlices tests whether the fractional error betweeen each element of a and b is less than maxErr
// It returns an integer equal to the number of elements where the factional error is greater than maxErr
// If Slices are not of same size, then all the elements are treated as wrong.
func EqualSlices(a, b *data.Slice, maxErr float64) int {

	if !a.CPUAccess() {
		a = a.HostCopy()
	}

	if !b.CPUAccess() {
		b = b.HostCopy()
	}

	A := a.Tensors()
	B := b.Tensors()

	nCompA := a.NComp()
	nCompB := b.NComp()
	sizeA := a.Size()
	sizeB := b.Size()
	if nCompA != nCompB || sizeA != sizeB {
		nElemA := sizeA[0] * sizeA[1] * sizeA[2] * nCompA
		nElemB := sizeB[0] * sizeB[1] * sizeB[2] * nCompB
		if nElemA > nElemB {
			return nElemA
		} else {
			return nElemB
		}
	}

	n := 0
	for c := 0; c < nCompA; c++ {
		for k := 0; k < a.Size()[2]; k++ {
			for j := 0; j < a.Size()[1]; j++ {
				for i := 0; i < a.Size()[0]; i++ {
					if math.Abs(float64((A[c][k][j][i]-B[c][k][j][i])/A[c][k][j][i])) > maxErr {
						n++
					}
				}
			}
		}
	}
	return n
}

// EqualCSlices tests whether the fractional error betweeen both the imaginary and real part of each element of a and b is less than maxErr.
// It returns an integer equal to the number of elements where the factional error is greater than maxErr
func EqualCSlices(a, b CSlice, maxErr float64) int {

	if !a.CPUAccess() {
		a = a.HostCopy()
	}

	if !b.CPUAccess() {
		b = b.HostCopy()
	}

	nCompA := a.NComp()
	nCompB := b.NComp()
	sizeA := a.Size()
	sizeB := b.Size()
	if nCompA != nCompB || sizeA != sizeB {
		nElemA := sizeA[0] * sizeA[1] * sizeA[2] * nCompA
		nElemB := sizeB[0] * sizeB[1] * sizeB[2] * nCompB
		if nElemA > nElemB {
			return nElemA
		} else {
			return nElemB
		}
	}

	Ar := a.Real().Tensors()
	Br := b.Real().Tensors()
	Ai := a.Imag().Tensors()
	Bi := b.Imag().Tensors()
	n := 0
	for c := 0; c < nCompA; c++ {
		for k := 0; k < a.Size()[2]; k++ {
			for j := 0; j < a.Size()[1]; j++ {
				for i := 0; i < a.Size()[0]; i++ {
					if math.Abs(float64((Ar[c][k][j][i]-Br[c][k][j][i])/Ar[c][k][j][i])) > maxErr ||
						math.Abs(float64((Ai[c][k][j][i]-Bi[c][k][j][i])/Ai[c][k][j][i])) > maxErr {
						n++
					}
				}
			}
		}
	}
	return n
}

type eigenpairs struct {
	f []float64
	m []CSlice
}

func (eps eigenpairs) Len() int           { return len(eps.f) }
func (eps eigenpairs) Less(i, j int) bool { return eps.f[i] < eps.f[j] }
func (eps eigenpairs) Swap(i, j int) {
	eps.f[i], eps.f[j] = eps.f[j], eps.f[i]
	eps.m[i], eps.m[j] = eps.m[j], eps.m[i]
}
func (eps eigenpairs) Slice(i, j int) eigenpairs {
	return eigenpairs{eps.f[i:j], eps.m[i:j]}
}

// EqualDecompositions tests whether the fractional error between eigenmodes, and
// It returns an integer equal to the number of elements where the factional error is greater than maxErr.
// It assumes that the vectors are normalised to 1.
func EqualDecompositions(aM, bM []CSlice, aF, bF []float64, valErr, vecErr float64) int {

	//first check that all of the arrays are the same length.
	if len(aM) != len(bM) || len(aF) != len(bF) || len(aM) != len(aF) {
		maxlen := len(aM)
		if len(bM) > maxlen {
			maxlen = len(bM)
		}
		if len(aF) > maxlen {
			maxlen = len(aF)
		}
		if len(bF) > maxlen {
			maxlen = len(bF)
		}
		return maxlen
	}

	A := eigenpairs{aF, aM}
	B := eigenpairs{bF, bM}

	sort.Sort(A)
	sort.Sort(B)

	return equalDecompositionsSorted(A, B, valErr, vecErr)
}

type eigenpairsSM struct {
	f []float64
	m []CSlice
}

func (eps eigenpairsSM) Len() int           { return len(eps.f) }
func (eps eigenpairsSM) Less(i, j int) bool { return math.Abs(eps.f[i]) < math.Abs(eps.f[j]) }
func (eps eigenpairsSM) Swap(i, j int) {
	eps.f[i], eps.f[j] = eps.f[j], eps.f[i]
	eps.m[i], eps.m[j] = eps.m[j], eps.m[i]
}
func (eps eigenpairsSM) Slice(i, j int) eigenpairs {
	return eigenpairs{eps.f[i:j], eps.m[i:j]}
}

func EqualSubDecompositions(aM, bM []CSlice, aF, bF []float64, valErr, vecErr float64) int {

	var nevs int
	if len(aM) < len(bM) {
		nevs = len(aM)
	} else {
		nevs = len(bM)
	}

	ASM := eigenpairsSM{aF, aM}
	BSM := eigenpairsSM{bF, bM}

	sort.Sort(ASM)
	sort.Sort(BSM)
	As := ASM.Slice(0, nevs)
	Bs := BSM.Slice(0, nevs)

	//this second sorting deals with the case where two evs have (to within numerical error) the same magnitude, but different signs.
	A := eigenpairs{As.f, As.m}
	B := eigenpairs{Bs.f, Bs.m}
	sort.Sort(A)
	sort.Sort(B)

	return equalDecompositionsSorted(A.Slice(0, nevs), B.Slice(0, nevs), valErr, vecErr)
}

// A is taken to be the correct values. i.e. the one that sets eigenspaces.
func equalDecompositionsSorted(A, B eigenpairs, valErr, vecErr float64) int {

	num := len(A.f)
	n := 0
	for i := 0; i < num; i++ {

		if math.Abs((A.f[i]-B.f[i])/A.f[i]) > valErr {
			n++
		} else { // so the eigenvalue is good. Is the eigenvector?
			// find the biggest possible range of eigenvalues of A that would be within the tolerance.
			imax := i
			imin := i
			for ; imax < num && math.Abs((A.f[imax]-A.f[i])/A.f[i]) < valErr; imax++ {
			}
			for ; imin >= 0 && math.Abs((A.f[imin]-A.f[i])/A.f[i]) < valErr; imin-- {
			}
			if !inEigenspace(A.m[imin+1:imax], B.m[i], vecErr) {
				n++
			}
		}
	}
	return n

}

func inEigenspace(Am []CSlice, bm CSlice, vecErr float64) bool {

	Nx := bm.Size()[0]
	Ny := bm.Size()[1]
	Nz := bm.Size()[2]

	inspace := NewCSliceCPU(3, [3]int{Nx, Ny, Nz})
	inspaceReal := inspace.Real().Vectors()
	inspaceImag := inspace.Imag().Vectors()

	for _, am := range Am {
		amReal := am.Real().Vectors()
		amImag := am.Imag().Vectors()

		factor := dotc(am, bm)

		for c := 0; c < 3; c++ {
			for k := 0; k < Nz; k++ {
				for j := 0; j < Ny; j++ {
					for i := 0; i < Nx; i++ {
						component := complex64(factor) * complex(amReal[c][k][j][i], amImag[c][k][j][i])
						inspaceReal[c][k][j][i] += real(component)
						inspaceImag[c][k][j][i] += imag(component)
					}
				}
			}
		}

	}

	diff := subtract(bm, inspace)
	norm := real(dotc(diff, diff))

	if norm > vecErr {
		return false
	}

	return true

}

func subtract(a, b CSlice) CSlice {
	aReal := a.Real().Vectors()
	aImag := a.Imag().Vectors()
	bReal := b.Real().Vectors()
	bImag := b.Imag().Vectors()

	c := NewCSliceCPU(3, a.Size())
	cReal := c.Real().Vectors()
	cImag := c.Imag().Vectors()

	for c := 0; c < 3; c++ {
		for k := 0; k < a.Size()[2]; k++ {
			for j := 0; j < a.Size()[1]; j++ {
				for i := 0; i < a.Size()[0]; i++ {
					cReal[c][k][j][i] = aReal[c][k][j][i] - bReal[c][k][j][i]
					cImag[c][k][j][i] = aImag[c][k][j][i] - bImag[c][k][j][i]
				}
			}
		}
	}

	return c

}

func dotc(a, b CSlice) complex128 {
	aReal := a.Real().Vectors()
	aImag := a.Imag().Vectors()
	bReal := b.Real().Vectors()
	bImag := b.Imag().Vectors()

	result := complex(0, 0)
	for c := 0; c < 3; c++ {
		for k := 0; k < a.Size()[2]; k++ {
			for j := 0; j < a.Size()[1]; j++ {
				for i := 0; i < a.Size()[0]; i++ {
					result += complex128(complex(aReal[c][k][j][i], -aImag[c][k][j][i]) * complex(bReal[c][k][j][i], bImag[c][k][j][i]))

				}
			}
		}
	}

	return result

}
