package tests

import (
	"github.com/mumax/3/data"
	. "github.com/will-henderson/mumax-vhf/data"
)

// EqualScalars tests whether the fractional error between A and B is less than maxErr
// It returns an integer, 0 if less than maxErr, 1 if greater than maxError
// The return value is integer rather than boolean for consistency with the other equality functions in the package.
func EqualScalars(A, B float64, maxErr float64) int {
	frac := A / B
	n := 0
	if frac > 1+maxErr || frac < 1-maxErr {
		n = 1
	}
	return n
}

// EqualSlices tests whether the fractional error betweeen each element of a and b is less than maxErr
// It returns an integer equal to the number of elements where the factional error is greater than maxErr
func EqualSlices(a, b *data.Slice, maxErr float32) int {

	if !a.CPUAccess() {
		a = a.HostCopy()
	}

	if !b.CPUAccess() {
		b = b.HostCopy()
	}

	A := a.Vectors()
	B := b.Vectors()

	n := 0
	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nz; i++ {
					frac := A[c][k][j][i] / B[c][k][j][i]
					if frac > 1+maxErr || frac < 1-maxErr {
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
func EqualCSlices(a, b CSlice, maxErr float32) int {

	if !a.CPUAccess() {
		a = a.HostCopy()
	}

	if !b.CPUAccess() {
		b = b.HostCopy()
	}

	Ar := a.Real().Vectors()
	Br := b.Real().Vectors()
	Ai := a.Imag().Vectors()
	Bi := b.Imag().Vectors()
	n := 0
	for c := 0; c < 3; c++ {
		for k := 0; k < Nz; k++ {
			for j := 0; j < Ny; j++ {
				for i := 0; i < Nx; i++ {
					fracReal := Ar[c][k][j][i] / Br[c][k][j][i]
					fracImag := Ai[c][k][j][i] / Bi[c][k][j][i]
					if fracReal > 1+maxErr || fracReal < 1-maxErr || fracImag > 1+maxErr || fracImag < 1-maxErr {
						n++
					}
				}
			}
		}
	}
	return n
}
