package field

import (
	"fmt"
	"testing"

	cuda "github.com/mumax/3/cuda"
	en "github.com/mumax/3/engine"

	mag "github.com/will-henderson/mumax-vhf/mag"
)

func TestField(t *testing.T) {
	//compare field of directly from mumax and from the exported functions.

	var tests = []mag.SystemParameters{
		{Nx: 10, Ny: 10, Nz: 10,
			Dx: 20e-9, Dy: 20e-9, Dz: 20e-9,
			Msat: 16074.649, Aex: 1.3e-12, Ku1: 756.3,
			AnisU: [3]float64{0, 0, 1},
			B_ext: [3]float64{.5, 0, 0},
		},
	}

	for _, test := range tests {
		runner(test, t)

	}
}

func runner(p mag.SystemParameters, t *testing.T) {

	defer en.InitAndClose()()
	p.Setup()

	en.M.Set(en.RandomMagSeed(0))
	M := en.M.Buffer()

	fmt.Println("Magnetisation:", M)
	fmt.Println("*Magnetisation:", *M)

	//create a slice on GPU to assign field to
	BMumaxGPU := cuda.NewSlice(3, en.Mesh().Size())
	en.SetDemagField(BMumaxGPU)
	BMumax := BMumaxGPU.HostCopy().Vectors()
	BMumaxGPU.Free()

	en.M.Set(en.RandomMagSeed(1))
	MNew := en.M.Buffer().HostCopy()

	BGPU := cuda.NewSlice(3, en.Mesh().Size())
	SetDemagField(M, BGPU)
	B := BGPU.HostCopy().Vectors()
	BGPU.Free()

	//check the fields are the same
	maxErr := float32(1e-3)
	err := false
loop:
	for c := 0; c < 3; c++ {
		for i := 0; i < mag.Nx; i++ {
			for j := 0; j < mag.Ny; j++ {
				for k := 0; k < mag.Nz; k++ {
					frac := B[c][k][j][i] / BMumax[c][k][j][i]
					if frac > 1+maxErr || frac < 1-maxErr {
						err = true
						break loop
					}
				}
			}
		}
	}

	if err {
		t.Error("Demag Fields are not Equal")
	}

	//check that the magnetisation has not been changed by the functions.
	MWant := MNew.Vectors()
	MGot := en.M.Buffer().HostCopy().Vectors()

	err = false
loop2:
	for c := 0; c < 3; c++ {
		for i := 0; i < mag.Nx; i++ {
			for j := 0; j < mag.Ny; j++ {
				for k := 0; k < mag.Nz; k++ {
					frac := MWant[c][k][j][i] / MGot[c][k][j][i]
					if frac > 1+maxErr || frac < 1-maxErr {
						err = true
						break loop2
					}
				}
			}
		}
	}

	if err {
		t.Error("Magnetisation was changed")
	}

}
