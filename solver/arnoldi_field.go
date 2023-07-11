package solver

import (
	"fmt"
	"math"
	"time"

	. "github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/field"
	"github.com/will-henderson/mumax-vhf/mag"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
	"github.com/mumax/3/util"
)

type ArnoldiField struct {
	eigenSolver
}

func (solver ArnoldiField) Modes() ([]float64, []CSlice) {

	totalSize := 2 * en.Mesh().NCell()

	le := field.NewLinearEvolution()
	rot := new(field.RotationToZ)
	rot.InitRotation()
	arn := newArnoldiS(totalSize, totalSize-2, -1, "I", "SM", 0, 100*totalSize, nil)

	xSl2 := cuda.NewSlice(2, en.MeshSize())
	ySl2 := cuda.NewSlice(2, en.MeshSize())
	xSl3 := cuda.NewSlice(3, en.MeshSize())
	ySl3 := cuda.NewSlice(3, en.MeshSize())

	ido, x, y := arn.iterate()
	for ido == 1 || ido == -1 {
		xArr := make([][]float32, 2)
		xArr[0] = x[0:en.Mesh().NCell()]
		xArr[1] = x[en.Mesh().NCell():totalSize]
		xSlCPU := data.SliceFromArray(xArr, en.Mesh().Size())
		data.Copy(xSl2, xSlCPU)

		rot.DerotateMode(xSl3, xSl2)
		le.Operate(ySl3, xSl3)
		rot.RotateMode(ySl2, ySl3)

		yArr := make([][]float32, 2)
		yArr[0] = y[0:en.Mesh().NCell()]
		yArr[1] = y[en.Mesh().NCell():totalSize]
		ySlCPU := data.SliceFromArray(yArr, en.Mesh().Size())
		data.Copy(ySlCPU, ySl2)

		ido, x, y = arn.iterate()
	}

	xSl2.Free()
	ySl2.Free()
	xSl3.Free()
	ySl3.Free()

	info, infoString := arn.iterateInfo()
	util.AssertMsg(info == 0, infoString)

	values, vectors := arn.extract(true, nil)
	info, infoString = arn.extractInfo()
	util.AssertMsg(info == 0, infoString)

	nevReturned := len(values)
	iterations := arn.iparam[2]
	util.Log(fmt.Sprintf("Found %d eigenvalues (out of total dimension %d) in %d iterations.", nevReturned, totalSize, iterations))

	freq := make([]float64, nevReturned)
	modes := make([]CSlice, nevReturned)

	Nx := en.MeshSize()[0]
	Ny := en.MeshSize()[1]
	Nz := en.MeshSize()[2]

	rotCPU := new(mag.RotationToZ)
	rotCPU.InitRotation()

	for p := 0; p < nevReturned; p++ {

		freq[p] = imag(values[p])

		mode := NewCSliceCPU(2, en.MeshSize())
		modeReal := mode.Real().Tensors()
		modeImag := mode.Imag().Tensors()

		for c := 0; c < 2; c++ {
			for i := 0; i < Nx; i++ {
				for j := 0; j < Ny; j++ {
					for k := 0; k < Nz; k++ {
						modeReal[c][k][j][i] = float32(real(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
						modeImag[c][k][j][i] = float32(imag(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
					}
				}
			}
		}

		modes[p] = rotCPU.DerotateMode(mode)
	}

	return freq, modes

}

type ArnoldiFieldUnrotated struct {
	eigenSolver
}

func (solver ArnoldiFieldUnrotated) Modes() ([]float64, []CSlice) {

	totalSize := 3 * en.Mesh().NCell()

	le := field.NewLinearEvolution()
	rot := new(field.RotationToZ)
	rot.InitRotation()
	defer rot.Free()
	arn := newArnoldiS(totalSize, totalSize-2, -1, "I", "LM", 0, 100*totalSize, nil)

	xSl3 := cuda.NewSlice(3, en.MeshSize())
	ySl3 := cuda.NewSlice(3, en.MeshSize())

	ido, x, y := arn.iterate()
	for ido == 1 || ido == -1 {
		xArr := make([][]float32, 3)
		xArr[0] = x[0:en.Mesh().NCell()]
		xArr[1] = x[en.Mesh().NCell() : 2*en.Mesh().NCell()]
		xArr[2] = x[2*en.Mesh().NCell() : 3*en.Mesh().NCell()]
		xSlCPU := data.SliceFromArray(xArr, en.Mesh().Size())
		data.Copy(xSl3, xSlCPU)

		le.Operate(ySl3, xSl3)

		yArr := make([][]float32, 3)
		yArr[0] = y[0:en.Mesh().NCell()]
		yArr[1] = y[en.Mesh().NCell() : 2*en.Mesh().NCell()]
		yArr[2] = y[2*en.Mesh().NCell() : 3*en.Mesh().NCell()]
		ySlCPU := data.SliceFromArray(yArr, en.Mesh().Size())
		data.Copy(ySlCPU, ySl3)

		ido, x, y = arn.iterate()
	}

	xSl3.Free()
	ySl3.Free()

	info, infoString := arn.iterateInfo()
	util.AssertMsg(info == 0, infoString)

	values, vectors := arn.extract(true, nil)
	info, infoString = arn.extractInfo()
	util.AssertMsg(info == 0, infoString)

	nevReturned := len(values)
	iterations := arn.iparam[2]
	util.Log(fmt.Sprintf("Found %d eigenvalues (out of total dimension %d) in %d iterations.", nevReturned, totalSize, iterations))

	Nx := en.MeshSize()[0]
	Ny := en.MeshSize()[1]
	Nz := en.MeshSize()[2]

	totalSize = 2 * Nx * Ny * Nz
	freqs := make([]float64, totalSize)
	modes := make([]CSlice, totalSize)

	q := 0
	for p := 0; p < nevReturned; p++ {

		freq := imag(values[p])

		if math.Abs(freq) > 1e5 {

			freqs[q] = freq
			modes[q] = NewCSliceCPU(3, [3]int{Nx, Ny, Nz})
			modeReal := modes[q].Real().Vectors()
			modeImag := modes[q].Imag().Vectors()

			for c := 0; c < 3; c++ {
				for k := 0; k < Nz; k++ {
					for j := 0; j < Ny; j++ {
						for i := 0; i < Nx; i++ {
							modeReal[c][k][j][i] = float32(real(vectors[q][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
							modeImag[c][k][j][i] = float32(imag(vectors[q][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
						}
					}
				}
			}

			q++

		}
	}

	fmt.Println(q, nevReturned, totalSize)

	return freqs[0:q], modes[0:q]
}

type ArnoldiField2 struct {
	eigenSolver
}

func (solver ArnoldiField2) Modes() ([]float64, []CSlice) {

	totalSize := 2 * en.Mesh().NCell()

	le := field.NewLinearEvolution()
	rot := new(mag.RotationToZ)
	rot.InitRotation()
	arn := newArnoldiS(totalSize, totalSize-2, -1, "I", "SM", 0, 100*totalSize, nil)

	xSl3 := cuda.NewSlice(3, en.MeshSize())
	ySl3 := cuda.NewSlice(3, en.MeshSize())

	ido, x, y := arn.iterate()
	for ido == 1 || ido == -1 {

		xArr := make([][]float32, 2)
		xArr[0] = x[0:en.Mesh().NCell()]
		xArr[1] = x[en.Mesh().NCell():totalSize]
		xSl2 := data.SliceFromArray(xArr, en.Mesh().Size())
		xSl3CPU := rot.DerotateModeReal(xSl2)
		data.Copy(xSl3, xSl3CPU)

		le.Operate(ySl3, xSl3)

		ySl3CPU := data.NewSlice(3, en.MeshSize())
		data.Copy(ySl3CPU, ySl3)
		ySl2 := rot.RotateModeReal(ySl3CPU)

		yArr := make([][]float32, 2)
		yArr[0] = y[0:en.Mesh().NCell()]
		yArr[1] = y[en.Mesh().NCell():totalSize]
		ySlval := data.SliceFromArray(yArr, en.Mesh().Size())
		data.Copy(ySlval, ySl2)

		ido, x, y = arn.iterate()
	}

	xSl3.Free()
	ySl3.Free()

	info, infoString := arn.iterateInfo()
	util.AssertMsg(info == 0, infoString)

	values, vectors := arn.extract(true, nil)
	info, infoString = arn.extractInfo()
	util.AssertMsg(info == 0, infoString)

	nevReturned := len(values)
	iterations := arn.iparam[2]
	util.Log(fmt.Sprintf("Found %d eigenvalues (out of total dimension %d) in %d iterations.", nevReturned, totalSize, iterations))

	freq := make([]float64, nevReturned)
	modes := make([]CSlice, nevReturned)

	Nx := en.MeshSize()[0]
	Ny := en.MeshSize()[1]
	Nz := en.MeshSize()[2]

	rotCPU := new(mag.RotationToZ)
	rotCPU.InitRotation()

	for p := 0; p < nevReturned; p++ {

		freq[p] = imag(values[p])

		mode := NewCSliceCPU(2, en.MeshSize())
		modeReal := mode.Real().Tensors()
		modeImag := mode.Imag().Tensors()

		for c := 0; c < 2; c++ {
			for i := 0; i < Nx; i++ {
				for j := 0; j < Ny; j++ {
					for k := 0; k < Nz; k++ {
						modeReal[c][k][j][i] = float32(real(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
						modeImag[c][k][j][i] = float32(imag(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
					}
				}
			}
		}

		modes[p] = rotCPU.DerotateMode(mode)
	}

	return freq, modes

}

type ArnoldiFieldTimes struct {
	eigenSolver
	lopTime, arpackTime, finishingTime time.Duration
}

func (solver *ArnoldiFieldTimes) Modes() ([]float64, []CSlice) {

	totalSize := 2 * en.Mesh().NCell()

	le := field.NewLinearEvolution()
	rot := new(field.RotationToZ)
	rot.InitRotation()
	arn := newArnoldiS(totalSize, totalSize-2, -1, "I", "SM", 0, 100*totalSize, nil)

	xSl2 := cuda.NewSlice(2, en.MeshSize())
	ySl2 := cuda.NewSlice(2, en.MeshSize())
	xSl3 := cuda.NewSlice(3, en.MeshSize())
	ySl3 := cuda.NewSlice(3, en.MeshSize())

	tAfter := time.Now()

	ido, x, y := arn.iterate()
	for ido == 1 || ido == -1 {
		xArr := make([][]float32, 2)
		xArr[0] = x[0:en.Mesh().NCell()]
		xArr[1] = x[en.Mesh().NCell():totalSize]
		xSlCPU := data.SliceFromArray(xArr, en.Mesh().Size())
		data.Copy(xSl2, xSlCPU)

		tBefore := time.Now()
		solver.arpackTime += tBefore.Sub(tAfter)

		rot.DerotateMode(xSl3, xSl2)
		le.Operate(ySl3, xSl3)
		rot.RotateMode(ySl2, ySl3)

		tAfter := time.Now()
		solver.lopTime += tAfter.Sub(tBefore)

		yArr := make([][]float32, 2)
		yArr[0] = y[0:en.Mesh().NCell()]
		yArr[1] = y[en.Mesh().NCell():totalSize]
		ySlCPU := data.SliceFromArray(yArr, en.Mesh().Size())
		data.Copy(ySlCPU, ySl2)

		ido, x, y = arn.iterate()
	}

	solver.arpackTime += time.Now().Sub(tAfter)

	xSl2.Free()
	ySl2.Free()
	xSl3.Free()
	ySl3.Free()

	info, infoString := arn.iterateInfo()
	util.AssertMsg(info == 0, infoString)

	tBefore := time.Now()
	values, vectors := arn.extract(true, nil)
	solver.finishingTime = time.Now().Sub(tBefore)
	info, infoString = arn.extractInfo()
	util.AssertMsg(info == 0, infoString)

	nevReturned := len(values)
	iterations := arn.iparam[2]
	util.Log(fmt.Sprintf("Found %d eigenvalues (out of total dimension %d) in %d iterations.", nevReturned, totalSize, iterations))

	freq := make([]float64, nevReturned)
	modes := make([]CSlice, nevReturned)

	Nx := en.MeshSize()[0]
	Ny := en.MeshSize()[1]
	Nz := en.MeshSize()[2]

	rotCPU := new(mag.RotationToZ)
	rotCPU.InitRotation()

	for p := 0; p < nevReturned; p++ {

		freq[p] = imag(values[p])

		mode := NewCSliceCPU(2, en.MeshSize())
		modeReal := mode.Real().Tensors()
		modeImag := mode.Imag().Tensors()

		for c := 0; c < 2; c++ {
			for i := 0; i < Nx; i++ {
				for j := 0; j < Ny; j++ {
					for k := 0; k < Nz; k++ {
						modeReal[c][k][j][i] = float32(real(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
						modeImag[c][k][j][i] = float32(imag(vectors[p][c*Nx*Ny*Nz+k*Nx*Ny+j*Nx+i]))
					}
				}
			}
		}

		modes[p] = rotCPU.DerotateMode(mode)
	}

	return freq, modes

}
