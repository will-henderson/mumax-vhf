package quickdisp

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"sort"

	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

var OnlyPlotPositiveFrequencies = true

const DELIM = "\t"
const NEWLINE = "\n"

type argSort struct {
	sort.Float64Slice
	idx []int
}

func (s argSort) Swap(i, j int) {
	s.Float64Slice.Swap(i, j)
	s.idx[i], s.idx[j] = s.idx[j], s.idx[i]
}

func newArgSort(arr []float64) *argSort {
	s := &argSort{Float64Slice: sort.Float64Slice(arr), idx: make([]int, len(arr))}
	for i := range s.idx {
		s.idx[i] = i
	}
	return s
}

func DispImageDat(ks [3][]int32, frequencies []float64, magnitudes [][]float32, direction [3]float64, name string) {

	buf, err := httpfs.Create(name)
	util.FatalErr(err)

	nK := len(magnitudes)
	kmag := make([]float64, nK)
	for kidx := 0; kidx < nK; kidx++ {
		kmag[kidx] = Kmag(ks[0][kidx], ks[1][kidx], ks[2][kidx], direction)
	}

	kas := newArgSort(kmag)
	sort.Sort(kas)
	kOrder := kas.idx
	//copy frequencies to avoid rearranging the input
	fcop := make([]float64, len(frequencies))
	copy(fcop, frequencies)

	fas := newArgSort(fcop)
	sort.Sort(fas)
	fOrder := fas.idx

	for i, kidx := range kOrder {
		k := kmag[i]
		fmt.Println(k)
		for j, fidx := range fOrder {
			f := fcop[j]

			if !(OnlyPlotPositiveFrequencies && f < 0) {
				fmt.Fprint(buf, k, f, math.Log10(float64(magnitudes[kidx][fidx])), NEWLINE)
			}

		}
		fmt.Fprint(buf, NEWLINE)
	}

	buf.Close()
}

func GnuplotColorMap(infile string) {
	basename := util.NoExt(infile)
	outfile := basename + ".png"

	gnucmdFormatted := fmt.Sprintf(gnucmd, outfile, infile)
	fmt.Println(gnucmdFormatted)
	gnuplotOut, _ := exec.Command("gnuplot", "-e", gnucmdFormatted).CombinedOutput()
	os.Stderr.Write(gnuplotOut)
}

var gnucmd = `set terminal png;

set output "%v";

set view map;
set xlabel "Wavenumber along direction";
set ylabel "Frequency (GHz)";
set zlabel "Logarithmic Intensity";

splot "%v" u 1:2:3 w pm3d;`
