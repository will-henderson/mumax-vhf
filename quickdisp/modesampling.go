package quickdisp

import (
	"math"

	en "github.com/mumax/3/engine"
)

// ExpandPoints takes the input dimpoints which are the points for each dimension at which we want every other points from
// every other dimension to be evaluated, and expands it.
func ExpandPoints(dimPoints [3][]int32) (samplePoints [3][]int32) {

	Sx := len(dimPoints[0])
	Sy := len(dimPoints[1])
	Sz := len(dimPoints[2])

	for c := 0; c < 3; c++ {
		samplePoints[c] = make([]int32, Sx*Sy*Sz)
	}

	for i := 0; i < Sx; i++ {
		for j := 0; j < Sy; j++ {
			for k := 0; k < Sz; k++ {
				samplePoints[0][(Sy*i+j)*Sz+k] = dimPoints[0][i]
				samplePoints[1][(Sy*i+j)*Sz+k] = dimPoints[1][j]
				samplePoints[2][(Sy*i+j)*Sz+k] = dimPoints[2][k]
			}
		}
	}

	return samplePoints

}

func AllModes() [3][]int32 {
	size := en.MeshSize()

	var dimPoints [3][]int32
	for c := 0; c < 3; c++ {
		dimPoints[c] = make([]int32, size[c])
		var limn int
		if size[c]%2 == 0 {
			limn = -size[c] / 2
		} else {
			limn = -(size[c] - 1) / 2
		}
		for i := 0; i < size[c]; i++ {
			dimPoints[c][i] = int32(i + limn)
		}
	}

	return ExpandPoints(dimPoints)
}

func ModeSample(kStart, kStop [3]float64, nPoints [3]int) [3][]int32 {

	size := en.Mesh().Size()
	cellsize := en.Mesh().CellSize()

	var dimPoints [3][]int32
	for c := 0; c < 3; c++ {
		pStart := int(math.Floor(float64(size[c]) * cellsize[c] * kStart[c] / (2 * math.Pi)))
		pStop := int(math.Floor(float64(size[c]) * cellsize[c] * kStop[c] / (2 * math.Pi)))

		// don't allow these to be larger than the limits.
		if size[c]%2 == 0 {
			// in even case we will let the extreme limit be sampled by either but only once.
			limn := -size[c] / 2
			hasextreme := false
			if pStart <= limn {
				pStart = limn
				hasextreme = true
			}
			limp := size[c]/2 - 1
			if pStop > limp {
				if hasextreme {
					pStop = limp
				} else {
					pStop = limp + 1
				}
			}

		} else {
			limn := -(size[c] - 1) / 2
			if pStart < limn {
				pStart = limn
			}
			limp := (size[c] - 1) / 2
			if pStop > limp {
				pStop = limp
			}

		}

		if pStop < pStart {
			// there are no valid points.
			continue
		}

		//now sample.

		maxPoints := pStop - pStart + 1
		if nPoints[c] > maxPoints {
			dimPoints[c] = make([]int32, maxPoints)
			for i := pStop; i <= pStart; i++ {
				dimPoints[c][i-pStop] = int32(i)
			}
		} else if nPoints[c] == 1 {
			dimPoints[c] = make([]int32, 1)
			dimPoints[c][0] = int32(math.Floor(float64(maxPoints) / 2.))
		} else {
			step := float64(maxPoints-1) / float64(nPoints[c]-1)
			dimPoints[c] = make([]int32, nPoints[c])
			for i := 0; i < nPoints[c]-1; i++ {
				dimPoints[c][i] = int32(math.Floor(float64(i) * step))
			}
			dimPoints[c][nPoints[c]-1] = int32(pStop)
		}

	}

	return ExpandPoints(dimPoints)

}

func AlongDirection(direction [3]float64) (ks [3][]int32) {
	//we want to find the values of k that lie closest to along the direction

	size := en.Mesh().Size()
	cellsize := en.Mesh().CellSize()

	direction = normalise(direction)
	var scal [3]float64
	for c := 0; c < 3; c++ {
		scal[c] = (float64(size[c]) * cellsize[c] * direction[c])
	}

	//set up the initial point.
	var prevPoint [3]int32
	var pks [3][]int32
	for c := 0; c < 3; c++ {
		pks[c] = append(ks[c], prevPoint[c])
	}

	var idx [3]int
	var α [3]float64
	for c := 0; c < 3; c++ {
		α[c] = 1 / scal[c]
	}

	var boundaries [3]int
	for c := 0; c < 3; c++ {
		boundaries[c] = (size[c] + 1) / 2
	}

	for idx[0] < boundaries[0] && idx[1] < boundaries[1] && idx[2] < boundaries[2] { // <= or < here???
		c := minElem(α)

		var point [3]int32
		for c_ := 0; c_ < 3; c_++ {
			point[c_] = int32(math.Round(α[c] * scal[c_]))
		}

		//check if this point is already hit. Because of the ordering of this algorithm
		//this will be the previous points.
		if point != prevPoint {
			for c_ := 0; c_ < 3; c_++ {
				pks[c_] = append(pks[c_], point[c_])
			}
			prevPoint = point
		}

		idx[c]++
		α[c] = float64(idx[c]) / scal[c]
	}

	//now construct the negative versions.
	n := len(pks[0])
	for c := 0; c < 3; c++ {
		ks[c] = make([]int32, 2*n-1)
		ks[c][0] = pks[c][0]
		for i := 1; i < n; i++ {
			ks[c][i] = pks[c][i]
			ks[c][2*n-1-i] = -pks[c][i]
		}
	}

	return ks

}

func AlongDirectionAP(direction [3]float64) (ks [3][]int32) {
	//we want to find the values of k that lie closest to along the direction

	size := en.Mesh().Size()
	cellsize := en.Mesh().CellSize()

	direction = normalise(direction)
	var scal [3]float64
	for c := 0; c < 3; c++ {
		scal[c] = (float64(size[c]) * cellsize[c] * direction[c])
	}

	//set up the initial point.
	var prevPoint [3]int32
	for c := 0; c < 3; c++ {
		ks[c] = append(ks[c], prevPoint[c])
	}

	var idx [3]int
	var α [3]float64
	for c := 0; c < 3; c++ {
		α[c] = 1 / scal[c]
	}

	for idx[0] < size[0] && idx[1] < size[1] && idx[2] < size[2] { // <= or < here???
		c := minElem(α)

		var point [3]int32
		for c_ := 0; c_ < 3; c_++ {
			point[c_] = int32(math.Round(α[c] * scal[c_]))
		}

		//check if this point is already hit. Because of the ordering of this algorithm
		//this will be the previous points.
		if point != prevPoint {
			for c_ := 0; c_ < 3; c_++ {
				ks[c_] = append(ks[c_], point[c_])
			}
			prevPoint = point
		}

		idx[c]++
		α[c] = float64(idx[c]) / scal[c]
	}

	return ks

}

func minElem(arr [3]float64) int {
	if arr[0] <= arr[1] && arr[0] <= arr[2] {
		return 0
	} else if arr[1] <= arr[2] {
		return 1
	} else {
		return 2
	}
}

func maxElem(arr [3]float64) int {
	if arr[0] >= arr[1] && arr[0] >= arr[2] {
		return 0
	} else if arr[1] >= arr[2] {
		return 1
	} else {
		return 2
	}
}

func Kmag(px, py, pz int32, direction [3]float64) float64 {
	size := en.Mesh().Size()
	cellsize := en.Mesh().CellSize()

	kx_ := float64(px) * direction[0] / (float64(size[0]) * cellsize[0])
	ky_ := float64(py) * direction[1] / (float64(size[1]) * cellsize[1])
	kz_ := float64(pz) * direction[2] / (float64(size[2]) * cellsize[2])

	return 2 * math.Pi * (kx_ + ky_ + kz_)
}
