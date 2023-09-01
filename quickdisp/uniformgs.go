package quickdisp

import (
	"math"
	"math/rand"

	"github.com/mumax/3/cuda"
	en "github.com/mumax/3/engine"
	"github.com/mumax/3/util"
)

// Arguments for the method of finding the ground state.
// AVERAGE_FIELD means that the global field for magnetisation is found, and then the energy
// corresponding to this field is minimised.
// AVERAGE_MAGNETISATION means that the magnetisation is minimised assuming a spatially varying field,
// and then averaged.
const (
	AVERAGE_FIELD         = 0
	AVERAGE_MAGNETISATION = 1
)

func UniformGroundState(method int) (m0 [3]float32) {

	switch method {
	default:
		util.Fatalf("Unknown minimisation scheme: %v", method)
	case AVERAGE_FIELD:
		au := NewAssumeUniform()
		m0 = au.Relax()
	case AVERAGE_MAGNETISATION:
		en.Relax()
		for c := 0; c < 3; c++ {
			m0[c] = cuda.Sum(en.M.Buffer().Comp(c)) / float32(en.Mesh().NCell())
		}
		m0 = normalise(m0)
	}

	return m0

}

type AssumeUniform struct {
	demagMatrix, uniAnisMatrix [3][3]float64
	zeemanVector               [3]float64
	dtsi                       float64
}

func NewAssumeUniform() (au AssumeUniform) {

	// these do all assume that the quantities determining these are time independent.
	// which seems reasonable because this is just used for minimisation.
	au.demagMatrix = makeDemagMatrix()
	au.uniAnisMatrix = makeUniAnisMatrix()
	au.zeemanVector = makeZeemanVector()

	return au

}

func makeDemagMatrix() (mat [3][3]float64) {

	m32 := en.UniformDemag()

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			mat[c][c_] = float64(m32[c][c_])
		}
	}

	return mat
}

func makeUniAnisMatrix() (mat [3][3]float64) {

	regionCounts := en.RegionCounts()
	for i := 0; i < en.NREGION; i++ {
		if regionCounts[i] != 0 {
			ku1 := en.Ku1.GetRegion(i)
			anisu := en.AnisU.GetRegion(i)
			msat := en.Msat.GetRegion(i)

			for c := 0; c < 3; c++ {
				for c_ := 0; c_ < 3; c_++ {
					mat[c][c_] += float64(regionCounts[i]) * anisu[c] * anisu[c_] * 2 * ku1 / msat
				}
			}
		}
	}
	return mat
}

func makeZeemanVector() (B [3]float64) {

	B_ext, rm := en.B_ext.Slice()
	if rm {
		defer B_ext.Free()
	}

	for c := 0; c < 3; c++ {
		B[c] = float64(cuda.Sum(B_ext.Comp(c)))
	}

	return B
}

func (au *AssumeUniform) UniAnis(m [3]float64) (B [3]float64) {
	return matvecmul(au.uniAnisMatrix, m)
}

func (au *AssumeUniform) Demag(m [3]float64) (B [3]float64) {
	return matvecmul(au.demagMatrix, m)
}

func (au *AssumeUniform) Zeeman() (B [3]float64) {
	return au.zeemanVector
}

func matvecmul(mat [3][3]float64, vec [3]float64) (res [3]float64) {

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			res[c] += mat[c][c_] * vec[c_]
		}
	}
	return res
}

func (au *AssumeUniform) Field(m [3]float64) (B [3]float64) {
	demag := au.Demag(m)
	uniAnis := au.UniAnis(m)
	zeeman := au.Zeeman()

	for c := 0; c < 3; c++ {
		B[c] = demag[c] + uniAnis[c] + zeeman[c]
	}
	return B
}

func (au *AssumeUniform) Energy(m [3]float64) float64 {
	return -.5 * dot(m, au.Field(m))
}

func (au *AssumeUniform) Relax() (m32 [3]float32) {
	au.dtsi = 1e-18

	//initially just set m in a random direction.
	seed := 0
	rng := rand.New(rand.NewSource(int64(seed)))
	var m [3]float64
	for c := 0; c < 3; c++ {
		m[c] = rng.Float64() - .5
	}
	m = normalise(m)

	const N = 3 // evaluate energy every 3 steps.

	k1 := au.torqueNP(m)
	m, k1 = au.NStep(m, k1, N)
	E0 := au.Energy(m)
	m, k1 = au.NStep(m, k1, N)
	E1 := au.Energy(m)

	for E1 < E0 {
		m, k1 = au.NStep(m, k1, N)
		E0, E1 = E1, au.Energy(m)
	}

	for c := 0; c < 3; c++ {
		m32[c] = float32(m[c])
	}

	return m32
}

func (au *AssumeUniform) NStep(m [3]float64, k1 [3]float64, N int) ([3]float64, [3]float64) {
	for i := 0; i < N; i++ {
		m, k1 = au.Step(m, k1)
	}

	return m, k1
}

func (au *AssumeUniform) Step(m0 [3]float64, k1 [3]float64) ([3]float64, [3]float64) {

	h := float64(au.dtsi * en.GammaLL)

	m1 := normalise(madd2(m0, k1, 1, .5*h))
	k2 := au.torqueNP(m1)

	m2 := normalise(madd2(m0, k2, 1, .75*h))
	k3 := au.torqueNP(m2)

	m3 := normalise(madd4(m0, k1, k2, k3, 1, (2./9.)*h, (1./3.)*h, (4./9.)*h))
	k4 := au.torqueNP(m3)

	Err := madd4(k1, k2, k3, k4, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1. / 8.))
	err := math.Sqrt(float64(Err[0]*Err[0] + Err[1]*Err[1] + Err[2]*Err[2]))

	if err < en.MaxErr || au.dtsi <= en.MinDt {
		au.dtsi = adaptDt(au.dtsi, math.Pow(en.MaxErr/err, 1./3.))
		return m3, k4
	} else {
		au.dtsi = adaptDt(au.dtsi, math.Pow(en.MaxErr/err, 1./4.))
		return m0, k1
	}
}

func adaptDt(dt, corr float64) float64 {

	if math.IsNaN(corr) {
		corr = 1
	}

	corr *= en.Headroom
	if corr > 2 {
		corr = 2
	} else if corr < .5 {
		corr = .5
	}

	dt *= corr

	if en.MinDt != 0 && dt < en.MinDt {
		dt = en.MinDt
	}
	if en.MaxDt != 0 && dt > en.MaxDt {
		dt = en.MaxDt
	}

	return dt

}

func (au AssumeUniform) torqueNP(m [3]float64) [3]float64 {
	//this calculates the torque without precession -- used for relaxation.

	field := au.Field(m)
	torque := cross(cross(m, field), m)
	return torque

}

type float interface {
	float32 | float64
}

func normalise[T float32 | float64](in [3]T) (out [3]T) {

	norm := T(0.)
	for c := 0; c < 3; c++ {
		norm += in[c] * in[c]
	}
	norm = T(math.Sqrt(float64(norm)))

	for c := 0; c < 3; c++ {
		out[c] = in[c] / norm
	}

	return out
}

func dot[T float32 | float64](a, b [3]T) (res T) {

	for c := 0; c < 3; c++ {
		res += a[c] * b[c]
	}
	return res
}

func cross[T float32 | float64](a, b [3]T) (res [3]T) {
	res[0] = a[1]*b[2] - a[2]*b[1]
	res[1] = a[2]*b[0] - a[0]*b[2]
	res[2] = a[0]*b[1] - a[1]*b[0]
	return res
}

func madd2(a, b [3]float64, f1, f2 float64) (res [3]float64) {
	for c := 0; c < 3; c++ {
		res[c] = f1*a[c] + f2*b[c]
	}
	return res
}

func madd4(a1, a2, a3, a4 [3]float64, f1, f2, f3, f4 float64) (res [3]float64) {
	for c := 0; c < 3; c++ {
		res[c] = f1*a1[c] + f2*a2[c] + f3*a3[c] + f4*a4[c]
	}
	return res
}
