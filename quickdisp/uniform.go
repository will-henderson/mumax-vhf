package quickdisp

import (
	"math"

	"github.com/mumax/3/cuda"
	en "github.com/mumax/3/engine"
)

type AssumeUniform struct {
	demagMatrix, uniAnisMatrix [3][3]float32
	zeemanVector               [3]float32
	dtsi                       float64
}

func NewAssumeUniform() (au AssumeUniform) {

	// these do all assume that the quantities determining these are time independent.
	// which seems reasonable because this is just used for minimisation.
	au.demagMatrix = en.UniformDemag()
	au.uniAnisMatrix = makeUniAnisMatrix()
	au.zeemanVector = makeZeemanVector()

	return au

}

func makeUniAnisMatrix() (mat [3][3]float32) {

	regionCounts := en.RegionCounts()
	for i := 0; i < en.NREGION; i++ {
		if regionCounts[i] != 0 {
			ku1 := en.Ku1.GetRegion(i)
			anisu := en.AnisU.GetRegion(i)
			msat := en.Msat.GetRegion(i)

			for c := 0; c < 3; c++ {
				for c_ := 0; c_ < 3; c_++ {
					mat[c][c_] += float32(regionCounts[i]) * float32(anisu[c]*anisu[c_]*2*ku1/msat)
				}
			}
		}
	}
	return mat
}

func makeZeemanVector() (B [3]float32) {

	B_ext, rm := en.B_ext.Slice()
	if rm {
		defer B_ext.Free()
	}

	for c := 0; c < 3; c++ {
		B[c] = cuda.Sum(B_ext.Comp(c))
	}

	return B
}

func (au *AssumeUniform) UniformUniAnis(m [3]float32) (B [3]float32) {
	return matvecmul(au.uniAnisMatrix, m)
}

func (au *AssumeUniform) UniformDemag(m [3]float32) (B [3]float32) {
	return matvecmul(au.demagMatrix, m)
}

func (au *AssumeUniform) UniformZeeman() (B [3]float32) {
	return au.zeemanVector
}

func matvecmul(mat [3][3]float32, vec [3]float32) (res [3]float32) {

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			res[c] += mat[c][c_] * vec[c_]
		}
	}
	return res
}

func (au *AssumeUniform) Field(m [3]float32) (B [3]float32){
	demag := au.UniformDemag(m)
	uniAnis := au.UniformUniAnis(m)
	zeeman := au.UniformZeeman()

	for c := 0; c < 3; c++ {
		B[c] = demag[c] + uniAnis[c] + zeeman[c]
	}
	return B
}

func (au *AssumeUniform) Energy(m [3]float32) float32{
	return -.5 * 
}

func (au *AssumeUniform) UniformMinima() (m [3]float32) {
	au.dtsi = 1e-15 * en.GammaLL

	//initially just set m in a random direction. 
	k1 := au.torqueNP(mInit)


	return m
}

func (au AssumeUniform) UniformRK23Step(m0 [3]float32, k1 [3]float32) ([3]float32, [3]float32) {

	h := float32(au.dtsi * en.GammaLL)

	m1 := normalise(madd2(m0, k1, 1, .5*h))
	k2 := au.torqueNP(m1)

	m2 := normalise(madd2(m0, k2, 1, .75))
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

func (au AssumeUniform) torqueNP(m [3]float32) [3]float32 {
	//this calculates the torque without precession -- used for relaxation.

	demag := au.UniformDemag(m)
	uniAnis := au.UniformUniAnis(m)
	zeeman := au.UniformZeeman()

	var field [3]float32
	for c := 0; c < 3; c++ {
		field[c] = demag[c] + uniAnis[c] + zeeman[c]
	}

	torque := cross(cross(m, field), m)
	return torque

}

func normalise(in [3]float32) (out [3]float32) {

	norm := float32(0.)
	for c := 0; c < 3; c++ {
		norm += in[c] * in[c]
	}
	norm = float32(math.Sqrt(float64(norm)))

	for c := 0; c < 3; c++ {
		out[c] = in[c] / norm
	}

	return out
}

func dot(a, b [3]float32) (res float32){

	for c := 0; c < 3; c++{
		res += a[c]*b[c] 
	}
	return res
}

func cross(a, b [3]float32) (res [3]float32) {
	res[0] = a[1]*b[2] - a[2]*b[1]
	res[1] = a[2]*b[0] - a[0]*b[2]
	res[2] = a[0]*b[1] - a[1]*b[0]
	return res
}

func madd2(a, b [3]float32, f1, f2 float32) (res [3]float32) {
	for c := 0; c < 3; c++ {
		res[c] = f1*a[c] + f2*b[c]
	}
	return res
}

func madd4(a1, a2, a3, a4 [3]float32, f1, f2, f3, f4 float32) (res [3]float32) {
	for c := 0; c < 3; c++ {
		res[c] = f1*a1[c] + f2*a2[c] + f3*a3[c] + f4*a4[c]
	}
	return res
}
