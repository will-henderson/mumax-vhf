package solver

/*
#cgo CFLAGS: -I /home/will/amd/aocl/4.0/include_LP64
#cgo LDFLAGS: -L /usr/local/lib -larpack
#cgo LDFLAGS: -L /home/will/amd/aocl/4.0/lib_LP64 -lflame -lblis -lm
#include <arpack/arpack.h>
*/
import "C"

type arnoldiD struct {
	n, nev, ncv  C.int
	bmat, which  *C.char
	ido          *C.int
	tol          C.double
	resid        []float64
	ldv          C.int
	v            []float64
	iparam       [11]C.int
	ipntr        [14]C.int
	workd, workl []float64
	lworkl       C.int
	info         *C.int
}

func newArnoldiD(n, nev, ncv int, bmat, which string, tol float64, maxIter int, v0 []float64) arnoldiD {

	if nev > n-1 {
		panic("Can only compute maximum n-1 eigenvalues")
	}

	//the default parameter scipy uses.
	if ncv < 0 {
		ncv_ := 2*nev + 1
		if ncv_ > n {
			ncv = n
		} else if ncv_ < 20 {
			ncv = 20
		} else {
			ncv = ncv_
		}
	}

	if ncv > n || ncv < nev+2 {
		panic("Must have nev + 2 <= ncv <= n")
	}

	// v0 allows starting guess to be entered
	var info C.int
	var resid []float64
	if v0 == nil {
		info = C.int(0)
		resid = make([]float64, n)
	} else {
		info = C.int(1)
		resid = v0
	}

	ishfts := 1
	mode1 := 1

	ido := C.int(0)
	v := make([]float64, n*ncv)
	workd := make([]float64, 3*n)
	workl := make([]float64, 3*ncv*ncv+6*ncv)

	var iparam [11]C.int
	iparam[0] = C.int(ishfts)
	iparam[2] = C.int(maxIter)
	iparam[3] = 1
	iparam[6] = C.int(mode1)

	var ipntr [14]C.int

	return arnoldiD{
		n:      C.int(n),
		nev:    C.int(nev),
		ncv:    C.int(ncv),
		bmat:   C.CString(bmat),
		which:  C.CString(which),
		ido:    &ido,
		tol:    C.double(tol),
		resid:  resid,
		ldv:    C.int(n),
		v:      v,
		iparam: iparam,
		ipntr:  ipntr,
		workd:  workd,
		workl:  workl,
		lworkl: C.int(3*ncv*ncv + 6*ncv),
		info:   &info,
	}
}

func (arn *arnoldiD) iterate() (int32, []float64, []float64) {

	C.dnaupd_c(arn.ido, arn.bmat, arn.n, arn.which, arn.nev, arn.tol, (*C.double)(&arn.resid[0]), arn.ncv,
		(*C.double)(&arn.v[0]), arn.ldv, &arn.iparam[0], &arn.ipntr[0], (*C.double)(&arn.workd[0]), (*C.double)(&arn.workl[0]),
		arn.lworkl, arn.info)

	if *arn.ido != C.int(99) {
		return int32(*arn.ido), arn.workd[arn.ipntr[0]-1 : arn.ipntr[0]+arn.n-1], arn.workd[arn.ipntr[1]-1 : arn.ipntr[1]+arn.n-1]
	} else {
		return 99, nil, nil
	}
}

func (arn *arnoldiD) extract(computeVectors bool, selection []int32) ([]complex128, [][]complex128) {

	var rvec C.int
	if computeVectors == true {
		rvec = 1
	} else {
		rvec = 0
	}

	var howmany *C.char
	if selection == nil {
		howmany = C.CString("A")
		selection = make([]int32, arn.ncv)
	} else {
		howmany = C.CString("S")
	}

	dr := make([]float64, arn.nev+1)
	di := make([]float64, arn.nev+1)
	workev := make([]float64, 3*arn.ncv)

	C.dneupd_c(rvec, howmany, (*C.int)(&selection[0]), (*C.double)(&dr[0]), (*C.double)(&di[0]), (*C.double)(&arn.v[0]),
		arn.ldv, C.double(0.), C.double(0.), (*C.double)(&workev[0]), arn.bmat, arn.n, arn.which, arn.nev, arn.tol,
		(*C.double)(&arn.resid[0]), arn.ncv, (*C.double)(&arn.v[0]), arn.ldv, &arn.iparam[0], &arn.ipntr[0], (*C.double)(&arn.workd[0]),
		(*C.double)(&arn.workl[0]), arn.lworkl, arn.info)

	nevReturned := arn.iparam[4]

	values := make([]complex128, nevReturned)
	vectors := make([][]complex128, nevReturned)
	for i := C.int(0); i < nevReturned; i++ {
		values[i] = complex(dr[i], di[i])
		vectors[i] = make([]complex128, arn.n)
	}

	for i := C.int(0); i < nevReturned; i++ {
		if di[i] == 0 {
			for j := C.int(0); j < arn.n; j++ {
				vectors[i][j] = complex(arn.v[arn.ldv*i+j], 0)
			}
		} else {
			for j := C.int(0); j < arn.n; j++ {
				vectors[i][j] = complex(arn.v[arn.ldv*i+j], arn.v[arn.ldv*(i+1)+j])
				vectors[i+1][j] = complex(arn.v[arn.ldv*i+j], -arn.v[arn.ldv*(i+1)+j])
			}
			i++
		}
	}

	return values, vectors

}

var iterateInfoDescription = map[C.int]string{
	0: `Normal exit.`,
	1: `Maximum number of iterations taken.
       All possible eigenvalues of OP has been found. IPARAM(5) 
       returns the number of wanted converged Ritz values.`,
	2: `No longer an informational error. Deprecated starting 
       with release 2 of ARPACK.`,
	3: `No shifts could be applied during a cycle of the 
       Implicitly restarted Arnoldi iteration. One possibility 
       is to increase the size of NCV relative to NEV. `,
	-1: `N must be positive.`,
	-2: `NEV must be positive.`,
	-3: `NCV-NEV >= 2 and less than or equal to N.`,
	-4: `The maximum number of Arnoldi update iterations allowed 
        must be greater than zero.`,
	-5:  `WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'`,
	-6:  `BMAT must be one of 'I' or 'G'.`,
	-7:  `Length of private work array WORKL is not sufficient.`,
	-8:  `Error return from LAPACK eigenvalue calculation;`,
	-9:  `Starting vector is zero.`,
	-10: `IPARAM(7) must be 1,2,3,4.`,
	-11: `IPARAM(7) = 1 and BMAT = 'G' are incompatible.`,
	-12: `IPARAM(1) must be equal to 0 or 1.`,
	-13: `NEV and WHICH = 'BE' are incompatible.`,
	-9999: `Could not build an Arnoldi factorization.
           IPARAM(5) returns the size of the current Arnoldi 
           factorization. The user is advised to check that 
           enough workspace and array storage has been allocated.`,
}

func (arn arnoldiD) iterateInfo() (int, string) {
	return int(*arn.info), iterateInfoDescription[*arn.info]
}

var extractInfoDescription = map[C.int]string{
	0: `Normal exit.`,
	1: `The Schur form computed by LAPACK routine dlahqr 
       could not be reordered by LAPACK routine dtrsen. 
       Re-enter subroutine dneupd  with IPARAM(5)NCV and 
       increase the size of the arrays DR and DI to have 
       dimension at least dimension NCV and allocate at least NCV 
       columns for Z. NOTE: Not necessary if Z and V share 
       the same space. Please notify the authors if this error
       occurs.`,
	-1: `N must be positive.`,
	-2: `NEV must be positive.`,
	-3: `NCV-NEV >= 2 and less than or equal to N.`,
	-5: `WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'`,
	-6: `BMAT must be one of 'I' or 'G'.`,
	-7: `Length of private work WORKL array is not sufficient.`,
	-8: `Error return from calculation of a real Schur form.
        Informational error from LAPACK routine dlahqr .`,
	-9: `Error return from calculation of eigenvectors.
    	Informational error from LAPACK routine dtrevc.`,
	-10: `IPARAM(7) must be 1,2,3,4.`,
	-11: `IPARAM(7) = 1 and BMAT = 'G' are incompatible.`,
	-12: `HOWMNY = 'S' not yet implemented`,
	-13: `HOWMNY must be one of 'A' or 'P' if RVEC = .true.`,
	-14: `DNAUPD  did not find any eigenvalues to sufficient 
         accuracy`,
	-15: `DNEUPD got a different count of the number of converged 
         Ritz values than DNAUPD got.  This indicates the user 
         probably made an error in passing data from DNAUPD to 
         DNEUPD or that the data was modified before entering 
         DNEUPD`,
}

func (arn arnoldiD) extractInfo() (int, string) {
	return int(*arn.info), extractInfoDescription[*arn.info]
}

// Reimplement but for the single precision case.

type arnoldiS struct {
	n, nev, ncv  C.int
	bmat, which  *C.char
	ido          *C.int
	tol          C.float
	resid        []float32
	ldv          C.int
	v            []float32
	iparam       [11]C.int
	ipntr        [14]C.int
	workd, workl []float32
	lworkl       C.int
	info         *C.int
}

func newArnoldiS(n, nev, ncv int, bmat, which string, tol float32, maxIter int, v0 []float32) arnoldiS {

	if nev > n-1 {
		panic("Can only compute maximum n-1 eigenvalues")
	}

	//the default parameter scipy uses.
	if ncv < 0 {
		ncv_ := 2*nev + 1
		if ncv_ > n {
			ncv = n
		} else if ncv_ < 20 {
			ncv = 20
		} else {
			ncv = ncv_
		}
	}

	if ncv > n || ncv < nev+2 {
		panic("Must have nev + 2 <= ncv <= n")
	}

	// v0 allows starting guess to be entered
	var info C.int
	var resid []float32
	if v0 == nil {
		info = C.int(0)
		resid = make([]float32, n)
	} else {
		info = C.int(1)
		resid = v0
	}

	ishfts := 1
	mode1 := 1

	ido := C.int(0)
	v := make([]float32, n*ncv)
	workd := make([]float32, 3*n)
	workl := make([]float32, 3*ncv*ncv+6*ncv)

	var iparam [11]C.int
	iparam[0] = C.int(ishfts)
	iparam[2] = C.int(maxIter)
	iparam[3] = 1
	iparam[6] = C.int(mode1)

	var ipntr [14]C.int

	return arnoldiS{
		n:      C.int(n),
		nev:    C.int(nev),
		ncv:    C.int(ncv),
		bmat:   C.CString(bmat),
		which:  C.CString(which),
		ido:    &ido,
		tol:    C.float(tol),
		resid:  resid,
		ldv:    C.int(n),
		v:      v,
		iparam: iparam,
		ipntr:  ipntr,
		workd:  workd,
		workl:  workl,
		lworkl: C.int(3*ncv*ncv + 6*ncv),
		info:   &info,
	}
}

func (arn *arnoldiS) iterate() (int32, []float32, []float32) {

	C.snaupd_c(arn.ido, arn.bmat, arn.n, arn.which, arn.nev, arn.tol, (*C.float)(&arn.resid[0]), arn.ncv,
		(*C.float)(&arn.v[0]), arn.ldv, &arn.iparam[0], &arn.ipntr[0], (*C.float)(&arn.workd[0]), (*C.float)(&arn.workl[0]),
		arn.lworkl, arn.info)

	if *arn.ido != C.int(99) {
		return int32(*arn.ido), arn.workd[arn.ipntr[0]-1 : arn.ipntr[0]+arn.n-1], arn.workd[arn.ipntr[1]-1 : arn.ipntr[1]+arn.n-1]
	} else {
		return 99, nil, nil
	}
}

func (arn *arnoldiS) extract(computeVectors bool, selection []int32) ([]complex128, [][]complex128) {

	var rvec C.int
	if computeVectors == true {
		rvec = 1
	} else {
		rvec = 0
	}

	var howmany *C.char
	if selection == nil {
		howmany = C.CString("A")
		selection = make([]int32, arn.ncv)
	} else {
		howmany = C.CString("S")
	}

	dr := make([]float32, arn.nev+1)
	di := make([]float32, arn.nev+1)
	workev := make([]float32, 3*arn.ncv)

	C.sneupd_c(rvec, howmany, (*C.int)(&selection[0]), (*C.float)(&dr[0]), (*C.float)(&di[0]), (*C.float)(&arn.v[0]),
		arn.ldv, C.float(0.), C.float(0.), (*C.float)(&workev[0]), arn.bmat, arn.n, arn.which, arn.nev, arn.tol,
		(*C.float)(&arn.resid[0]), arn.ncv, (*C.float)(&arn.v[0]), arn.ldv, &arn.iparam[0], &arn.ipntr[0], (*C.float)(&arn.workd[0]),
		(*C.float)(&arn.workl[0]), arn.lworkl, arn.info)

	nevReturned := arn.iparam[4]

	values := make([]complex128, nevReturned)
	vectors := make([][]complex128, nevReturned)
	for i := C.int(0); i < nevReturned; i++ {
		values[i] = complex128(complex(dr[i], di[i]))
		vectors[i] = make([]complex128, arn.n)
	}

	for i := C.int(0); i < nevReturned; i++ {
		if di[i] == 0 {
			for j := C.int(0); j < arn.n; j++ {
				vectors[i][j] = complex128(complex(arn.v[arn.ldv*i+j], 0))
			}
		} else {
			for j := C.int(0); j < arn.n; j++ {
				vectors[i][j] = complex128(complex(arn.v[arn.ldv*i+j], arn.v[arn.ldv*(i+1)+j]))
				vectors[i+1][j] = complex128(complex(arn.v[arn.ldv*i+j], -arn.v[arn.ldv*(i+1)+j]))
			}
			i++
		}
	}

	return values, vectors

}

func (arn arnoldiS) iterateInfo() (int, string) {
	return int(*arn.info), iterateInfoDescription[*arn.info]
}

func (arn arnoldiS) extractInfo() (int, string) {
	return int(*arn.info), extractInfoDescription[*arn.info]
}
