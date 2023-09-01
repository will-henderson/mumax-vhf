#include "cholesky.h"
#include <stdio.h>
#include <lapacke.h>

int cholesky(int n, double* A){
    lapack_int lda = n;
    lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A, lda);
    return info;
}

int triinv(int n, double* A){
    lapack_int lda = n;
    lapack_int info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N', n, A, lda);
    return info;
}