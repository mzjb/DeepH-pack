# OpenMX
Install [OpenMX package version 3.9](http://www.openmx-square.org/download.html) for density functional theory Hamiltonian matrix calculation to construct datasets.

If you are using Intel MKL and Intel MPI environments, you can use the following variable definitions for makefile
```
CC = mpiicc -O3 -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include/fftw -I${MKLROOT}/include
FC = mpiifort -O3 -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include
LIB = ${CMPLR_ROOT}/linux/compiler/lib/intel64_lin/libiomp5.a ${MKLROOT}/lib/intel64/libmkl_blas95_lp64.a ${MKLROOT}/lib/intel64/libmkl_lapack95_lp64.a ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group ${CMPLR_ROOT}/linux/compiler/lib/intel64_lin/libifcoremt.a -lpthread -lm -ldl
```
Or edit the makefile yourself according to your environment to install OpenMX version 3.9.

# 'overlap only' OpenMX
A modified OpenMX package is also used to compute overlap matrices only for large-scale materials structure. Install 'overlap only' OpenMX according to the *readme* document in this [repository](https://github.com/mzjb/overlap-only-OpenMX).