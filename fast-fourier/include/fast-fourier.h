#ifndef FAST_FOURIER_H
#define FAST_FOURIER_H

#include <thrust/complex.h>

namespace fast_fourier
{
	typedef thrust::complex<float> cfloat;

	__host__ __device__
	void fast_fourier_transform(cfloat* x, cfloat* y, unsigned n);
	__host__ __device__
	void fast_fourier_transform(cfloat* x, cfloat* y, unsigned n, unsigned p);
	__host__ __device__
	cfloat* discrete_fourier_transform(cfloat* x, unsigned n);
}

#endif
