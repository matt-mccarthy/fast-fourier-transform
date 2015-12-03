#ifndef FAST_FOURIER_H
#define FAST_FOURIER_H

#include <thrust/complex.h>

namespace fast_fourier
{
	typedef thrust::complex<double> cfloat;

	__host__// __device__
	cfloat* fast_fourier_transform(cfloat* x, unsigned n);
	__host__// __device__
	cfloat* fast_fourier_transform(cfloat* x, unsigned n, unsigned p);
	__host__// __device__
	cfloat* discrete_fourier_transform(cfloat* x, unsigned n);
	__host__
	void	test();
}

#endif
