#ifndef FAST_FOURIER_H
#define FAST_FOURIER_H

#include <cuComplex.h>

namespace fast_fourier
{
	__host__ __device__
	cuFloatComplex* fast_fourier_transform(cuFloatComplex* x, unsigned n);
	__host__ __device__
	cuFloatComplex* fast_fourier_transform(cuFloatComplex* x, unsigned n,
											unsigned p);
	__host__ __device__
	cuFloatComplex* discrete_fourier_transform(cuFloatComplex* x, unsigned n);
}

#endif
