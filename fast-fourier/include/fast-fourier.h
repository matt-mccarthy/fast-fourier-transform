#ifndef FAST_FOURIER_H
#define FAST_FOURIER_H

#include <cuComplex.h>

namespace fast_fourier
{
	cuFloatComplex* fast_fourier_transform(cuFloatComplex* x, unsigned n,
											unsigned p);
}

#endif
