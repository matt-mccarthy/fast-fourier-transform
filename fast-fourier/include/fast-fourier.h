#ifndef FAST_FOURIER_H
#define FAST_FOURIER_H

#include <thrust/complex.h>

namespace fast_fourier
{
	typedef thrust::complex<float> cfloat;

	/// Does fast fourier transform.
	/// @param x The input vector (MUTABLE!!!).
	/// @param y The output vector.
	/// @param n The size of the vectors.
	/// @param binary_stor A place to store binary representations.
	__host__ __device__
	void fast_fourier_transform(cfloat* x, cfloat* y, unsigned n, bool* binary_stor);
	/// Does parallel fast fourier transform.
	/// @param x The input vector (MUTABLE!!!).
	/// @param y The output vector.
	/// @param n The size of the vectors.
	/// @param blk_count The number of blocks you wish to spawn.
	/// @param thd_count The number of threads per block.
	/// @param binary_stor A place to store binary representations.
	__global__
	void fast_fourier_transform(cfloat* x, cfloat* y, unsigned n, int blk_count, int thd_count, bool* binary_stor);
	/// Does discrete fourier transform.
	/// @param x The input vector.
	/// @param n The size of the vector.
	/// @return A pointer to the output vector.
	__host__ __device__
	cfloat* discrete_fourier_transform(cfloat* x, unsigned n);
}

#endif
