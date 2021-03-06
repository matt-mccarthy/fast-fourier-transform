// Compiles with nvcc -std=c++11 -rdc=true -arch=compute_50 -code=sm_50
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <math_constants.h>
#include <math_functions.h>

#include "../fast-fourier.h"

using thrust::copy_n;
using thrust::exp;
using thrust::fill_n;

using fast_fourier::cfloat;

/// Increments a number represented in binary.
/// @param i Our binary representation.
/// @param lg_n The size of the bool vector (it had better be lg n!).
__host__ __device__
void	binary_inc(bool* i, int lg_n);
/// Converts a binary representaion to an integer.
/// @param i Our binary representation.
/// @param lg_n The size of the bool vector (it had better be lg n!).
/// @return The decimal value
__host__ __device__
int		bin2dec(const bool* i, int lg_n);
/// Writes a binary representation to a bit vector.
/// @param i Our binary representation.
/// @param l The int to convert.
/// @param lg_n The size of the bool vector (it had better be lg n!).
__host__ __device__
void	dec2bin(bool* i, int l, int lg_n);
/// Our wierd binary exponent.
/// @param l Our binary representation.
/// @param lg_n The size of the bool vector (it had better be lg n!).
/// @param m The current iteration.
__host__ __device__
int		wierd_bin_thingy(const bool* l, int lg_n, int m);
/// Returns the nth root of unity raised to the k
__host__ __device__
cfloat	k_root_unity(int k, int n);

/// The inner loop of FFT
/// @param r The vector to which we write.
/// @param s The vector from which we read.
/// @param lg_n lg n.
/// @param blk_off The block offset (used for reading from binary_stor).
/// @param thd_off The thread offset (used for reading from binary_stor).
/// @param n The size of our I/O vectors.
/// @param m The current iteration.
/// @param binary_stor our bit vector where we store binary representations.
/// @param thd_count The number of threads per block.
__global__
void transformer(cfloat* r, cfloat* s, unsigned lg_n, unsigned blk_off,
					unsigned thd_off, unsigned n, int m, bool* binary_stor, int thd_count);
/// A parallel copy algorithm.
/// @param src The source.
/// @param dst The destination.
/// @param n The size of the vectors.
/// @param blk_off The number of things each block gets.
/// @param thd_off The number of things each thread gets.
__global__
void parallel_copy(const cfloat* src, cfloat* dst, unsigned n, unsigned blk_off,
					unsigned thd_off);

cfloat* fast_fourier::discrete_fourier_transform(cfloat* x,	unsigned n)
{
	cfloat* y(new cfloat[n]);

	fill_n(y, n, cfloat(0.0f));

	for (int j(0) ; j < n ; j++)
		for (int k(0) ; k < n ; k++)
			y[j] += x[k] * k_root_unity(k*j, n);

	return y;
}

void fast_fourier::fast_fourier_transform(cfloat* x, cfloat* y, unsigned n,
											bool* binary_stor)
{
	cfloat*	s(x);
	cfloat*	r(y);
	cfloat*	tmp_ptr;
	int		lg_n(ilogbf(n));
	int		j,k,u_exp;
	bool*	l_bi(binary_stor);
	bool	tmp(false);

	for (int j(0) ; j < lg_n ; j++ )
		l_bi[j] = false;

	for (int m(0) ; m < lg_n ; m++)
	{
		tmp_ptr = s;
		s = r;
		r = tmp_ptr;

		for (int l(0) ; l < n ; l++)
		{
			tmp = l_bi[m];
			l_bi[m] = false;

			j = bin2dec(l_bi, lg_n);
			k = j + (int)exp2f(lg_n - m - 1);

			l_bi[m] = tmp;

			u_exp = wierd_bin_thingy(l_bi, lg_n, m);
			r[l] = s[j] + s[k] * k_root_unity(u_exp, n);

			binary_inc(l_bi, lg_n);
		}
	}

	for (int j(0) ; j < n ; j++)
		y[j] = r[j];
}

__global__
void fast_fourier::fast_fourier_transform(cfloat* x, cfloat* y, unsigned n,
											int blk_count, int thd_count, bool* binary_stor)
{
	int		lg_n(ilogbf(n));

	int		blk_off = n / blk_count;
	int		thd_off	= blk_off / thd_count;

	cfloat *r(x), *s(y);
	cfloat*	tmp_ptr;

	for (int m(0) ; m < lg_n ; m++)
	{
		// Swap s and r so the last output becomes the new input
		tmp_ptr = r;
		r		= s;
		s		= tmp_ptr;

		// Perform the next step of the transform
		transformer<<<blk_count, thd_count>>>(r, s, lg_n, blk_off, thd_off, n, m, binary_stor, thd_count);
		cudaDeviceSynchronize();
	}

	// Copy r into y
	parallel_copy<<<blk_count, thd_count>>>(r, y, n, blk_off, thd_off);
	cudaDeviceSynchronize();

	// delete[] r, s;
}

__global__
void transformer(cfloat* r, cfloat* s, unsigned lg_n, unsigned blk_off,
					unsigned thd_off, unsigned n, int m, bool* binary_stor, int thd_count)
{
	int		l_min( blockIdx.x * blk_off + threadIdx.x * thd_off );
	int		l_max( l_min + thd_off );
	int		j, k, u_exp;

	// bool*	l_bi(new bool[lg_n]);
	bool*	l_bi(binary_stor + (blockIdx.x * thd_count + threadIdx.x) * lg_n);
	bool	tmp;

	dec2bin(l_bi, l_min, lg_n);

	for (int l(l_min) ; l < l_max ; l++)
	{
		tmp = l_bi[m];

		l_bi[m]	= false;
		j		= bin2dec(l_bi, lg_n);
		k		= j + (int)exp2f(lg_n - m - 1);
		l_bi[m]	= tmp;

		u_exp	= wierd_bin_thingy(l_bi, lg_n, m);
		r[l]	= s[j] + s[k] * k_root_unity(u_exp, n);

		binary_inc(l_bi, lg_n);
	}

	// delete[] l_bi;
}

__global__
void parallel_copy(const cfloat* src, cfloat* dst, unsigned n, unsigned blk_off,
					unsigned thd_off)
{
	int		l_min( blockIdx.x * blk_off + threadIdx.x * thd_off );
	int		l_max( l_min + thd_off );

	for (int l(l_min) ; l < l_max ; l++)
		dst[l] = src[l];
}

void binary_inc(bool* i, int lg_n)
{
	bool flag(true);

	for (int j(lg_n - 1) ; j > -1 && flag ; j--)
	{
		flag = i[j];
		i[j] = !flag;
	}
}

int bin2dec(const bool* i, int lg_n)
{
	int m(0), two_j(1);

	for (int j(lg_n - 1) ; j > -1 ; j--)
	{
		m += (int)i[j] * two_j;
		two_j *= 2;
	}

	return m;
}

int wierd_bin_thingy(const bool* l, int lg_n, int m)
{
	int exponent(0), two_j(1);

	for (int j(0) ; j < m + 1 ; j++)
	{
		exponent	+= l[j] * two_j;
		two_j		*= 2;
	}

	for (int j(m + 1) ; j < lg_n ; j++)
		exponent	*= 2;

	return exponent;
}

cfloat k_root_unity(int k, int n)
{
	float	e_img(2.0f * ((float)k) * CUDART_PI_F /((float) n));
	cfloat	exponent(cfloat(0,e_img));

	return exp(exponent);
}

void dec2bin(bool* i, int l, int lg_n)
{
	for (int j(lg_n - 1) ; j > -1 ; j--)
	{
		i[j]	= (bool) (l % 2);
		l		/= 2;
	}
}
