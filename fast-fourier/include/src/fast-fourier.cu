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

__host__ __device__
void	binary_inc(bool* i, int lg_n);
__host__ __device__
int		bin2dec(const bool* i, int lg_n);
__host__ __device__
void	dec2bin(bool* i, int l, int lg_n);
__host__ __device__
int		wierd_bin_thingy(const bool* l, int lg_n, int m);
__host__ __device__
cfloat	k_root_unity(int k, int n);

__global__
void transformer(cfloat* r, cfloat* s, unsigned lg_n, unsigned blk_off,
					unsigned thd_off, unsigned n, int m);
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

void fast_fourier::fast_fourier_transform(cfloat* x, cfloat* y, unsigned n)
{
	cfloat*	s(y);
	cfloat*	r(x);
	cfloat*	tmp_ptr;
	int		lg_n(ilogbf(n));
	int		j,k,u_exp;
	bool*	l_bi(new bool[lg_n]);
	bool	tmp(false);

	// for (int j(0) ; j < n ; j++)
	// 	r[j] = x[j];

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

	// for (int j(0) ; j < n ; j++)
	// 	y[j] = r[j];

	delete[] l_bi;
}

__global__
void fast_fourier::fast_fourier_transform(cfloat* x, cfloat* y, unsigned n,
											int blk_count, int thd_count)
{
	int		lg_n(ilogbf(n));

	int		blk_off = n / blk_count;
	int		thd_off	= blk_off / thd_count;

	cfloat* r(x);
	cfloat* s(y);
	cfloat*	tmp_ptr;

	// Copy x into r
	// parallel_copy<<<blk_count, thd_count>>>(x, r, n, blk_off, thd_off);
	// cudaDeviceSynchronize();

	for (int m(0) ; m < lg_n ; m++)
	{
		// Swap s and r so the last output becomes the new input
		tmp_ptr = r;
		r		= s;
		s		= tmp_ptr;

		// Perform the next step of the transform
		transformer<<<blk_count, thd_count>>>(r, s, lg_n, blk_off, thd_off, n, m);
		cudaDeviceSynchronize();
	}

	// Copy r into y
	// parallel_copy<<<blk_count, thd_count>>>(r, y, n, blk_off, thd_off);
	// cudaDeviceSynchronize();

	// delete[] r, s;
}

__global__
void transformer(cfloat* r, cfloat* s, unsigned lg_n, unsigned blk_off,
					unsigned thd_off, unsigned n, int m)
{
	int		l_min( blockIdx.x * blk_off + threadIdx.x * thd_off );
	int		l_max( l_min + thd_off );
	int		j, k, u_exp;

	bool*	l_bi(new bool[lg_n]);
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

	delete[] l_bi;
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
