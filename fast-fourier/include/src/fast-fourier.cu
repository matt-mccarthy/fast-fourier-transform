#include <thrust/fill.h>
#include <thrust/copy.h>
#include <math_constants.h>
#include <math_functions.h>

#include "../fast-fourier.h"

using thrust::copy_n;
using thrust::fill_n;

__host__ __device__
void	binary_inc(bool* i, int lg_n);
__host__ __device__
int		bin2dec(const bool* i, int lg_n);
__host__ __device__
int		wierd_bin_thingy(const bool* l, int lg_n, int m);
__host__ __device__
cfloat	k_root_unity(int k, int n);

cfloat* fast_fourier::discrete_fourier_transform(cfloat* x,	unsigned n)
{
	cfloat* y(new cfloat[n]);

	fill_n(y, n, make_cuFloatComplex(0.0f,0.0f));

	for (int j(0) ; j < n ; j++)
		for (int k(0) ; k < n ; k++)
			y[j] = cuCaddf( y[j], cuCmulf(x[k], k_root_unity(k*j, n)) );

	return y;
}

cuFloatComplex* fast_fourier::fast_fourier_transform(cuFloatComplex* x,
														unsigned n, unsigned p)
{
	cuFloatComplex* y(new cuFloatComplex[n]);
	return y;
}

cuFloatComplex* fast_fourier::fast_fourier_transform(cuFloatComplex* x,
														unsigned n)
{
	cuFloatComplex*	s(new cuFloatComplex[1]);
	cuFloatComplex*	r(new cuFloatComplex[n]);
	int				lg_n(ilogbf(n));
	int				j,k,u_exp;
	bool*			l_bi(new bool[lg_n]);
	bool			tmp(false);

	for (int j(0) ; j < n ; j++)
		r[j] = x[j];

	for (int m(0) ; m < lg_n ; m++)
	{
		delete[] s;
		s = r;
		r = new cuFloatComplex[n];

		fill_n(l_bi, lg_n, false);

		for (int l(0) ; l < n ; l++)
		{
			tmp = l_bi[l];
			l_bi[l] = 0;

			j = bin2dec(l_bi, lg_n);
			k = j + (int)exp2f(l);

			l_bi[l] = tmp;

			u_exp = wierd_bin_thingy(l_bi, lg_n, m);
			r[l] = cuCaddf(s[j], cuCmulf(s[k], k_root_unity(u_exp, n)) );

			binary_inc(l_bi, lg_n);
		}
	}

	return r;
}

void binary_inc(bool* i, int lg_n)
{
	bool flag(true);

	for (int j(0) ; j < lg_n && flag ; j++)
	{
		flag = i[j];
		i[j] = !flag;
	}
}

int bin2dec(const bool* i, int lg_n)
{
	int m(0), two_j(1);

	for (int j(0) ; j < lg_n ; j++)
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
		exponent	+= (int)l[m - j] * two_j;
		two_j		*= 2;
	}

	return exponent * (int)exp2f(lg_n - 1 - m);
}

cuFloatComplex	k_root_unity(int k, int n)
{
	float exponent(2.0f * ((float)k) * CUDART_PI_F /((float) n));
	float sin_exp, cos_exp;

	sincosf(exponent, &sin_exp, &cos_exp);

	return make_cuFloatComplex(cos_exp, sin_exp);
}
