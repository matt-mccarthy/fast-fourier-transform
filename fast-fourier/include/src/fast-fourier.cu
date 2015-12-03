#include <algorithm>

#include <math_constants.h>
#include <math_functions.h>

#include "../fast-fourier.h"

using std::copy_n;
using std::fill_n;

void			binary_inc(bool* i, int lg_n);
int				bin2dec(const bool* i, int lg_n);
int				wierd_bin_thingy(const bool* l, int lg_n, int m);
cuFloatComplex	k_root_unity(int k, int n);

cuFloatComplex* fast_fourier::fast_fourier_transform(cuFloatComplex* x,
														unsigned n, unsigned p)
{
	cuFloatComplex* y(new cuFloatComplex[n]);
	return nullptr;
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

	copy_n(x, n, r);

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
