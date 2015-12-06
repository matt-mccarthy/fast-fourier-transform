#include <thrust/fill.h>
#include <thrust/copy.h>
#include <math_constants.h>
#include <math_functions.h>

#include "../fast-fourier.h"

#include <iostream>
using std::cout;
using std::endl;

using thrust::copy_n;
using thrust::exp;
using thrust::fill_n;

using fast_fourier::cfloat;

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

	fill_n(y, n, cfloat(0.0f));

	for (int j(0) ; j < n ; j++)
		for (int k(0) ; k < n ; k++)
			y[j] += x[k] * k_root_unity(k*j, n);

	return y;
}

void fast_fourier::fast_fourier_transform(cfloat* x, cfloat* y, unsigned n, unsigned p)
{

}

void fast_fourier::fast_fourier_transform(cfloat* x, cfloat* y, unsigned n)
{
	cfloat*	s(new cfloat[n]);
	cfloat*	r(new cfloat[n]);
	cfloat*	tmp_ptr;
	int		lg_n(ilogbf(n));
	int		j,k,u_exp;
	bool*	l_bi(new bool[lg_n]);
	bool	tmp(false);

	for (int j(0) ; j < n ; j++)
		r[j] = x[j];

	fill_n(l_bi, lg_n, false);

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

cfloat	k_root_unity(int k, int n)
{
	float exponent(2.0f * k * CUDART_PI_F /((float) n));

	return exp(cfloat(0, exponent));
}

void fast_fourier::test()
{
	cout << "bininc" << endl;

	bool*	l_bi(new bool[4]);
	fill_n(l_bi, 4, false);

	for (int j(0) ; j < 16 ; j++)
	{
		for (int k(0) ; k < 4 ; k++)
			cout << l_bi[k];

		cout << "\t" << bin2dec(l_bi, 4) << endl;

		binary_inc(l_bi, 4);
	}

	for (int k(0) ; k < 4 ; k++)
		cout << l_bi[k];
	cout << "\t" << bin2dec(l_bi, 4) << endl;
}
