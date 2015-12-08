#include <iostream>

#include <math_functions.h>

#include <thrust/complex.h>

#include "../fast-fourier.h"

using namespace std;
using namespace fast_fourier;

int main()
{
	int		n(16);
	int		num_blk(2), num_thd(4);
	cfloat	input[]		= {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	cfloat*	expected	= discrete_fourier_transform(input, n);

	cfloat*	d_input(nullptr);
	cfloat*	d_actual(nullptr);
	bool*	b_s(nullptr);
	cfloat	actual[n];

	// Allocate an input and output array on the GPU
	if (cudaMalloc( &d_input, sizeof(cfloat) * n ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Failed to allocate input: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}
	if (cudaMalloc( &d_actual, sizeof(cfloat) * n ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Failed to allocate output: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}
	if (cudaMalloc( &b_s, sizeof(bool) * ilogbf(n) * num_blk * num_thd ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Failed to allocate boolean storage: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}
	// Copy the input array to the GPU
	if (cudaMemcpy( d_input, input, sizeof(cfloat) * n, cudaMemcpyHostToDevice ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Input failed to copy: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}

	fast_fourier_transform<<<1,1>>>(d_input, d_actual, n, num_blk, num_thd, b_s);

	// Copy the output array from the GPU
	if (cudaMemcpy( actual, d_actual, sizeof(cfloat) * n, cudaMemcpyDeviceToHost ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Output failed to copy: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}

	for (int j(0) ; j < n ; j++)
		cout << actual[j] << "\t\t\t\t" << expected[j] << endl;

	delete[] expected;
	cudaFree( d_actual );
	cudaFree( d_input );

	return 0;
}
