#include <iostream>

#include <thrust/complex.h>

#include "../fast-fourier.h"

using namespace std;
using namespace fast_fourier;

__global__
void run_test(cfloat* input, cfloat* output, int n)
{
	fast_fourier_transform(input, output, n);
}

int main()
{
	int		n(8);
	cfloat	input[]		= {1,2,3,4,5,6,7,8};
	cfloat*	expected	= discrete_fourier_transform(input, n);

	cfloat*	d_input(nullptr);
	cfloat*	d_actual(nullptr);
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
	// Copy the input array to the GPU
	if (cudaMemcpy( d_input, input, sizeof(cfloat) * n, cudaMemcpyHostToDevice ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Input failed to copy: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}

	run_test<<<1,1>>>(d_input, d_actual, n);

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

	cudaFree( d_actual );
	cudaFree( d_input );

	return 0;
}
