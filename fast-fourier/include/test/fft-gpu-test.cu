#include <iostream>

#include <thrust/complex.h>
#include "../fast-fourier.h"

using namespace std;
using namespace fast_fourier;

__global__
void run_test(cfloat*& input, cfloat*& output, int n)
{
	fast_fourier_transform(input, output, n);
}

int main()
{
	cfloat	input[]		= {1,2,3,4,5,6,7,8};
	cfloat*	expected	= discrete_fourier_transform(input, 8);

	cfloat* d_input, *d_actual, *actual;

	if (cudaMalloc( &d_input, sizeof(cfloat) * 8 ) != cudaSuccess)
	{
		cout << "Failed to allocate input" << endl;
		return 1;
	}
	if (cudaMalloc( &d_actual, sizeof(cfloat) * 8 ) != cudaSuccess)
	{
		cout << "Failed to allocate output" << endl;
		return 1;
	}
	if (cudaMemcpy( d_input, input, sizeof(cfloat) * 8, cudaMemcpyHostToDevice ) != cudaSuccess)
	{
		cout << "Input failed to copy" << endl;
		return 1;
	}

	run_test<<<1,1>>>(d_input, d_actual, 8);

	actual		= new cfloat[8];
	cudaMemcpy( actual, d_actual, sizeof(cfloat) * 8, cudaMemcpyDeviceToHost );

	for (int j(0) ; j < 8 ; j++)
		cout << actual[j] << "\t\t\t\t" << expected[j] << endl;

	delete[] actual;
	cudaFree( d_actual );
	cudaFree( d_input );

	return 0;
}
