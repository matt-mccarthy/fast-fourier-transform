// This program takes n, block count, thread count, and number of trials as parameters.
// It is assumed that these are all powers of 2.
// Compiles with nvcc -std=c++11 -rdc=true -arch=compute_50 -code=sm_50
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <utility>

#include "include/fast-fourier.h"

using namespace std;
using namespace chrono;
using namespace fast_fourier;

void	gen_array(cfloat* output, int n);
long double	sum(long double* in, int n);
long double	std_dev(long double* in, int n, long double sum);

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cerr << "Usage is " << argv[0] << " n num_trials num_blk num_thd" << endl;
		return 1;
	}

	int		n(atoi(argv[1]));
	int		trial_count(atoi(argv[2]));
	int		num_blk(atoi(argv[3]));
	int		num_thd(atoi(argv[4]));

	cfloat*	input(new cfloat[n]);
	cfloat* output(new cfloat[n]);
	cfloat* d_input(nullptr);
	cfloat* d_output(nullptr);
	bool*	binary_stor(nullptr);

	long double	times[trial_count];
	high_resolution_clock::time_point tp2, tp1;
	duration<long double> time_span;

	// Allocate device arrays
	if (cudaMalloc( &d_input, sizeof(cfloat) * n ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Failed to allocate input: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}
	if (cudaMalloc( &d_output, sizeof(cfloat) * n ) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Failed to allocate output: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}
	if (cudaMalloc( &binary_stor, sizeof(bool) * ilogbf(n) * num_thd * num_blk) != cudaSuccess)
	{
		auto t = cudaGetLastError();
		cout << "Failed to allocate boolean storage: "
			<< cudaGetErrorName(t) << ", "
			<< cudaGetErrorString(t) << endl;
		return 1;
	}

	// Run experiment
	for (int j(0) ; j < trial_count ; j++)
	{
		// Generate random input
		gen_array(input, n);

		// Run the test
		tp1 = system_clock::now();
		// Copy the input array to the GPU
		if (cudaMemcpy( d_input, input, (long) n * sizeof(cfloat), cudaMemcpyHostToDevice ) != cudaSuccess)
		{
			auto t = cudaGetLastError();
			cout << "Iteration: " << j
				<< " Input failed to copy: "
				<< cudaGetErrorName(t) << ", "
				<< cudaGetErrorString(t) << endl;
			return 1;
		}
		fast_fourier_transform<<<1,1>>>(d_input, d_output, n, num_blk, num_thd, binary_stor);
		if (cudaMemcpy( output, d_output, (long) n * sizeof(cfloat), cudaMemcpyDeviceToHost ) != cudaSuccess)
		{
			auto t = cudaGetLastError();
			cout << "Iteration: " << j
				<< " Output failed to copy: "
				<< cudaGetErrorName(t) << ", "
				<< cudaGetErrorString(t) << endl;
			return 1;
		}
		tp2 = system_clock::now();

		time_span	= duration_cast< duration<long double> >(tp2 - tp1)*1000.0;
		times[j]	= time_span.count();
	}

	// Calculate statistics
	long double av(sum(times, trial_count));
	long double sd(std_dev(times, trial_count, av));
	av /= (long double)n;

	cout << av << "\t" << sd << endl;

	cudaFree( binary_stor );
	cudaFree( d_input );
	cudaFree( d_output );
	return 0;
}

void gen_array(cfloat* output, int n)
{
	srand(time(nullptr));

	for (int j = 0; j < n; j++)
         output[j] = cfloat(rand(), rand());
}

long double	sum(long double* in, int n)
{
	long double s(0.0);

	for (int j(0) ; j < n ; j++)
		s += in[j];

	return s;
}

long double	std_dev(long double* in, int n, long double sum)
{
	long double var = 0;
	long double tmp = 0;

	for (int i = 0 ; i < n ; i++)
	{
		tmp = (n * in[i] - sum);
		var += tmp * tmp;
	}

	long double stdDev = sqrt(var/n) / n;

	return stdDev;
}
