#include <iostream>

#include <thrust/complex.h>
#include "../fast-fourier.h"

using namespace std;
using namespace fast_fourier;

int main()
{
	cfloat	input[]		= {1,2,3,4,5,6,7,8};
	cfloat*	expected	= discrete_fourier_transform(input, 8);
	cfloat*	actual		= fast_fourier_transform(input, 8);

	for (int j(0) ; j < 8 ; j++)
		cout << actual[j] << "\t" << expected[j] << endl;

	delete[] actual;

	return 0;
}
