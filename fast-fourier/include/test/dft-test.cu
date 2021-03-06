#include <iostream>

#include <thrust/complex.h>
#include "../fast-fourier.h"

using namespace std;
using namespace fast_fourier;

int main()
{
	cfloat	input[]		= {1,1,1,1};
	cfloat	expected[]	= {4,0,0,0};
	cfloat*	actual		= discrete_fourier_transform(input, 4);
	
	for (int j(0) ; j < 4 ; j++)
		cout << actual[j] << "\t" << expected[j] << endl;

	delete[] actual;

	return 0;
}
