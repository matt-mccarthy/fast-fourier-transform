#include <cppunit/TestCase.h>
#include <cppunit/TestFixture.h>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/XmlOutputter.h>

#include "../fast-fourier.h"

using fast_fourier::discrete_fourier_transform;
using fast_fourier::fast_fourier_transform;

class fft_test : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(fft_test);

	CPPUNIT_TEST_SUITE_END();

	protected:
		__host__
		void test_dft1()
		{
			int n(4);
			cuFloatComplex	input[] = {1,1,1,1};
			cuFloatComplex	e[] = {4,0,0,0};
			cuFloatComplex*	a(discrete_fourier_transform(input, n));

			for (int j(0); j < n ; j++)
			{
				CPPUNIT_ASSERT_EQUAL_MESSAGE("Unequal Real",
					cuCrealf(e[j]), cuCrealf(a[j]));
				CPPUNIT_ASSERT_EQUAL_MESSAGE("Unequal Imaginary",
					cuCimagf(e[j]), cuCimagf(a[j]));
			}
		}
};
