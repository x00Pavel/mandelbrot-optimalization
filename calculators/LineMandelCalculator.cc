/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization
 * over lines
 * @date DATE
 */
#include "LineMandelCalculator.h"

#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize,
                                           unsigned limit)
    : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator") {
    data = (int*)(malloc(height * width * sizeof(int)));
    current_line = NULL;
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;
}

int* LineMandelCalculator::calculateMandelbrot() {
// @TODO implement the calculator & return array of integers
	int *pdata = data;
    for (int i = 0; i < height; i++)
	{
        // current_line = &data[i];
        float y = y_start + i * dy; // current img valu
        float imag  = y;

        #pragma omp simd
		for (int j = 0; j < width; j++)
		{
			float x = x_start + j * dx; // current real valu
			float real = x;
            int iter;
	
			for (iter = 0; iter < 100; iter++) {
                // have a vector of x's
                // have a vector of y's
                // have a vector of imag values
                // have a vector of real values
                // count vector of values with this r2 + i2 > 4.0f
                float r2 = x * x;
				float i2 = y * y;
				if (r2 + i2 > 4.0f)
				{
					break;
				}

				imag = 2.0f * real * imag + y;
				real = r2 - i2 + x;
            }

            *(pdata++) = iter;
        }
	}
    return data;
}
