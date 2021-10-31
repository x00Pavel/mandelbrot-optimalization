/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization
 * over lines
 * @date DATE
 */
#include "LineMandelCalculator.h"

#include <immintrin.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize,
                                           unsigned limit)
    : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator") {
    data = (int*)(malloc(height * width * sizeof(int))); // allocate aligned memory
    x_vec = (int *)(malloc(width * sizeof(int)));
    r2_vec = (float *)(malloc(width * sizeof(float)));
    i2_vec = (float *)(malloc(width * sizeof(float)));
    // initialise vector of x values as it is not change for each iteration 
    for (int k = 0; k < width; k++){
        x_vec[k] = x_start + k * dx;
    }

}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;
    free(x_vec);
    data = NULL;
}

int* LineMandelCalculator::calculateMandelbrot()
{
   	int *pdata = data;
    
	for (int i = 0; i < height; i++)
	{
        // this value is used in current iteration and do not change
		float y = y_start + i * dy; // current imaginary value
        
        bool less_then_4 = true;
        int iter = 0;

        for(; (less_then_4) && iter < limit; iter++){
            #pragma omp simd
            for (int j = 0; j < width; i++){
                
            }
            // do calculations
            // count second power of vectors
            // x_vec[] = x_vec[0:width] * x_vec[0:width];
             // reduction should be here?

            // how to do reduction?
            // less_then_4 = r2 + i2 > 4.0f ? false : true;
        }
        // store iter to matrix (or to store vector of iters)?

	}
	return data;
}
