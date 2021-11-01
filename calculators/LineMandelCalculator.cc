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
    initial_real_vec = (int *)(malloc(width * sizeof(int)));
    r2_vec = (float *)(malloc(width * sizeof(float)));
    i2_vec = (float *)(malloc(width * sizeof(float)));
    current_real_vec = (float *)(malloc(width * sizeof(float)));
    // initialise vector of x values as it is not change for each iteration 
    for (int k = 0; k < width; k++){
        initial_real_vec[k] = x_start + k * dx;
        current_real_vec[k] = initial_real_vec[k];
    }
    
    current_img_vec = (float *)(malloc(width * sizeof(float)));
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;

    free(initial_real_vec);
    initial_real_vec = NULL;
    
    free(current_real_vec);
    current_real_vec = NULL;
    
    free(current_img_vec);
    current_img_vec = NULL;
}

// pragma omp reduction(operation:value where result will be stored)
int* LineMandelCalculator::calculateMandelbrot()
{
   	int *pdata = data;
    
	for (int i = 0; i < height; i++)
	{
        // this value is used in current iteration and do not change
		float initial_imag = y_start + i * dy; // current imaginary value
        bool gt_4 = false;
        int iter;
        #pragma omp reduction(>:gt_4)
        for(int iter = 0; !gt_4 && iter < limit; iter++){
            #pragma omp simd
            for (int j = 0; j < width; j++) {
                // need current_real and current_img from previous iteration
                float current_img = current_img_vec[j];
                float current_real = current_real_vec[j];
                
                float i2 = current_img * current_img;
                float r2 = current_real * current_real;
                gt_4 = (r2 + i2) > 4.0f;

                // update values from the next iteration with values form current iteration
                current_img_vec[j] =
                    2.0f * current_img * current_real + initial_imag;
                current_real_vec[j] = r2 - i2 + initial_real_vec[j];
            }
        // store iter to matrix (or to store vector of iters)?
        }
        *(pdata++) = iter;
	}
	return data;
}
