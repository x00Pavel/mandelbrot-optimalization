/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	// @TODO allocate & prefill memory
    data = (int *)(malloc(height * width * sizeof(int)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
    free(data);
    data = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
    int *pdata = data;
    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (i; i < limit; ++i) {
            
				float x = x_start + j * dx;  // current real value
				float y = y_start + i * dy;  // current imaginary value
				float zReal = x;
				float zImag = y;
				int i = 0;

				float r2 = zReal * zReal;
				float i2 = zImag * zImag;

				if (r2 + i2 > 4.0f)
					break;

				zImag = 2.0f * zReal * zImag + y;
				zReal = r2 - i2 + x;
            }

            *(pdata++) = i;
        }
    }
    return data;
}