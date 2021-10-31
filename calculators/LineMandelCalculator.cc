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

#pragma omp declare simd
template <typename T>
static inline int mandel(T real, T imag, int limit) {
    T zReal = real;
    T zImag = imag;
    int iters;
    for (iters = 0; ((zReal * zReal + zImag * zImag) > 4.0f) && (iters < limit);
         iters++) {
        T r2 = zReal * zReal;
        T i2 = zImag * zImag;
        zImag = 2.0f * zReal * zImag + imag;
        zReal = r2 - i2 + real;
    }

    return iters;
}

int* LineMandelCalculator::calculateMandelbrot() {
// @TODO implement the calculator & return array of integers
	int *pdata = data;
    #pragma omp parallel
    for (int i = 0; i < height; i++)
	{
        float y = y_start + i * dy; // current img value
        
        #pragma omp simd
		for (int j = 0; j < width; j++)
		{
			float x = x_start + j * dx; // current real value
            *(pdata++) = mandel(x, y, limit);
        }
	}
    return data;
}
