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
    current_real_vec = (float *)(malloc(width * sizeof(float)));  
    current_img_vec = (float *)(malloc(width * sizeof(float)));
    for (int i = 0; i < width * height; i++){
        data[i] = limit;
    }
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;

    free(current_real_vec);
    current_real_vec = NULL;

    free(current_img_vec);
    current_img_vec = NULL;
    
}

void LineMandelCalculator::mandel(float initial_imag, int iter, int i, int *pdata){
    #pragma omp simd reduction(|: gt_4) simdlen(64)
	for (int j = 0; j < width; j++) {
	    float initial_real = x_start + j * dx;

        float current_img = (iter == 0) ? initial_imag : current_img_vec[j];
		float current_real = (iter == 0) ? initial_real : current_real_vec[j];

        float i2 = current_img * current_img;
        float r2 = current_real * current_real;
        // if greater then 4 and cell is not set
        bool gt = ((r2 + i2) > 4.0f) && (pdata[j + i * width] == limit);
        gt_4 |= gt;

        // update values from the next iteration with values form
        // current iteration
        if (gt) {
            pdata[i * width + j] = iter;
        } else {
            current_img_vec[j] =
                2.0f * current_img * current_real + initial_imag;
            current_real_vec[j] = r2 - i2 + initial_real;
        }
    }
}

// pragma omp reduction(operation:value where result will be stored)
int* LineMandelCalculator::calculateMandelbrot()
{
   	int *pdata = data;

    for (int i = 0; i < height; i++) {
        // this value is used in current iteration and do not change
        float initial_imag = y_start + i * dy;
        gt_4 = true;
        
        for (int iter = 0; gt_4 && (iter < limit); iter++) {
			mandel(initial_imag, iter, i, pdata);
        }
    }
    return data;
}
