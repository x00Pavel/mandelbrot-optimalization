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
    initial_real = (float *)(malloc(width * sizeof(float)));
    initial_imag = (float *)(malloc(height * sizeof(float)));
    current_real_vec = (float *)(malloc(width * sizeof(float)));
    current_img_vec = (float *)(malloc(width * sizeof(float)));
    int i;

#pragma omp simd linear(data : 1) simdlen(64)
    for (i = 0; i < height * width; i++) {
            data[i] = limit;
    }
#pragma omp simd linear(initial_real : 1) simdlen(64)
    for (i = 0; i < width; i++) {
        initial_real[i] = x_start + i * dx;
    }

#pragma omp simd linear(initial_imag : 1) simdlen(64)
    for (i = 0; i < height; i++){
        initial_imag[i] = y_start + i * dy;
    }
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;

    free(current_real_vec);
    current_real_vec = NULL;

    free(current_img_vec);
    current_img_vec = NULL;

    free(initial_imag);
    initial_imag = NULL;

    free(initial_real);
    initial_real = NULL;
}

int* LineMandelCalculator::calculateMandelbrot()
{
   	int *pdata = data;

    for (int i = 0; i < height; i++) {
        // this value is used in current iteration and do not change
        bool gt_4 = false;
        for (int iter = 0; !gt_4 && (iter < limit); iter++) {
#pragma omp simd reduction(& : gt_4) simdlen(64) aligned(pdata:64)
            for (int j = 0; j < width; j++) {
                float current_real =
                    (iter == 0) ? initial_real[j] : current_real_vec[j];
                float current_img =
                    (iter == 0) ? initial_imag[i] : current_img_vec[j];

                float i2 = current_img * current_img;
                float r2 = current_real * current_real;
                // if greater then 4 and cell is not set
                gt_4 &= ((r2 + i2) > 4.0f);

                // update values from the next iteration with values form
                // current iteration
                 if ((r2 + i2) > 4.0f && (pdata[i * width + j] == limit)) {
                        pdata[i * width + j] = iter;
                } else {
                    current_img_vec[j] =
                        2.0f * current_img * current_real + initial_imag[i];
                    current_real_vec[j] = r2 - i2 + initial_real[j];
                }
            }
        }
    }
    return data;
}
