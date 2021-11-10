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
    data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
initial_real =
        (float *)(_mm_malloc(width * sizeof(float), 64));
    initial_imag =
        (float *)(_mm_malloc(height * sizeof(float), 64));

    current_real_vec =
        (float *)(_mm_malloc(width * sizeof(float), 64));
    current_img_vec =
        (float *)(_mm_malloc(width * sizeof(float), 64));
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
    _mm_free(data);
    data = NULL;

    _mm_free(current_real_vec);
    current_real_vec = NULL;

    _mm_free(current_img_vec);
    current_img_vec = NULL;

    _mm_free(initial_imag);
    initial_imag = NULL;

    _mm_free(initial_real);
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
                const float current_real =
                    (iter == 0) ? initial_real[j] : current_real_vec[j];
                const float current_img =
                    (iter == 0) ? initial_imag[i] : current_img_vec[j];

                const float i2 = current_img * current_img;
                const float r2 = current_real * current_real;
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
