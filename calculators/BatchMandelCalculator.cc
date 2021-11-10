/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization
 * over small batches
 * @date DATE
 */

#include "BatchMandelCalculator.h"

#include <immintrin.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

BatchMandelCalculator::BatchMandelCalculator(unsigned matrixBaseSize,
                                             unsigned limit)
    : BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator") {
    const int block_size = 64;
    data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
    initial_real =
        (float *)(_mm_malloc(block_size * sizeof(float), 64));
    initial_imag =
        (float *)(_mm_malloc(block_size * sizeof(float), 64));

    current_real_vec =
        (float *)(_mm_malloc(block_size * sizeof(float), 64));
    current_img_vec =
        (float *)(_mm_malloc(block_size * sizeof(float), 64));

#pragma omp simd linear(data : 1) simdlen(64)
    for (int i = 0; i < height * width; i++) {
        data[i] = limit;
    }
}

BatchMandelCalculator::~BatchMandelCalculator() {
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

int *BatchMandelCalculator::calculateMandelbrot() {
    constexpr int blockSize = 64;
    for (int blockN = 0; blockN < height; blockN++) {
        for (int blockP = 0; blockP < width / blockSize; blockP++) {
            const float imag = y_start + blockN * dy;

#pragma omp simd simdlen(64)
            for (int k = 0; k < blockSize; k++) {
                current_real_vec[k] = x_start + (blockP * blockSize + k) * dx;
                initial_real[k] = x_start + (blockP * blockSize + k) * dx;
                current_img_vec[k] = imag;
                initial_imag[k] = imag;
            }

            bool gt_4 = false;
            int *pdata = &data[blockN * width + blockP * blockSize];
            for (int iter = 0; !gt_4 && (iter < limit); iter++) {
#pragma omp simd reduction(& : gt_4) simdlen(64) aligned(pdata : 64) lastprivate(pdata, current_img_vec, current_real_vec)
                for (int j = 0; j < blockSize; j++) {
                    const float init_real = initial_real[j];
                    const float current_imag = current_img_vec[j];
                    const float current_real = current_real_vec[j];
                    const float init_imag = initial_imag[j];
                    const float r2 = current_real * current_real;
                    const float i2 = current_imag * current_imag;

                    gt_4 &= (r2 + i2 > 4.0f);

                    if ((r2 + i2 > 4.0f) && (pdata[j] == limit)) {
                        pdata[j] = iter;
                    } else {
                        current_img_vec[j] =
                            2.0f * current_imag * current_real + imag;
                        current_real_vec[j] = r2 - i2 + init_real;
                    }
                }
            }
        }
    }
    return data;
}