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
    block_size = 128;
    data = (int *)(_mm_malloc(height * width * sizeof(int), block_size));
    initial_real = (float *)(_mm_malloc(block_size * block_size * sizeof(float),
                                        block_size));
    initial_imag = (float *)(_mm_malloc(block_size * block_size * sizeof(float),
                                        block_size));
    // current_real_vec = (float *)(_mm_malloc(
    //     block_size * block_size * sizeof(float), block_size));
    // current_img_vec = (float *)(_mm_malloc(
    //     block_size * block_size * sizeof(float), block_size));
    current_real_vec =
        (float *)(_mm_malloc(block_size * sizeof(float), block_size));
    current_img_vec =
        (float *)(_mm_malloc(block_size * sizeof(float), block_size));
    line_done = (float *)(_mm_malloc(block_size * sizeof(float), block_size));
    int i;

#pragma omp simd linear(data : 1) simdlen(128)
    for (i = 0; i < height * width; i++) {
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

    _mm_free(line_done);
    line_done = NULL;

}

int *BatchMandelCalculator::calculateMandelbrot() {
    int *pdata = data;
    // Split the number of rows of the C matrix into a few blocks
    for (size_t blockN = 0; blockN < height; blockN++) {
        // Split the number of cols of the C matrix into a few blocks
        for (size_t blockP = 0; blockP < width / block_size; blockP++) {
            const float imag = y_start + blockN * dy;
            bool gt_4 = false;

#pragma omp simd simdlen(128)
            for (int k = 0; k < block_size; k++) {
                current_real_vec[k] = x_start + (blockP * block_size + k) * dx;
                initial_real[k] = x_start + (blockP * block_size + k) * dx;
                current_img_vec[k] = imag;
                initial_imag[k] = imag;
                line_done[k] = limit;
            }

            for (int iter = 0; !gt_4 && (iter < limit); iter++) {
#pragma omp simd reduction(& : gt_4) simdlen(128) aligned(pdata : 128)
                for (int j = 0; j < block_size; j++) {
                    const float current_imag = current_img_vec[j];
                    const float current_real = current_real_vec[j];
                    const float init_real = initial_real[j];
                    const float init_imag = initial_imag[j];
                    const float r2 = current_real_vec[j] * current_real_vec[j];
                    const float i2 = current_img_vec[j] * current_img_vec[j];

                    gt_4 &= (r2 + i2 > 4.0f);

                    if (r2 + i2 > 4.0f && (line_done[j] == limit)) {
                        line_done[j] = iter;
                    } else {
                        current_img_vec[j] =
                            2.0f * current_imag * current_real + init_imag;
                        current_real_vec[j] = r2 - i2 + init_real;
                    }
                }
            }

#pragma omp simd simdlen(128) linear(pdata:1)
            for (int k = 0; k < block_size; k++) {               
                *(pdata++) = line_done[k];
            }
        }
    }
    return data;
}