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
    current_real_vec = (float *)(_mm_malloc(
        block_size * block_size * sizeof(float), block_size));
    current_img_vec = (float *)(_mm_malloc(
        block_size * block_size * sizeof(float), block_size));

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

}

int *BatchMandelCalculator::calculateMandelbrot() {
    // int *pdata = data;
    // Split the number of rows of the C matrix into a few blocks
    for (size_t blockN = 0; blockN < height / block_size; blockN++) {
        // Split the number of cols of the C matrix into a few blocks
        for (size_t blockP = 0; blockP < width / block_size; blockP++) {
            // Initialize init values for next block

#pragma omp simd simdlen(128) linear(initial_real : 1) linear(initial_imag : 1)
            for (int k_line = 0; k_line < block_size; k_line++) {
                for (int k_row = 0; k_row < block_size; k_row++) {
                    initial_real[k_line * block_size + k_row] =
                        x_start + (blockP * block_size + k_row) * dx;
                    initial_imag[k_line * block_size + k_row] =
                        y_start + (blockN * block_size  + k_line) * dy;
                }
            }
            int *pdata = &data[blockN * block_size * width +
                              blockP * block_size];

            for (int i = 0; i < block_size; i++) {
                const int iGlobal = blockN * block_size + i;

                bool gt_4 = false;

                for (int iter = 0; !gt_4 && (iter < limit); iter++) {
#pragma omp simd reduction(& : gt_4) simdlen(128) aligned(pdata : 128) linear(update:1)
                    for (int j = 0; j < block_size; j++) {
                        const int jGlobal = blockP * block_size + j;
                        const int index = i * block_size + j;
                        // const int indexGlobal = iGlobal * width + jGlobal;

                        float current_real = (iter == 0)
                                                 ? initial_real[index]
                                                 : current_real_vec[index];
                        float current_imag = (iter == 0)
                                                 ? initial_imag[index]
                                                 : current_img_vec[index];

                        float r2 = current_real * current_real;
                        float i2 = current_imag * current_imag;

                        gt_4 &= (r2 + i2 > 4.0f);

                        if ((r2 + i2 > 4.0f) && pdata[index] == limit) {
                            pdata[index] = iter;
                        }
                        else{
                            current_img_vec[index] =
                                2.0f * current_imag * current_real +
                                initial_imag[index];
                            current_real_vec[index] =
                                r2 - i2 + initial_real[index];
                        }
                    }
                }
            }
        }
    }
    return data;
}