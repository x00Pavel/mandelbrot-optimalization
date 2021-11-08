/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization
 * over small batches
 * @date DATE
 */

#include "BatchMandelCalculator.h"

#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

BatchMandelCalculator::BatchMandelCalculator(unsigned matrixBaseSize,
                                             unsigned limit)
    : BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator") {
    block_size = 256;
    data = (int *)(malloc(height * width * sizeof(int)));
    initial_real = (int *)(malloc(width * sizeof(int)));
    initial_imag = (int *)(malloc(height * sizeof(int)));
    current_real_vec = (float *)(malloc(block_size * sizeof(float)));
    current_img_vec = (float *)(malloc(block_size * sizeof(float)));
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
    for (i = 0; i < height; i++) {
        initial_imag[i] = y_start + i * dy;
    }
}

BatchMandelCalculator::~BatchMandelCalculator() {
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

int *BatchMandelCalculator::calculateMandelbrot() {
    int *pdata = data;


    // Split the number of rows of the C matrix into a few blocks
    for (size_t blockN = 0; blockN < height / block_size; blockN++) {
        // Split the number of cols of the C matrix into a few blocks
        for (size_t blockP = 0; blockP < width / block_size; blockP++) {
            // Split M into a few blocks.
                // Go over all rows in the result matrix
            for (size_t i = 0; i < block_size; i++) {
                const size_t iGlobal = blockN * block_size + i;
                bool gt_4 = false;
                for (int iter = 0; !gt_4 && (iter < limit); iter++) {
#pragma omp simd reduction(& : gt_4) simdlen(64) aligned(pdata : 64)
                    for (size_t j = 0; j < block_size; j++) {
                        const size_t jGlobal = blockP * block_size + j;
                        float current_real =
                            (iter == 0)
                                ? initial_real[jGlobal]
                                : current_real_vec[j];
                        float current_imag =
                            (iter == 0)
                                ? initial_imag[iGlobal]
                                : current_img_vec[j];

                        float r2 = current_real * current_real;
                        float i2 = current_imag * current_imag;

                        bool gt = (r2 + i2 > 4.0f) &
                                  pdata[iGlobal * height + jGlobal] == limit;
                        gt_4 &= gt;
                        
                        if (gt) {
                            pdata[iGlobal * height + jGlobal] = iter;
                        } else {
                            current_img_vec[j] =
                                2.0f * current_imag * current_real +
                                initial_imag[iGlobal];
                            current_real_vec[j] =
                                r2 - i2 + initial_real[jGlobal];
                        }
                        // a[iGlobal * width + kGlobal] *
                        // b[kGlobal * height + jGlobal];
                    }
                }
            }
        }  // block size over P
    }      // block size over N

    //     for (size_t blockN = 0; blockN < height; blockN++) {
    //         for (size_t blockP = 0; blockP < width / block_size; blockP++) {
    //             bool gt_4 = false;
    //             for (int iter = 0; !gt_4 && iter < limit; iter++) {
    // #pragma omp simd simdlen(64) aligned(pdata : 64) reduction(& : gt_4)
    //                 for (size_t j = 0; j < block_size; j++) {
    //                     const size_t jGlobal = blockP * block_size + j;

    //                     float current_real = (iter == 0) ?
    //                     initial_real[jGlobal]
    //                                                      :
    //                                                      current_real_vec[j];
    //                     float current_img =
    //                         (iter == 0) ? initial_imag[blockN] :
    //                         current_img_vec[j];

    //                     float r2 = current_real * current_real;
    //                     float i2 = current_img * current_img;
    //                     gt_4 &= (r2 + i2 > 4.0f);

    //                     if (r2 + i2 > 4.0f &&
    //                         pdata[blockN * width + jGlobal] == limit) {
    //                         pdata[blockN * width + jGlobal] = iter;
    //                     } else {
    //                         current_img_vec[j] = 2.0f * current_img *
    //                         current_real +
    //                                              initial_imag[blockN];
    //                         current_real_vec[j] = r2 - i2 +
    //                         initial_real[jGlobal];
    //                     }
    //                 }
    //             }

    //         }  // block size over P
    //     }
    return data;
    }