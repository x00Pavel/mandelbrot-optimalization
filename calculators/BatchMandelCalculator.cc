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
    block_size = 64;
    data = (int *)(malloc(height * width *
                          sizeof(int)));  // allocate aligned memory
    current_real_vec = (float *)(malloc(height * width * sizeof(float)));
    current_img_vec = (float *)(malloc(height * width * sizeof(float)));
    
    for (int i = 0; i < height; i++) {
        
        for (int j = 0; j < width; j++){
            data[i * width + j] = limit;
            current_real_vec[i * width + j] = x_start + j * dx;
            current_img_vec[i * width + j] = y_start + i *dy;
        }
    }
}

BatchMandelCalculator::~BatchMandelCalculator() {
    free(data);
    data = NULL;

    free(current_real_vec);
    current_real_vec = NULL;

    free(current_img_vec);
    current_img_vec = NULL;
}

int *BatchMandelCalculator::calculateMandelbrot() {
    int *pdata = data;

    for (size_t blockN = 0; blockN < height; blockN++) {
        float initial_imag = y_start + blockN * dy;
        for (size_t blockP = 0; blockP < width / block_size; blockP++) {
            bool gt_4 = true;
            for (int iter = 0; gt_4 && iter < limit; iter++) {
#pragma omp simd simdlen(64) aligned(pdata : 64) reduction(| : gt_4)
                for (size_t j = 0; j < block_size; j++) {
                    const size_t jGlobal = blockP * block_size + j;

                    float initial_real = x_start + jGlobal * dx;
                    float current_img =
                        current_img_vec[blockN * width + jGlobal];
                    float current_real =
                        current_real_vec[blockN * width + jGlobal];

                    float r2 = current_real * current_real;
                    float i2 = current_img * current_img;
                    gt_4 |= r2 + i2 > 4.0f;

                    if (r2 + i2 > 4.0f &&
                        pdata[blockN * width + jGlobal] == limit) {
                        pdata[blockN * width + jGlobal] = iter;
                    } else {
                        current_img_vec[blockN * width + jGlobal] =
                            2.0f * current_img * current_real + initial_imag;
                        current_real_vec[blockN * width + jGlobal] =
                            r2 - i2 + initial_real;
                    }
                }
            }

        }  // block size over P
    }
    return data;
}