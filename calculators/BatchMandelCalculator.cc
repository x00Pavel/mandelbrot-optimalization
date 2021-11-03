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
    data = (int *)(malloc(height * width *
                          sizeof(int)));  // allocate aligned memory
    current_real_vec = (float *)(malloc(block_size * sizeof(float)));
    current_img_vec = (float *)(malloc(block_size * sizeof(float)));
    block_size = 128;

    for (int i = 0; i < height * width; i++) {
        data[i] = limit;
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
    for (int block_line = 0; block_line < height / block_size; block_line++) {
        for (int block_column = 0; block_column < width / block_size; block_column++) {
            for (int i = 0; i < block_size; i++) {
                float initial_imag = y_start + (block_size * block_line + i) * dy;
                gt_4 = true;
                for (int iter = 0; gt_4 && (iter < limit); iter++) {
					
                    #pragma omp simd reduction(|:gt_4) simdlen(64)
					for (int j = 0; j < block_size; j++) {
                        float initial_real =
                            x_start + (block_size * block_column + j) * dx;

                        float current_img =
                            (iter == 0) ? initial_imag : current_img_vec[j];
                        float current_real =
                            (iter == 0) ? initial_real : current_real_vec[j];

                        float i2 = current_img * current_img;
                        float r2 = current_real * current_real;
                        // if greater then 4 and cell is not set
						int index = (block_line * block_size * width + i * width) + (block_column * block_size + j);
                        bool gt = ((r2 + i2) > 4.0f) &&
                                  (pdata[index] == limit);
                        gt_4 |= gt;

                        // update values from the next iteration with values
                        // form current iteration
                        if (gt) {
                            pdata[index] = iter;
                        } else {
                            current_img_vec[j] =
                                2.0f * current_img * current_real +
                                initial_imag;
                            current_real_vec[j] = r2 - i2 + initial_real;
                        }
                    }
                }
            }
        }
        // this value is used in current iteration and do not change
    }
    return data;
}