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
    // initialise vector of x values as it is not change for each iteration 
    initial_real_vec = (int *)(malloc(width * sizeof(int)));
    initial_imag_vec = (int *)(malloc(height * sizeof(int)));
    current_real_vec = (float *)(malloc(width * sizeof(float)));
    current_img_vec = (float *)(malloc(width * sizeof(float)));
    for(int i = 0; i < width; i++){
        initial_real_vec[i] = x_start + i * dx;
        current_real_vec[i] = x_start + i * dx;
    }

    for(int i = 0; i < height; i++){
        initial_imag_vec[i] = y_start + i * dy;
        current_img_vec[i] = y_start + i * dy;
    }
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;

    free(initial_real_vec);
    initial_real_vec = NULL;

    free(initial_imag_vec);
    initial_imag_vec = NULL;

    free(current_real_vec);
    current_real_vec = NULL;
    
    free(current_img_vec);
    current_img_vec = NULL;
}

void LineMandelCalculator::resetInitial(int line_number) {
    for (int i = 0; i < width; i++){
        current_real_vec[i] = initial_real_vec[i];
    }
    current_img_vec[line_number] = y_start + line_number * dy;
}
// pragma omp reduction(operation:value where result will be stored)
int* LineMandelCalculator::calculateMandelbrot()
{
   	int *pdata = data;
    for (int i = 0; i < width * height; i++){
        pdata[i] = -1;
    }

    for (int i = 0; i < height; i++) {
        // this value is used in current iteration and do not change
        float initial_imag = y_start + i * dy;

        float current_img = initial_imag;
        bool less_then_4 = true;
        int iter;
        for (iter = 0; less_then_4 && (iter < limit); iter++) {
            float current_real = -1;
#pragma omp simd reduction(| : gt_4)
            for (int j = 0; j < width; j++) {
                float initial_real = x_start + j * dx;
                float current_real = (current_real == -1) ? initial_real : current_real;

                float i2 = current_img * current_img;
                float r2 = current_real * current_real;
                // if greater then 4 and cell is not set
                bool gt = ((r2 + i2) > 4.0f) &  (pdata[j + i * width] == -1);
                less_then_4 |= !gt;

                // update values from the next iteration with values form
                // current iteration
                if (gt) {
                    pdata[i * width + j] = iter;
                } else {
                    current_img = 2.0f * current_img * current_real + initial_imag;
                    current_real_vec[j] = r2 - i2 + initial_real;
                }
            }
        }
    }
    return data;
}
