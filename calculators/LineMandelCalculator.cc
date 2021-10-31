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
    data = (int*)(_mm_malloc(height * width * sizeof(int), 32)); // allocate aligned memory
    current_line = (float*)(_mm_malloc(width * sizeof(float), 32));
}

LineMandelCalculator::~LineMandelCalculator() {
    _mm_free(data);
    _mm_free(current_line);
    data = NULL;
    current_line = NULL;
}

int* LineMandelCalculator::calculateMandelbrot() {
    __m256 x_start_vec = _mm256_set1_ps(x_start);  // vector of x_start values
    __m256 y_start_vec = _mm256_set1_ps(y_start);  // vector of y_start values
    __m256 dx_vec = _mm256_set1_ps(dx);  // vector of dx values
    __m256 dy_vec = _mm256_set1_ps(dy); // vector of dy values
    __m256 cmp_vec = _mm256_set1_ps(4); // for comapring
    __m256 one_vec = _mm256_set1_ps(1);
    int *pdata = data;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < height; i++)
	{   
        __m256 i_vec = _mm256_set1_ps(i);
        __m256 i_dy_vec = _mm256_mul_ps(i_vec, dy_vec);
        __m256 y_img_vec = _mm256_add_ps(i_dy_vec, y_start_vec);
        
        // line = data + i * width;
        for (int j = 0; j < width; j++){
            current_line[j] = j;
        }
        __m256 j_vec = _mm256_load_ps(current_line);
        __m256 j_dx_vec = _mm256_mul_ps(j_vec, dx_vec);
        __m256 x_real_vec = _mm256_add_ps(i_dy_vec, y_start_vec);

        int iter = 1;
        __m256 iter_vec = _mm256_set1_ps(iter);

        // count iterations for cells in vectors
        __m256 r2_vec =  _mm256_mul_ps(x_real_vec, x_real_vec);
        __m256 i2_vec =  _mm256_mul_ps(y_img_vec, y_img_vec);

        __m256 r2_i2_vec = _mm256_add_ps(r2_vec, i2_vec);
        // create vector with comparising values from vector of r^2 + i^2 compared with vector of 4
        __m256 mask = _mm256_cmp_ps(r2_i2_vec, cmp_vec, _CMP_LT_OS);

        #pragma omp simd
        for (; !_mm256_testz_ps(mask, _mm256_set1_ps(-1)) && iter < limit; iter++) {
            r2_vec =  _mm256_mul_ps(x_real_vec, x_real_vec);
            i2_vec =  _mm256_mul_ps(y_img_vec, y_img_vec);

            r2_i2_vec = _mm256_add_ps(r2_vec, i2_vec);
            
            // create vector with comparising values from vector of r^2 + i^2 compared with vector of 4
            mask = _mm256_cmp_ps(r2_i2_vec, cmp_vec, _CMP_LT_OS);
            
            // find in which cell count of iterations should be updated
            __m256 mask_and_one_vec= _mm256_and_ps(mask, one_vec);
            
            // increment values in vector of iterations
            iter_vec = _mm256_add_ps(mask_and_one_vec, iter_vec);
        }
        __m256i iters_int_vec = _mm256_cvtps_epi32(iter_vec);
        _mm256_store_si256((__m256i *)pdata, iters_int_vec);
        pdata += width; // move pointer to fill next line
	}
    return data;
}
