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
    data = (int*)(malloc(height * width * sizeof(int)));
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;
}

int* LineMandelCalculator::calculateMandelbrot() {
    __m256 x_start_vec = _mm256_set1_ps(x_start);  // vector of x_start values
    __m256 y_start_vec = _mm256_set1_ps(y_start);  // vector of y_start values
    __m256 dx_vec = _mm256_set1_ps(dx);  // vector of dx values
    __m256 dy_vec = _mm256_set1_ps(dy); // vector of dy values
    __m256 cmp_vec = _mm256_set1_ps(4); // for comapring
    __m256 one_vec = _mm256_set1_ps(1);
    int *pdata = data;
    for (int i = 0; i < height; i++)
	{   
        __m256 i_vec = _mm256_set1_ps(i);
        __m256 i_dy_vec = _mm256_mul_ps(i_vec, dy_vec);
        __m256 y_img_vec = _mm256_add_ps(i_dy_vec, y_start_vec);

		for (int j = 0; j < width; j += 8) // +8 because function requires 8 x floats vectors
		{
            __m256 j_vec = _mm256_set_ps(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j + 0);
            __m256 j_dx_vec = _mm256_mul_ps(j_vec, dx_vec);
            __m256 x_real_vec = _mm256_add_ps(i_dy_vec, y_start_vec);

            int iter = 1;
            __m256 iter_vec = _mm256_set1_ps(iter);

            // count iterations for cells in vectors
			for (; iter < limit; iter++) {
                __m256 r2_vec =  _mm256_mul_ps(x_real_vec, x_real_vec);
                __m256 i2_vec =  _mm256_mul_ps(y_img_vec, y_img_vec);

                __m256 r2_i2_vec = _mm256_add_ps(r2_vec, i2_vec);
                
                // create vector with comparising values from vector of r^2 + i^2 compared with vector of 4
                __m256 mask = _mm256_cmp_ps(r2_i2_vec, cmp_vec, _CMP_LT_OS);
                
                // find in which cell count of iterations should be updated
                __m256 mask_and_one_vec= _mm256_and_ps(mask, one_vec);
                
                // increment values in vector of iterations
                iter_vec = _mm256_add_ps(mask_and_one_vec, iter_vec);
                
                if (_mm256_testz_ps(mask, _mm256_set1_ps(-1))) break; // stops iteration
                // imag = 2.0f * real * imag + y;
				// real = r2 - i2 + x;
            }
            __m256i iters_int_vec = _mm256_cvtps_epi32(iter_vec);
            int *iters_int_array = (int *)&iters_int_vec;
            for (int i = 0; i < 8; i++) {
                *(pdata++) = iters_int_array[i];
            }
        }
	}
    return data;
}
