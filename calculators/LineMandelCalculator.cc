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
    current_line = NULL;
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

			float x = x_start + j * dx; // current real value
			float real = x;
            int iter;
	
			for (iter = 0; iter < limit; iter++) {
                float r2 = x * x;
				float i2 = y * y;
				// if (r2 + i2 > 4.0f)
				// {
				// 	break;
				// }
                __m256 r2_i2 = _mm256_add_ps(zr2, zi2);
                __m256 mask = _mm256_cmp_ps(r2_i2, cmp_vec, _CMP_LT_OS);
                if (_mm256_testz_ps(mask, _mm256_set1_ps(-1))) break; // stops iteration
                // imag = 2.0f * real * imag + y;
				// real = r2 - i2 + x;
            }

            // *(pdata++) = iter;
        }
	}
    return data;
}
