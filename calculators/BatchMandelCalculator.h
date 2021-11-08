/**
 * @file BatchMandelCalculator.h
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();
    void mandel(int iter, int i, float initial_imag, int blockP, int* pdata);

private:
    size_t block_size;
    int *data;
    float *current_real_vec __attribute__((aligned(64)));
    float *current_img_vec __attribute__((aligned(64)));
};

#endif