/**
 * @file LineMandelCalculator.h
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization
 * over lines
 * @date DATE
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator {
   public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

   protected:
    void resetInitial(int line_number);

   private:
    int *data;
    int *initial_real_vec __attribute__((aligned(32)));
    int *initial_imag_vec __attribute__((aligned(32)));
    float *current_real_vec __attribute__((aligned(32)));
    float *current_img_vec __attribute__((aligned(32)));
};