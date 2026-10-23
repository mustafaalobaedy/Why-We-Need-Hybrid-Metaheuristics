// Shared-library wrapper around the official CEC2017 C++ implementation
// (Awad et al., 2016; fast-pow variant by V. Stanovov), unchanged.
#include "cec17_test_fast_pow.cpp"

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;

extern "C" void cec17_eval(double *x, double *f, int nx, int mx, int func_num)
{
    cec17_test_func(x, f, nx, mx, func_num);
}
