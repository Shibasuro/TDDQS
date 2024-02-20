#ifndef UTILS
#define UTILS

typedef std::complex<double> cd;
inline bool is_approx_equal(cd x, cd y) {
    const double epsilon = 0.0001;
    return std::abs(x - y) <= epsilon * std::abs(x);
}

#endif