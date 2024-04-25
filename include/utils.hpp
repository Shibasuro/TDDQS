#ifndef UTILS
#define UTILS

typedef std::complex<double> cd;
inline bool is_approx_equal(cd x, cd y, double epsilon = 0.000001) {
    if (std::abs(x - cd(0,0)) < epsilon) {
        return std::abs(y) < epsilon;
    }
    else if (std::abs(y - cd(0,0)) < epsilon) {
        return std::abs(x) < epsilon;
    }
    return std::abs(x - y) <= epsilon * std::max(std::abs(x), std::abs(y));
}

#endif