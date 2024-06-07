#ifndef UTILS
#define UTILS

typedef std::complex<double> cd;
inline bool is_approx_equal(cd x, cd y, double epsilon = 1e-16) {
    if (std::norm(x - cd(0,0)) < epsilon) {
        return std::norm(y) < epsilon;
    }
    else if (std::norm(y - cd(0,0)) < epsilon) {
        return std::norm(x) < epsilon;
    }
    return (std::abs(std::real(x) - std::real(y)) <= epsilon) && (std::abs(std::imag(x) - std::imag(y)) <= epsilon);
}

double convert_bits_to_kb(uint64_t bits) {
    double bytes = bits / 8.0;
    double kb = bytes / 1024.0;
    return kb;
}

#endif