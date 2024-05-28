#ifndef QGATES
#define QGATES
#include <complex>
#include <math.h>
#include <vector>

typedef std::complex<double> cd;
using namespace xt;

// post-measurement update matrices (not strictly gates)
xarray<cd> update_to_0(std::vector<double> params) {
    double p0 = params[0];
    return {{cd(pow(p0, -0.5), 0), cd(0,0)}, {cd(0,0), cd(0,0)}};
}
xarray<cd> update_to_1(std::vector<double> params) {
    double p1 = params[0];
    return {{cd(0, 0), cd(0,0)}, {cd(0,0), cd(pow(p1, -0.5),0)}};
}

// Non-parametric single qubit gates

xarray<cd> pauli_x_gate() {
    return {{cd(0,0),cd(1,0)}, {cd(1,0), cd(0,0)}};
}

xarray<cd> pauli_y_gate() {
    return {{cd(0,0), cd(0, -1)}, {cd(0, 1), cd(0,0)}};
}

xarray<cd> pauli_z_gate() {
    return {{cd(1,0), cd(0, 0)}, {cd(0, 0), cd(-1,0)}};
}

// square root of x gate
xarray<cd> sx_gate() {
    double h = 0.5;
    return {{cd(h,h),cd(h,-h)}, {cd(h,-h), cd(h,h)}};
}

xarray<cd> hadamard_gate() {
    double r2 = pow(2, -0.5);
    return {{cd(r2, 0), cd(r2, 0)}, {cd(r2, 0), cd(-r2, 0)}};
}

xarray<cd> phase_gate() {
    return {{cd(1,0), cd(0,0)}, {cd(0,0), cd(0,1)}};
}

xarray<cd> t_gate() {
    double r2 = pow(2, -0.5);
    return {{cd(1,0), cd(0,0)}, {cd(0,0), cd(r2,r2)}};
}

xarray<cd> t_dagger_gate() {
    double r2 = pow(2, -0.5);
    return {{cd(1,0), cd(0,0)}, {cd(0,0), cd(r2,-r2)}};
}

// Parametric single qubit gates

xarray<cd> phase_shift_gate(std::vector<double> params) {
    double theta = params[0];
    double r = cos(theta);
    double i = sin(theta);
    return {{cd(1,0), cd(0,0)}, {cd(0,0), cd(r, i)}};
}

xarray<cd> rx_gate(std::vector<double> params) {
    double theta = params[0];
    double c = cos(theta/2.0);
    double s = sin(theta/2.0);
    return {{cd(c,0), cd(0,-s)}, {cd(0, -s), cd(c, 0)}};
}

xarray<cd> ry_gate(std::vector<double> params) {
    double theta = params[0];
    double c = cos(theta/2.0);
    double s = sin(theta/2.0);
    return {{cd(c,0), cd(-s, 0)}, {cd(s, 0), cd(c, 0)}};
}

xarray<cd> rz_gate(std::vector<double> params) {
    double theta = params[0];
    double c = cos(theta/2.0);
    double s = sin(theta/2.0);
    return {{cd(c, -s), cd(0,0)}, {cd(0,0), cd(c, s)}};
}

xarray<cd> u1_gate(std::vector<double> params) {
    double lambda = params[0];
    xarray<cd> u1 = zeros<cd>({2,2});
    u1(0,0) = cd(1,0);
    u1(1,1) = cd(cos(lambda),sin(lambda));
    return u1;
}

// simulates OpenQASM U gate - arbitrary single-qubit rotation

xarray<cd> u_gate(std::vector<double> params) {
    double theta = params[0];
    double phi = params[1];
    double lambda = params[2];
    double c = cos(theta/2.0);
    double s = sin(theta/2.0);
    double cp = cos(phi);
    double sp = sin(phi);
    double cl = cos(lambda);
    double sl = sin(lambda);
    double clp = cos(lambda + phi);
    double slp = sin(lambda + phi);
    return {{cd(c, 0), cd(-s * cl, -s * sl)}, {cd(s * cp, s * sp), cd(c * clp, c * slp)}};
}

// Non-parametric two qubit gates

xarray<cd> controlled_not_gate() {
    xarray<cd> controlled_not = zeros<cd>({4,4});
    controlled_not(0,0) = cd(1,0);
    controlled_not(1,1) = cd(1,0);
    controlled_not(2,3) = cd(1,0);
    controlled_not(3,2) = cd(1,0);
    return controlled_not;
}

xarray<cd> controlled_z_gate() {
    xarray<cd> controlled_z = zeros<cd>({4,4});
    controlled_z(0,0) = cd(1,0);
    controlled_z(1,1) = cd(1,0);
    controlled_z(2,2) = cd(1,0);
    controlled_z(3,3) = cd(-1,0);
    return controlled_z;
}

xarray<cd> swap_gate() {
    xarray<cd> swap = zeros<cd>({4,4});
    swap(0,0) =  cd(1,0);
    swap(1,2) =  cd(1,0);
    swap(2,1) =  cd(1,0);
    swap(3,3) =  cd(1,0);
    return swap;
}

// Parametric two qubit gates

// this is equivalent to a controlled phase gate
xarray<cd> controlled_u1_gate(std::vector<double> params) {
    double lambda = params[0];
    xarray<cd> controlled_u1 = zeros<cd>({4,4});
    controlled_u1(0,0) = cd(1,0);
    controlled_u1(1,1) = cd(1,0);
    controlled_u1(2,2) = cd(1,0);
    controlled_u1(3,3) = cd(cos(lambda),sin(lambda));
    return controlled_u1;
}


#endif