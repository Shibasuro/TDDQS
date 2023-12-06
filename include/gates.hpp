#ifndef QGATES
#define QGATES
#include <Eigen/Dense>
#include <complex>
#include <math.h>

typedef std::complex<double> cd;
using Eigen::Matrix2cd;
using Eigen::Vector2cd;
using Eigen::Matrix4cd;
using Eigen::Vector4cd;

// Non-parametric single qubit gates

Matrix2cd pauli_x_gate() {
    Matrix2cd pauli_y; 
    pauli_y << cd(0,0), cd(1, 0), cd(1, 0), cd(0,0);
    return pauli_y;
}

Matrix2cd pauli_y_gate() {
    Matrix2cd pauli_y; 
    pauli_y << cd(0,0), cd(0, -1), cd(0, -1), cd(0,0);
    return pauli_y;
}

Matrix2cd pauli_z_gate() {
    Matrix2cd pauli_z; 
    pauli_z << cd(1,0), cd(0, 0), cd(0, 0), cd(-1,0);
    return pauli_z;
}

Matrix2cd hadamard_gate() {
    Matrix2cd hadamard;
    double r2 = pow(2, 0.5);
    hadamard << cd(r2, 0), cd(r2, 0), cd(r2, 0), cd(-r2, 0);
    return hadamard;
}

Matrix2cd phase_gate() {
    Matrix2cd phase;
    phase << cd(1,0), cd(0,0), cd(0,0), cd(0,1);
    return phase;
}

Matrix2cd t_gate() {
    Matrix2cd t;
    double r22 = pow(2, 0.5) / 2.0;
    t << cd(1,0), cd(0,0), cd(0,0), cd(r22,r22);
    return t;
}

// Parametric single qubit gates

Matrix2cd phase_shift_gate(double theta) {
    Matrix2cd phase_shift;
    double r = cos(theta);
    double i = sin(theta);
    phase_shift << cd(1,0), cd(0,0), cd(0,0), cd(r, i);
    return phase_shift;
}

Matrix2cd rx_gate(double theta) {
    Matrix2cd rx;
    double c = cos(theta/2);
    double s = sin(theta/2);
    rx << cd(c,0), cd(0,-s), cd(0, -s), (c, 0);
    return rx;
}

Matrix2cd ry_gate(double theta) {
    Matrix2cd ry;
    double c = cos(theta/2);
    double s = sin(theta/2);
    ry << cd(c,0), cd(-s, 0), cd(s, 0), (c, 0);
    return ry;
}

Matrix2cd rz_gate(double theta) {
    Matrix2cd rz;
    double c = cos(theta/2);
    double s = sin(theta/2);
    rz << cd(c, -s), cd(0,0), cd(0,0), cd(c, s);
    return rz;
}

// simulates OpenQASM U gate - arbitrary single-qubit rotation
Matrix2cd u_gate(double theta, double phi, double lambda) {
    Matric2cd u;
    double c = cos(theta/2);
    double s = sin(theta/2);
    double cp = cos(phi);
    double sp = sin(phi);
    double cl = cos(lambda);
    double sl = sin(lambda);
    double clp = cos(lambda + phi);
    double slp = sin(lambda + phi);
    u << cd(c, 0), cd(-s * cl, -s * -sl), cd(s * cp, s * -sp), cd(c * clp, c * slp);
    return u;
}


// Non-parametric two qubit gates

// Parametric two qubit gates

#endif