#include <iostream>
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include "simulator.hpp"
 

 
int main()
{

    uint32_t num_nodes = 5;
    
    TN_Arch MPS = MPS_Arch(num_nodes);

    Circuit circ = Circuit();
    Simulator sim(&MPS, &circ);

    Gate h(&hadamard_gate, true);
    sim.apply_squbit_gate(0, h);
    Gate t(&t_gate, true);
    sim.apply_squbit_gate(0, t);
    Gate pauli_x(&pauli_x_gate, true);
    sim.apply_squbit_gate(0, pauli_x);
    Gate pauli_y(&pauli_y_gate, true);
    sim.apply_squbit_gate(0, pauli_y);
    Gate pauli_z(&pauli_z_gate, true);
    sim.apply_squbit_gate(0, pauli_z);

    return 0;
}