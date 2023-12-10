#include <iostream>
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp>
#include "simulator.hpp"


int main()
{

    // squbit gate + amplitude test
    std::cout << "single qubit gate superposition test" << std::endl;
    uint32_t num_nodes = 10;
    
    TN_Arch MPS = MPS_Arch(num_nodes);

    Circuit circ = Circuit();
    Simulator sim(&MPS, &circ);

    Gate h(&hadamard_gate, true);
    std::vector<uint32_t> bitstring;
    for (uint32_t i = 0; i < num_nodes; i++) {
        sim.apply_squbit_gate(i, h);
        bitstring.push_back(i % 2);
    }
    long double p = sim.get_probability(bitstring);
    std::cout << p << std::endl;
    std::cout << pow(2, num_nodes) * p << std::endl;
    // Gate t(&t_gate, true);
    // sim.apply_squbit_gate(0, t);
    // Gate pauli_x(&pauli_x_gate, true);
    // sim.apply_squbit_gate(0, pauli_x);
    // Gate pauli_y(&pauli_y_gate, true);
    // sim.apply_squbit_gate(0, pauli_y);
    // Gate pauli_z(&pauli_z_gate, true);
    // sim.apply_squbit_gate(0, pauli_z);

    // rq decomposition test
    // xarray<cd> rq_test = random::randn<double>({5,5});
    // std::cout << rq_test << std::endl;
    // std::tuple<xarray<cd>, xarray<cd>> second_decomp = rq(rq_test);
    // xarray<cd> q2 = get<0>(second_decomp);
    // xarray<cd> r2 = get<1>(second_decomp);
    // xarray<cd> out = linalg::tensordot(r2, q2, 1);
    // std::cout << out << std::endl;


    //tqubit gate test
    std::cout << "two qubit gate superposition test" << std::endl;
    uint32_t nodes = 2;
    TN_Arch MPS2 = MPS_Arch(nodes);
    Circuit circ2 = Circuit();
    Simulator sim2(&MPS2, &circ2);
    Gate cnot(&controlled_not_gate, false);
    Gate pp(&pauli_x_gate, true);
    // sim2.apply_squbit_gate(1,pp);
    //sim2.apply_squbit_gate(1, h);
    sim2.apply_squbit_gate(0, h);
    sim2.apply_tqubit_gate(0, 1, cnot);

    std::vector<uint32_t> bits;
    bits.push_back(0);
    bits.push_back(0);
    long double p00 = sim2.get_probability(bits);
    std::cout << "p00: " << p00 << std::endl;
    bits.clear();
    bits.push_back(0);
    bits.push_back(1);
    long double p01 = sim2.get_probability(bits);
    std::cout << "p01: " << p01 << std::endl;
    bits.clear();
    bits.push_back(1);
    bits.push_back(0);
    long double p10 = sim2.get_probability(bits);
    std::cout << "p10: " << p10 << std::endl;
    bits.clear();
    bits.push_back(1);
    bits.push_back(1);
    long double p11 = sim2.get_probability(bits);
    std::cout << "p11: " << p11 << std::endl;

    return 0;
}