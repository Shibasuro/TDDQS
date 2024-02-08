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
#include "tdd_arch.hpp"

TDD_Map cache_map;

void tn_test() {
    // squbit gate + amplitude test
    std::cout << "single qubit gate superposition test" << std::endl;
    uint32_t num_nodes = 2;
    
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

    // testing get_qubit_probability/measurement
    for (uint32_t i = 0; i < num_nodes; i++) {
        std::cout << sim.get_qubit_probability(i, 0) << std::endl;
    }

    std::cout << "measuring qubit 0" << std::endl;
    sim.measure(0);
    std::cout << "p0: " << sim.get_qubit_probability(0, 0) << " p1: " << sim.get_qubit_probability(0, 1) << std::endl;


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
}

void tdd_playground() {
    xarray<cd> gate = controlled_z_gate();
    TDD tdd = convert_tensor_to_TDD(gate);
    std::cout << gate << std::endl;

    for (uint32_t i = 0; i < gate.shape()[0]; i++) {
        for (uint32_t j = 0; j < gate.shape()[1]; j++) {
            std::cout << tdd.get_value({i,j}) << std::endl;
        }
    }

    std::cout << "nodes: " << cache_map.num_nodes() << std::endl;
    std::cout << "edges: " << cache_map.num_nodes() << std::endl;
}

int main()
{
    //tn_test();
    tdd_playground();

    return 0;
}