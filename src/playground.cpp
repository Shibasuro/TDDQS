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
#include "tdd_circuit.hpp"
#include "simulator.hpp"
#include "parser.hpp"

#include <chrono>

TDD_Map cache_map;

void time_circuit(TDD_Circuit &circuit) {
    auto t1 = std::chrono::high_resolution_clock::now();
    circuit.simulate();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "Time taken: " << ms_double.count() << "ms" << std::endl;
}

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
    xarray<cd> gate = zeros<cd>({4,4});
    gate(0,0) = cd(1,0);
    gate(2,1) = cd(1,0);
    gate(1,2) = cd(1,0);
    gate(3,1) = cd(1,0);
    gate(0,3) = cd(1,0);
    gate(2,2) = cd(1,0);
    gate(3,3) = cd(1,0);
    gate(3,0) = cd(1,0);
    gate.reshape({2,2,4});
    xarray<cd> g2 = ones<cd>({4,4});
    g2.reshape({2,2,4});
    xarray<cd> g3 = controlled_z_gate();
    g3.reshape({2,2,4});
    g3 *= cd(-2,0);

    xarray<cd> t_sum = gate + g2 + g3;

    TDD tdd1 = convert_tensor_to_TDD(gate);
    TDD tdd2 = convert_tensor_to_TDD(g2);
    TDD tdd3 = convert_tensor_to_TDD(g3);
    std::cout << t_sum << std::endl;
    std::vector<TDD> tdds;
    tdds.push_back(tdd1);
    tdds.push_back(tdd2);
    tdds.push_back(tdd3);
    TDD tdd = add_tdds(tdds);

    std::vector<size_t> shape = tdd.get_shape();
    for (uint32_t i = 0; i < shape.size(); i++) {
        std::cout << "dimension for index " << i << " is " << shape[i] << std::endl;
    }

    bool valid = true;

    for (uint32_t i = 0; i < gate.shape()[0]; i++) {
        for (uint32_t j = 0; j < gate.shape()[1]; j++) {
            for (uint32_t k = 0; k < gate.shape()[2]; k++) {
                cd value = tdd.get_value({i,j,k});
                std::cout << value << std::endl;
                if (value != t_sum(i,j,k)) {
                    valid = false;
                }
            }
        }
    }
    std::cout << "Validity: " << valid << std::endl;

    std::cout << "nodes: " << cache_map.num_unique_nodes() << std::endl;
    std::cout << "edges: " << cache_map.num_unique_edges() << std::endl;
}

void tdd_contract_test() {
    xarray<cd> gate = zeros<cd>({4,4});
    gate(0,0) = cd(1,0);
    gate(2,1) = cd(1,0);
    gate(1,2) = cd(1,0);
    gate(3,1) = cd(1,0);
    gate(0,3) = cd(1,0);
    gate(2,2) = cd(1,0);
    gate(3,3) = cd(1,0);
    gate(3,0) = cd(1,0);
    gate.reshape({2,2,4});
    xarray<cd> g2 = ones<cd>({4,4});
    g2.reshape({2,2,4});

    xarray<cd> contracted = linalg::tensordot(gate, g2, {1}, {1});

    // reorder so that it is same ordering as TDD implementation
    contracted = swapaxes(contracted, 1, 2);

    std::cout << contracted << std::endl;

    TDD tdd_expected = convert_tensor_to_TDD(contracted);

    std::cout << "expected nodes: " << cache_map.num_unique_nodes() << std::endl;
    std::cout << "expected edges: " << cache_map.num_unique_edges() << std::endl;

    svector<size_t> shape = contracted.shape();
    for (uint32_t i = 0; i < shape.size(); i++) {
        std::cout << "dimension for index " << i << " is " << shape[i] << std::endl;
    }


    TDD tdd1 = convert_tensor_to_TDD(gate);
    TDD tdd2 = convert_tensor_to_TDD(g2);

    std::cout << "contracting... " << std::endl;

    TDD contracted_tdd = contract_tdds(tdd1, tdd2, {1}, {1});

    std::vector<size_t> t_shape = contracted_tdd.get_shape();
    
    for (uint32_t i = 0; i < t_shape.size(); i++) {
        std::cout << "dimension for index " << i << " is " << t_shape[i] << std::endl;
    }

    bool valid = true;
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {
                for (size_t l = 0; l < shape[3]; l++) {
                    cd value = contracted_tdd.get_value({i,j,k,l});
                    // std::cout << value << " vs expected: " << contracted(i,j,k,l) << std::endl;
                    if (value != contracted(i,j,k,l)) {
                        valid = false;
                    }
                }
            }
        }
    }
    std::cout << "Validity: " << valid << std::endl;

    std::cout << "nodes: " << cache_map.num_unique_nodes() << std::endl;
    std::cout << "edges: " << cache_map.num_unique_edges() << std::endl;
}

void tdd_circuit_test() {
    uint32_t num_qubits = 3;
    TDD_Circuit circ(num_qubits, "000");
    // something seems to be wrong with cnot, always targets adjacent one?
    circ.add_instruction(Instruction(Instr_type::GATE, Gate(&hadamard_gate, true), 0));
    circ.add_instruction(Instruction(Instr_type::GATE, Gate(&controlled_not_gate, false), 0, 2));
    circ.simulate();
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                cd value = circ.get_amplitude({i,j,k});
                std::cout << "Amplitude of " << i << j << k << " = " << value << std::endl;
            }
        }
    }
}

void toffoli_test() {
    uint32_t num_qubits = 3;
    TDD_Circuit circ(num_qubits, "101");
    circ.toffoli(0,1,2);
    time_circuit(circ);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                cd value = circ.get_amplitude({i,j,k});
                std::cout << "Amplitude of " << i << j << k << " = " << value << std::endl;
            }
        }
    }
}

#define PI 3.14159265

double acot(double x) {
    return atan2(1, x);
}

double cheb(double i, double x) {
    return cosh(i * acosh(x));
}

double gamma(double l, double delta) {
    return 1/cheb(1/l, 1/delta);
}

void fixed_point_grovers_test() {
    // small fixed point grovers test on two qubits effectively
    TDD_Circuit circ(3, "000");
    for (uint32_t i = 0; i < 2; i++) {
        circ.h(i);
    }
    std::string marked_state = "01";

    uint32_t steps = 8;
    double delta = 0.5;
    for (uint32_t i = 0; i < steps; i++) {
        double l = 2 * steps + 1;
        double alpha = 2 * acot((tan(2*PI*(i+1)/l)) * sqrt(1 - pow(gamma(l, delta), 2)));
        double beta = -2 * acot((tan(2*PI*(steps-i)/l)) * sqrt(1 - pow(gamma(l, delta), 2)));
        // repeat U and W steps times

        // carry out U
        for (uint32_t j = 0; j < marked_state.size(); j++) {
            if (marked_state[j] == '0') {
                circ.x(j);
            }
        }
        circ.toffoli(0, 1, 2);
        circ.rz(2, beta);
        circ.toffoli(0, 1, 2);
        for (uint32_t j = 0; j < marked_state.size(); j++) {
            if (marked_state[j] == '0') {
                circ.x(j);
            }
        }

        // carry out W
        for (uint32_t j = 0; j < marked_state.size(); j++) {
            circ.h(j);
        }
        circ.x(0);
        circ.rz(1, -alpha/2);
        circ.cx(0, 1);
        circ.cx(0, 2);
        circ.rz(1, -alpha/2);
        circ.rz(2, -alpha/2);
        circ.cx(0, 1);
        circ.cx(0, 2);
        circ.x(0);
        circ.rz(1, alpha);
        for (uint32_t j = 0; j < marked_state.size(); j++) {
            circ.h(j);
        }
    }
    //circ.simulate();
    time_circuit(circ);

    cd p_sum = 0;

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                cd value = circ.get_amplitude({i,j,k});
                std::cout << "Probability of " << i << j << k << " = " << value * std::conj(value) << std::endl;
                p_sum += value * std::conj(value);
            }
        }
    }
    std::cout << "Total probability: " << p_sum << std::endl;
}

void parsing_test() {
    MPS_Circuit circ = parse_circuit("/home/shibasuro/tn_project/TNQS/src/qasm_bench/qft22.qasm");
    time_circuit(circ);
    xarray<size_t> indices = zeros<size_t>({circ.get_num_qubits()});
    indices(0) = 1;
    cd amp = circ.get_amplitude(indices);
    double prob = std::real(amp * std::conj(amp));
    std::cout << "all 0 prob: " << prob << std::endl;
    double total_prob = circ.get_qubit_probability(7, 0);
    std::cout << "probability of qubit 8 being 0: " << total_prob << std::endl;
    // std::cout << "nodes: " << cache_map.num_unique_nodes() << std::endl;
    // std::cout << "edges: " << cache_map.num_unique_edges() << std::endl;
    // std::cout << "peak nodes: " << cache_map.peak_nodes() << std::endl;
    // std::cout << "peak edges: " << cache_map.peak_edges() << std::endl;
}

void tdd_conversion_test() {
    xarray<cd> gate = zeros<cd>({4,4});
    for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            gate(i,j) = cd(rand() / (rand() + 0.5),rand() / (rand() + 0.5));
        }
    }
    std::cout << gate << std::endl;
    TDD gate_TDD = convert_tensor_to_TDD(gate);
    xarray<cd> gate2 = convert_TDD_to_tensor(gate_TDD);
    if (gate == gate2) {
        std::cout << "success" << std::endl;
    }
    else {
        std::cout << "failure" << std::endl;
        std::cout << gate2 << std::endl;
    }
}

void swap_axes_test() {
    xarray<cd> matrix = zeros<cd>({1,2,3});
    matrix(0,0,0) = 2;
    matrix(0,0,1) = 0;
    matrix(0,0,2) = 3;
    matrix(0,1,0) = 4;
    matrix(0,1,1) = 0;
    matrix(0,1,2) = 1;
    std::cout << matrix << std::endl;
    TDD tdd = convert_tensor_to_TDD(matrix);
    TDD swapped = swap_adjacent_axes(tdd, 1, 2);
    xarray<cd> swap_tensor = convert_TDD_to_tensor(swapped);
    std::cout << swap_tensor << std::endl;
}

// to record memory usage, can run with valgrind --tool=massif./build/apps/program
// and then print out with ms_print <massif_file>
// Is this a good way to record memory usage? increases time complexity
// Might be better to track with node counts
// Want to record peak memory usage

// /usr/bin/time -v ./build/apps/program gives peak memory usage in KB?
// can also add -v

int main()
{
    // tn_test();
    // tdd_playground();
    // tdd_contract_test();
    // tdd_circuit_test();
    // toffoli_test();
    // fixed_point_grovers_test();
    // parsing_test();
    // tdd_conversion_test();
    swap_axes_test();


    return 0;
}

// benchmarking
// can easily time execution for this and use python time (for timing the python based simulators
// such as qiskit mps, original TDD, QuEST )
// QASM SUPPORTED ON DD, TDD, Qiskit
// NOT SUPPORTED ON QuEST
// DD supports qasm simulation 
// (and can be done with either computing unitary of circuit or computing final statevector)
// Is it worth writing a parser for QuEST to run qasm?
// Alternatively is it preferable to just use Qiskit Statevector Sim?

// Can verify correctness by comparing some sampled amplitudes with qiskit?

// TODO is it possible to reshape a TDD


// TODO TDD's may be useful for other uses of tensors in general (especially with support
// for contraction, reshape, swap axes, addition and general dimensions)