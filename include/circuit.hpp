#ifndef QCIRCUIT
#define QCIRCUIT
#include "gates.hpp"
#include <vector>
#include "xtensor/xarray.hpp"

using namespace xt;

class Gate {
    private:
        // supports single and two-qubit gates
        bool single;
        xarray<cd> (*gate_builder)();

    public:
        Gate(xarray<cd> (*gate_function)(), bool s) {
            single = s;
        }
        xarray<cd> get_gate() {
            return gate_builder();
        }

}

enum Instr_type { gate = 0, measurement = 1, collapse = 2 }

class Instruction {
    private:
        Instr_type type;

    public:
}

class Circuit {
    private:
        uint32_t num_qubits;
        vector<Instruction> instructions;
    
    public:

}


#endif