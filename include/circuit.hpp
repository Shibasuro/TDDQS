#ifndef QCIRCUIT
#define QCIRCUIT
#include "gates.hpp"
#include <vector>
#include "xtensor/xarray.hpp"

using namespace xt;
using namespace std;

class Gate {
    private:
        // supports single and two-qubit gates
        bool single;
        bool has_params;
        union Gate_Function {
            xarray<cd> (*gate_builder_no_params)();
            xarray<cd> (*gate_builder_with_params)(vector<double>);
        } gate_builder;
        vector<double> gate_params;

    public:
        Gate(xarray<cd> (*gate_function)(), const bool &s) {
            single = s;
            has_params = false;
            gate_builder.gate_builder_no_params = gate_function;
        }
        Gate(xarray<cd> (*gate_function)(vector<double>), const bool &s, const vector<double> &params) {
            single = s;
            has_params = true;
            gate_params = params;
            gate_builder.gate_builder_with_params = gate_function;
        }
        xarray<cd> get_gate() {
            if (has_params) {
                return gate_builder.gate_builder_with_params(gate_params);
            }
            return gate_builder.gate_builder_no_params();
        }

};

enum Instr_type { gate = 0, measurement = 1, collapse = 2 };

class Instruction {
    private:
        Instr_type type;

    public:
};

class Circuit {
    private:
        uint32_t num_qubits;
        vector<Instruction> instructions;
    
    public:
        Circuit() {
            
        }

};


#endif