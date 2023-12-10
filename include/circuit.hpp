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
        bool has_params;
        union Gate_Function {
            xarray<cd> (*gate_builder_no_params)();
            xarray<cd> (*gate_builder_with_params)(std::vector<double>);
        } gate_builder;
        std::vector<double> gate_params;

    public:
        Gate(xarray<cd> (*gate_function)(), const bool &s) {
            single = s;
            has_params = false;
            gate_builder.gate_builder_no_params = gate_function;
        }
        Gate(xarray<cd> (*gate_function)(std::vector<double>), const bool &s, const std::vector<double> &params) {
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

enum Instr_type { GATE = 0, MEASUREMENT = 1, COLLAPSE = 2 };

class Instruction {
    private:
        Instr_type type;
        
    public:
        Instr_type get_type() {
            return type;
        }
};

class Circuit {
    private:
        uint32_t num_qubits;
        std::vector<Instruction> instructions;
    
    public:
        Circuit() {
            
        }

};


#endif