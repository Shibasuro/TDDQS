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
            xarray<cd> (*gate_builder_with_params)(std::vector<double> *);
        } gate_builder;
        std::vector<double> *gate_params;

    public:
        Gate() {

        }
        Gate(xarray<cd> (*gate_function)(), const bool &s) {
            single = s;
            has_params = false;
            gate_builder.gate_builder_no_params = gate_function;
        }
        Gate(xarray<cd> (*gate_function)(std::vector<double> *), const bool &s, std::vector<double> *params) {
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
        bool is_single_qubit_gate() {
            return single;
        }

};

enum Instr_type { GATE = 0, MEASUREMENT = 1, COLLAPSE = 2 };

class Instruction {
    private:
        Instr_type type;
        Gate gate;
        uint32_t q1;
        uint32_t q2;
        
    public:
        Instruction(Instr_type t, uint32_t qubit1) {
            type = t;
            q1 = qubit1;
        }
        Instruction(Instr_type t, Gate g, uint32_t qubit1) {
            type = t;
            gate = g;
            q1 = qubit1;
        }
        Instruction(Instr_type t, Gate g, uint32_t qubit1, uint32_t qubit2) {
            type = t;
            gate = g;
            q1 = qubit1;
            q2 = qubit2;
        }
        void apply() {

        }
        Instr_type get_type() {
            return type;
        }
        uint32_t get_q1() {
            return q1;
        }
        uint32_t get_q2() {
            return q2;
        }
        Gate get_gate() {
            return gate;
        }
        bool is_single_qubit_gate() {
            return gate.is_single_qubit_gate();
        }
};

class Circuit {
    private:
        uint32_t num_qubits;
        std::vector<Instruction> instructions;
    
    public:
        std::vector<Instruction> get_instructions() {
            return instructions;
        }
        void add_instruction(Instruction instr) {
            instructions.push_back(instr);
        }
        Circuit() {
            num_qubits = 0;
        }
        Circuit(std::vector<Instruction> instrs) {
            instructions = instrs;
        }

};


#endif