#ifndef QPARSER
#define QPARSER

#include "qasmtools/parser/parser.hpp"
#include "tdd_circuit.hpp"

using namespace qasmtools;

TDD_Circuit parse_circuit(std::string fname) {
    ast::ptr<ast::Program> parsed_circuit = parser::parse_file(fname);
    // initialise state
    uint32_t num_qubits = parsed_circuit->qubits();
    TDD_Circuit circ(num_qubits);
    for (std::list<ast::ptr<ast::Stmt>>::iterator it = parsed_circuit->begin(); it != parsed_circuit->end(); it++) {
        if (dynamic_cast<ast::Gate*>(&(**it)) != nullptr) {
            if (dynamic_cast<ast::DeclaredGate*>(&(**it)) != nullptr) {
                auto gate = dynamic_cast<ast::DeclaredGate*>(&(**it));
                std::string gate_type = gate->name();

                // now compare gate type to check what type of gate it is, and add to circuit accordingly
                // single qubit
                if (gate->num_qargs() == 1) {
                    uint32_t q1 = gate->qarg(0).offset().value();
                    if (gate_type == "x") {
                        circ.x(q1);
                    }
                    else if (gate_type == "y") {
                        circ.y(q1);
                    }
                    else if (gate_type == "z") {
                        circ.z(q1);
                    }
                    else if (gate_type == "h") {
                        circ.h(q1);
                    }
                    else if (gate_type == "s") {
                        circ.s(q1);
                    }
                    else if (gate_type == "t") {
                        circ.t(q1);
                    }
                    else if (gate_type == "sx") {
                        circ.sx(q1);
                    }
                    else if (gate_type == "rz") {
                        double theta = gate->carg(0).constant_eval().value();
                        circ.rz(q1, theta);
                    }
                    else if (gate_type == "tdg") {
                        circ.tdg(q1);
                    }
                    else {
                        std::cout << "Unsupported gate: " << gate_type << std::endl;
                    }
                }
                // otherwise two qubit?
                else if (gate->num_qargs() == 2) {
                    uint32_t q1 = gate->qarg(0).offset().value();
                    uint32_t q2 = gate->qarg(1).offset().value();
                    if (gate_type == "cx") {
                        circ.cx(q1, q2);
                    }
                    else if (gate_type == "cu1") {
                        double lambda = gate->carg(0).constant_eval().value();
                        circ.cu1(q1, q2, lambda);
                    }
                    else {
                        std::cout << "Unsupported gate: " << gate_type << std::endl;
                    }   
                }
            }
        }
    }
    return circ;
}

#endif