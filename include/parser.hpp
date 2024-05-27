#ifndef QPARSER
#define QPARSER

#include "qasmtools/parser/parser.hpp"
#include "tdd_circuit.hpp"

using namespace qasmtools;

MPS_Circuit parse_circuit(std::string fname) {
    ast::ptr<ast::Program> parsed_circuit = parser::parse_file(fname);
    // initialise state
    uint32_t num_qubits = parsed_circuit->qubits();
    // MPS_Circuit circ(num_qubits, "101010101");
    MPS_Circuit circ(num_qubits);
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
                    // if q1, q2 are not adjacent (i.e. abs(q1-q2) != 1)
                    // then need to add series of swap gates as well
                    uint32_t difference;
                    uint32_t q1_shifted = q1;
                    uint32_t q2_shifted = q2;
                    if (q1 < q2) {
                        difference = q2 - q1;
                    }
                    else {
                        difference = q1 - q2;
                    }
                    if (difference != 1) {
                        // apply swap gates until they are adjacent
                        // make shift direction consistent (always shift left)
                        for (uint32_t i = 1; i < difference; i++) {
                            if (q1 < q2) {
                                circ.swap(q2 - i + 1, q2 - i);
                                q2_shifted--;
                            }
                            else {
                                circ.swap(q1 - i + 1, q1 - i);
                                q1_shifted--;
                            }
                        }
                    }
                    if (gate_type == "cx") {
                        circ.cx(q1_shifted, q2_shifted);
                    }
                    // cp and cu1 are the same gate
                    else if (gate_type == "cu1" || gate_type == "cp") {
                        double lambda = gate->carg(0).constant_eval().value();
                        circ.cu1(q1_shifted, q2_shifted, lambda);
                    }
                    else if (gate_type == "swap") {
                        circ.swap(q1_shifted, q2_shifted);
                    }
                    else {
                        std::cout << "Unsupported gate: " << gate_type << std::endl;
                    }  
                    if (difference != 1) {
                        // apply swap gates until they are back in original positions
                        for (uint32_t i = 1; i < difference; i++) {
                            if (q1 < q2) {
                                circ.swap(q1 + i, q1 + i + 1);
                            }
                            else {
                                circ.swap(q2 + i, q2 + i + 1);
                            }
                        }
                    } 
                }
            }
        }
        else if (dynamic_cast<ast::MeasureStmt*>(&(**it)) != nullptr) {
            auto measure_stmt = dynamic_cast<ast::MeasureStmt*>(&(**it));
            uint32_t qubit = measure_stmt->q_arg().offset().value();
            circ.add_instruction(Instruction(Instr_type::MEASUREMENT, qubit));
        }
    }
    return circ;
}

#endif