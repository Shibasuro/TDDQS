#ifndef QSIM
#define QSIM
#include "tn_arch.hpp"
#include "circuit.hpp"
#include <xtensor-blas/xlinalg.hpp>

//temporary for output
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <iostream>

using namespace xt;

class Simulator {
    private:
        TN_Arch *architecture;
        Circuit *circuit;
        vector<xarray<cd>> gamma;

        void setup_tensors() {
            for (TN_Node node : architecture->get_nodes()) {
                xarray<cd> arr{cd(1,0), cd(0,0)};
                std::vector<size_t> shape = {2};
                for (uint32_t i = 0; i < node.degree(); i++) {
                    shape.push_back(1);
                }
                arr.resize(shape);
                gamma.push_back(arr);
            }
        }

        void apply_single_qubit_gate(const uint32_t &qubit, Gate &gate) {
            gamma[qubit] = linalg::tensordot(gate.get_gate(), gamma[qubit], 1);
        }

        void apply_instruction() {

        }

        void collapse() {

        }

    public:
        Simulator(TN_Arch *arch, Circuit *circ) {
            architecture = arch;
            circuit = circ;
            setup_tensors();
        }

        void simulate() {
            
        }
        
        void apply_squbit_gate(const uint32_t &qubit, Gate &gate) {
            apply_single_qubit_gate(qubit, gate);
            std::cout << gamma[qubit] << std::endl;
        }
};

#endif