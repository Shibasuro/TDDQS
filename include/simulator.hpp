#ifndef QSIM
#define QSIM
#include "tn_arch.hpp"
#include "circuit.hpp"
#include <xtensor-blas/xlinalg.hpp>
#include <set>

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
        vector<uint32_t> contraction_order;

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

        TN_Node get_node(uint32_t nindex) {
            return architecture->get_node(nindex);
        }

    public:
        Simulator(TN_Arch *arch, Circuit *circ) {
            architecture = arch;
            circuit = circ;
            setup_tensors();
            for (uint32_t i = 0; i < architecture->size(); i++) {
                contraction_order.push_back(i);
            }
        }

        void simulate() {
            
        }
        
        void apply_squbit_gate(const uint32_t &qubit, Gate &gate) {
            apply_single_qubit_gate(qubit, gate);
            std::cout << gamma[qubit] << std::endl;
        }

        cd get_amplitude(vector<uint32_t> &target_bitstring) {
            // given order 0-1-2-3-4, contract from 0 to 1, then contract from 01 to 2, contract from 012 to 3, and so on
            uint32_t first_index = contraction_order[0];
            TN_Node first_node = get_node(first_index);
            // amalgam contains the current state of the contracted part, ready to contract its edges with the next node in the order
            xarray<cd> amalgam = row(gamma[first_index], target_bitstring[first_index]);
            set<uint32_t> amalgamated_nodes;
            amalgamated_nodes.insert(first_index);
            // edges_to_contract contains the edges going into the contracted component that have not yet been contracted
            vector<std::pair<uint32_t, uint32_t>> edges_to_contract;
            for (uint32_t target : first_node.get_neighbours()) {
                edges_to_contract.push_back(std::pair<uint32_t, uint32_t>(first_index, target));
            }
            for (uint32_t i = 1; i < contraction_order.size(); i++) {
                uint32_t current_index = contraction_order[i];
                TN_Node current_node = get_node(current_index);

                vector<uint32_t> new_neighbours = current_node.get_neighbours();

                vector<size_t> amalgam_axes;
                vector<size_t> current_node_axes;
                for (uint32_t j = 0; j < edges_to_contract.size(); j++) {
                    // each edge which targets the current index is an edge we must contract at this step
                    if (edges_to_contract[j].second == current_index) {
                        // need to record indices of edges now that we have found the ones to contract
                        amalgam_axes.push_back(j);
                        // calculate current_node index for this edge
                        current_node_axes.push_back(current_node.get_index(edges_to_contract[j].first));
                        // removes edges that we have processed from edges to contract
                        vector<pair<uint32_t, uint32_t>>::iterator new_end = remove(edges_to_contract.begin(), edges_to_contract.end(), edges_to_contract[j]);
                        edges_to_contract.erase(new_end, edges_to_contract.end());
                    }
                }
                // carry out the contraction for this step between amalgam and the current node, need to compute axes to contract over
                // for amalgam, these axes are the j's above?
                // for current node, these axes are the position of each source node in the vector of neighbours
                xarray<cd> target = row(gamma[current_index], target_bitstring[current_index]);
                amalgam = linalg::tensordot(amalgam, target, amalgam_axes, current_node_axes);

                // Update the state of the set to keep track of new nodes that has been contracted into amalgam
                // new edges only added if they are to a node not yet contracted into the amalgam
                for (uint32_t target : new_neighbours) {
                    if (amalgamated_nodes.find(target) == amalgamated_nodes.end()) {
                        edges_to_contract.push_back(std::pair<uint32_t, uint32_t>(current_index, target));
                    }
                }
                amalgamated_nodes.insert(current_index);

            }
            return amalgam(0);
        }

        long double get_probability(vector<uint32_t> &target_bitstring) {
            cd amplitude = get_amplitude(target_bitstring);
            return std::real(amplitude * std::conj(amplitude));
        }
};

#endif