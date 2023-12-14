#ifndef QSIM
#define QSIM
#include "tn_arch.hpp"
#include "circuit.hpp"
#include <set>
#include <xtensor-blas/xlinalg.hpp>
// for views 
#include <xtensor/xview.hpp>
#include <xtensor/xstrided_view.hpp>
// for randomisation
#include <random>

//temporary for output
#include <xtensor/xio.hpp>
#include <iostream>

using namespace xt;

// compute rq decomposition using the available routine for qr decomposition
// assumes a is a matrix (i.e. tensor of rank 2);
std::tuple<xarray<cd>, xarray<cd>> rq(xarray<cd> &a) {
    std::tuple<xarray<cd>, xarray<cd>> result;
    xarray<cd> a_bar = flip(a, 0);
    xarray<cd> a_bar_dagger = conj(transpose(a_bar,{1,0}));
    std::tuple<xarray<cd>, xarray<cd>> intermediate = linalg::qr(a_bar_dagger, linalg::qrmode::complete); 
    xarray<cd> q_bar = get<0>(intermediate);
    xarray<cd> r_bar = get<1>(intermediate);
    xarray<cd> q_bar_dagger = conj(transpose(q_bar,{1,0}));
    get<0>(result) = flip(q_bar_dagger, 0);
    xarray<cd> r_bar_dagger = conj(transpose(r_bar,{1,0}));
    get<1>(result) = flip(flip(r_bar_dagger, 0), 1);
    return result;
}

class Simulator {
    private:
        TN_Arch *architecture;
        Circuit *circuit;
        std::vector<xarray<cd>> gamma;
        std::vector<uint32_t> contraction_order;

        // random setup
        std::default_random_engine generator;

        void setup_tensors() {
            for (TN_Node node : architecture->get_nodes()) {
                xarray<cd> arr{cd(1,0), cd(0,0)};
                std::vector<size_t> shape = {2};
                for (uint32_t i = 0; i < node.degree(); i++) {
                    shape.push_back(1);
                }
                arr.reshape(shape);
                gamma.push_back(arr);
            }
        }
        double uniform_rand_in_range(double min, double max) {
            std::uniform_real_distribution<double> distribution(min, max);
            return distribution(generator);
        }

        void apply_single_qubit_gate(const uint32_t &qubit, Gate &gate) {
            gamma[qubit] = linalg::tensordot(gate.get_gate(), gamma[qubit], 1);
        }

        // assumes dim >= 2, 0 < index < dim, dimension should be the degree of the node + 1
        std::vector<size_t> calculate_transpose(const uint32_t &dim, const uint32_t &index) {
            std::vector<size_t> t;
            for (uint32_t i = 1; i < index + 1; i++) {
                t.push_back(i);
            }
            for (uint32_t i = index + 2; i < dim; i++) {
                t.push_back(i);
            }
            t.push_back(0);
            t.push_back(index + 1);
            return t;
        }

        std::vector<size_t> calculate_inverse_transpose(const uint32_t &dim, const uint32_t &index) {
            std::vector<size_t> t;
            for (uint32_t i = 0; i < index + 1; i++) {
                t.push_back(i);
            }
            t.push_back(dim - 1);
            for (uint32_t i = index + 2; i < dim; i++) {
                t.push_back(i - 1);
            }
            return t;
        }

        std::vector<size_t> push_to_front(const uint32_t &dim, const uint32_t &index) {
            std::vector<size_t> t;
            t.push_back(index);
            for (uint32_t i = 1; i <= index; i++) {
                t.push_back(i - 1);
            }
            for (uint32_t i = index + 1; i < dim; i++) {
                t.push_back(i);
            }
            return t;
        }

        void print_shape(xarray<cd> v) {
            svector<size_t> s = v.shape();
            for (uint32_t i = 0; i < s.size(); i++) {
                std::cout << s[i] << std::endl;
            }
        }

        size_t shape_prod(svector<size_t> shape) {
            size_t prod = 1;
            for (uint32_t i = 0; i < shape.size(); i++) {
                prod *= shape[i];
            }
            return prod;
        }

        // following https://arxiv.org/pdf/2002.07730.pdf to some extent
        // QR-SVD algorithm
        // Steps to take
        // 1. Tranpose linked indices to the back
        // 2. QR decompose gamma[qubit1], gamma[qubit2]
        // 3. Apply gate to R1, R2 (i.e. contract this section)
        // 4. Carry out SVD on the contracted result of Step 2
        // 5. Cull the unwanted singular values (depending on whether we want to preserve chi or preserve fidelity)
        // 6. Take R1' = S'V', R2' = D' (where S'V'D' are the SVD post culling);
        // 7. Recompute temporarily gamma[qubit1] = Q1R1', gamma[qubit2] = Q2R2'
        // 8. Rearrange linked indices back into place to set final gammas
        
        void apply_two_qubit_gate(const uint32_t &qubit1, const uint32_t &qubit2, Gate &gate) {
            // if the two qubits are not adjacent, do nothing for now, in future may consider swapping them into position?
            // Alternatively, assume swaps are coded as gates into the circuit when they are required
            // index of q2 in q1's list of neighbouring nodes and vice versa
            uint32_t q1_to_q2_bond_index = architecture->get_node(qubit1).get_index(qubit2);
            uint32_t q2_to_q1_bond_index = architecture->get_node(qubit2).get_index(qubit1);
            if (q1_to_q2_bond_index == architecture->get_node(qubit1).degree()) {
                return;
            }

            // + 1 to include the physical dimension
            uint32_t dim1 = architecture->get_node(qubit1).degree() + 1; 
            uint32_t dim2 = architecture->get_node(qubit2).degree() + 1;
            
            // 1. Transpose to group linked indices at the end (outputs transposed and reshaped matrices gamma1, gamma2)
            std::vector<size_t> transpose1 = calculate_transpose(dim1, q1_to_q2_bond_index);
            std::vector<size_t> transpose2 = calculate_transpose(dim2, q2_to_q1_bond_index);
            uint32_t old_chi = row(gamma[qubit1],q1_to_q2_bond_index + 1).size();

            // uses 2 * old_chi as the two indices to move to the back are i and alpha, of dimension 2 and old_chi respectively
            xarray<cd> gamma1 = transpose(gamma[qubit1], transpose1);
            svector<size_t> gamma1_transposed_shape = gamma1.shape();
            size_t mat_dim1 = shape_prod(gamma1_transposed_shape) / (2 * old_chi);
            gamma1.reshape({mat_dim1, 2 * old_chi});

            xarray<cd> gamma2 = transpose(gamma[qubit2], transpose2);
            svector<size_t> gamma2_transposed_shape = gamma2.shape();
            size_t mat_dim2 = shape_prod(gamma2_transposed_shape) / (2 * old_chi);
            gamma2.reshape({mat_dim2, 2 * old_chi});
            // 2. RQ decomposition for gamma1 and gamma2
            // gamma1 ----- gamma2 -> Q1 ---- R1 ----- R2 ------- Q2
            std::tuple<xarray<cd>, xarray<cd>> first_decomp = rq(gamma1);
            std::tuple<xarray<cd>, xarray<cd>> second_decomp = rq(gamma2);
            xarray<cd> q1 = get<0>(first_decomp);
            xarray<cd> r1 = get<1>(first_decomp);
            xarray<cd> q2 = get<0>(second_decomp);
            xarray<cd> r2 = get<1>(second_decomp);

            // 3. Gate application on Rs (outputs theta, the contraction of r1,r2 and the gate)
            q1.reshape({2 * old_chi, 2, old_chi});
            q2.reshape({2 * old_chi, 2, old_chi});

            // contract the two R gates first, 1 axis to sum over
            xarray<cd> intermediate = linalg::tensordot(q1, q2, {2}, {2}); // bma, cna -> bmcn

            // now apply the two qubit gate, 2 axes to sum over as it is applied to both qubits simultaneously
            // ????? Is it known that this will be the first 2 axes of intermediate though? Or will I need to index it exactly ????
            //intermediate.reshape({4 * old_chi, 4 * old_chi});
            xarray<cd> gate_matrix = gate.get_gate();
            // 2 qubit gate is 4x4 matrix, can be reshaped as 2x2x2x2
            gate_matrix.reshape({2, 2, 2, 2});

            xarray<cd> theta = linalg::tensordot(gate_matrix, intermediate, {2,3}, {1,3}); // ijmn, bmcn -> ijbc
            // reorder from ijbc to ibjc
            theta = swapaxes(theta, 1, 2);
            theta.reshape({4 * old_chi, 4 * old_chi});

            // 4. SVD 
            // SVD returns unitary matrices as U and V
            std::tuple<xarray<cd>, xarray<cd>, xarray<cd>> svd = linalg::svd(theta);
            xarray<cd> u = get<0>(svd);
            xarray<cd> s = get<1>(svd);
            xarray<cd> v = get<2>(svd);

            // 5. SVD Culling and Renormalisation (outputs new u, s, v)
            // remove values that come from floating point errors;
            double double_error = 0.0000000001;
            filtration(u, real(u * conj(u)) < double_error) = 0;
            xarray<cd> temp_s = filter(s, real(s) > double_error);
            filtration(v, real(v * conj(v)) < double_error) = 0;

            uint32_t new_chi = temp_s.size();
            xarray<cd> u_prime = view(u, all(), range(0, new_chi));
            xarray<cd> s_prime = view(s, range(0, new_chi));
            xarray<cd> v_prime = view(v, range(0, new_chi), all());

            // renormalise s_prime
            s_prime *= pow( sum(s * s) / sum(s_prime * s_prime), 0.5);

            // 6. Update Rs (r1_prime = us, r2_prime = v) (outputs r1_prime, r2_prime)
            xarray<cd> q1_prime = linalg::dot(u_prime, diag(s_prime));
            xarray<cd> q2_prime = v_prime;

            q1_prime.reshape({2, 2 * old_chi, new_chi});
            q2_prime.reshape({new_chi, 2, 2 * old_chi});

            // 7. Recomposition (gamma1 gamma2 are still matrices here, so need to reshape correctly)

            gamma1 = linalg::tensordot(r1, q1_prime, {1}, {1}); // kb, ibt -> kit (want ikt)
            gamma2 = linalg::tensordot(r2, q2_prime, {1}, {2}); // ly, tjy -> ltj (want jlt)

            // 8. Reshape, get desired order of axis, then restore original index order, then update gamma values
            gamma1_transposed_shape[gamma1_transposed_shape.size() - 1] = new_chi;
            gamma2_transposed_shape[gamma2_transposed_shape.size() - 1] = new_chi;
            gamma1.reshape(gamma1_transposed_shape);
            gamma2.reshape(gamma2_transposed_shape);

            // get desired axis order (i,k,t), (j,l,t) (i.e. move i,j back to the front) (for gamma1 i is axis dim1 - 2, gamma 2 j is axis dim2 - 1
            gamma1 = transpose(gamma1, push_to_front(dim1, dim1 - 2));
            gamma2 = transpose(gamma2, push_to_front(dim2, dim2 - 1));

            // shift t index back to original index position for that bond
            gamma[qubit1] = transpose(gamma1, calculate_inverse_transpose(dim1, q1_to_q2_bond_index));
            gamma[qubit2] = transpose(gamma2, calculate_inverse_transpose(dim2, q2_to_q1_bond_index));\
        }

        void apply_instruction(Instruction instruction) {
            switch(instruction.get_type()) {
                case Instr_type::GATE:
                    break;
                case Instr_type::MEASUREMENT:
                    break;
                case Instr_type::COLLAPSE:
                    break;
            }
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
            // default contraction order
            for (uint32_t i = 0; i < architecture->size(); i++) {
                contraction_order.push_back(i);
            }
            generator.seed(time(NULL));
        }

        void simulate() {
            
        }
        
        void apply_squbit_gate(const uint32_t &qubit, Gate &gate) {
            apply_single_qubit_gate(qubit, gate);
        }

        void apply_tqubit_gate(const uint32_t &qubit1, const uint32_t &qubit2, Gate &gate) {
            apply_two_qubit_gate(qubit1, qubit2, gate);
        }

        cd get_amplitude(std::vector<uint32_t> &target_bitstring) {
            // given order 0-1-2-3-4, contract from 0 to 1, then contract from 01 to 2, contract from 012 to 3, and so on
            uint32_t first_index = contraction_order[0];
            TN_Node first_node = get_node(first_index);
            // amalgam contains the current state of the contracted part, ready to contract its edges with the next node in the order
            xarray<cd> amalgam = view(gamma[first_index], target_bitstring[first_index], all());
            std::set<uint32_t> amalgamated_nodes;
            amalgamated_nodes.insert(first_index);
            // edges_to_contract contains the edges going into the contracted component that have not yet been contracted
            std::vector<std::pair<uint32_t, uint32_t>> edges_to_contract;
            for (uint32_t target : first_node.get_neighbours()) {
                edges_to_contract.push_back(std::pair<uint32_t, uint32_t>(first_index, target));
            }
            for (uint32_t i = 1; i < contraction_order.size(); i++) {
                uint32_t current_index = contraction_order[i];
                TN_Node current_node = get_node(current_index);

                std::vector<uint32_t> new_neighbours = current_node.get_neighbours();

                std::vector<size_t> amalgam_axes;
                std::vector<size_t> current_node_axes;
                // Graph is simple, so should not have problems with multiple edges between two nodes
                for (uint32_t j = 0; j < edges_to_contract.size(); j++) {
                    // each edge which targets the current index is an edge we must contract at this step
                    if (edges_to_contract[j].second == current_index) {
                        // need to record indices of edges now that we have found the ones to contract
                        amalgam_axes.push_back(j);
                        // calculate current_node index for this edge
                        current_node_axes.push_back(current_node.get_index(edges_to_contract[j].first));
                    }
                }
                // second loop to remove edges which we have marked to contract
                for (size_t j : amalgam_axes) {
                    // removes edges that we have processed from edges to contract
                    std::vector<std::pair<uint32_t, uint32_t>>::iterator new_end = remove(edges_to_contract.begin(), edges_to_contract.end(), edges_to_contract[j]);
                    edges_to_contract.erase(new_end, edges_to_contract.end());
                }

                // carry out the contraction for this step between amalgam and the current node, need to compute axes to contract over
                // for amalgam, these axes are the j's above?
                // for current node, these axes are the position of each source node in the vector of neighbours
                xarray<cd> target = view(gamma[current_index], target_bitstring[current_index], all());

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

        double get_probability(std::vector<uint32_t> &target_bitstring) {
            cd amplitude = get_amplitude(target_bitstring);
            return std::real(amplitude * std::conj(amplitude));
        }

        // returns probability of qubit being equal to value when measured
        double get_qubit_probability(const uint32_t &qubit, const uint32_t value) {
            // given order 0-1-2-3-4, contract from 0 to 1, then contract from 01 to 2, contract from 012 to 3, and so on
            uint32_t first_index = contraction_order[0];
            TN_Node first_node = get_node(first_index);
            // amalgam contains the current state of the contracted part, ready to contract its edges with the next node in the order
            xarray<cd> amalgam = gamma[first_index];

            std::set<uint32_t> amalgamated_nodes;
            amalgamated_nodes.insert(first_index);
            // edges_to_contract contains the edges going into the contracted component that have not yet been contracted
            std::vector<std::pair<uint32_t, uint32_t>> edges_to_contract;
            for (uint32_t target : first_node.get_neighbours()) {
                edges_to_contract.push_back(std::pair<uint32_t, uint32_t>(first_index, target));
            }
            for (uint32_t i = 1; i < contraction_order.size(); i++) {
                uint32_t current_index = contraction_order[i];
                TN_Node current_node = get_node(current_index);

                std::vector<uint32_t> new_neighbours = current_node.get_neighbours();

                std::vector<size_t> amalgam_axes;
                std::vector<size_t> current_node_axes;

                // Graph is simple, so should not have problems with multiple edges between two nodes
                for (uint32_t j = 0; j < edges_to_contract.size(); j++) {
                    // each edge which targets the current index is an edge we must contract at this step
                    if (edges_to_contract[j].second == current_index) {
                        // need to record indices of edges now that we have found the ones to contract
                        amalgam_axes.push_back(1 + j);
                        // calculate current_node index for this edge
                        current_node_axes.push_back(1 + current_node.get_index(edges_to_contract[j].first));
                    }
                }
                // second loop to remove edges which we have marked to contract
                for (size_t j : amalgam_axes) {
                    // removes edges that we have processed from edges to contract
                    std::vector<std::pair<uint32_t, uint32_t>>::iterator new_end = remove(edges_to_contract.begin(), edges_to_contract.end(), edges_to_contract[j]);
                    edges_to_contract.erase(new_end, edges_to_contract.end());
                }

                // carry out the contraction for this step between amalgam and the current node, need to compute axes to contract over
                // for amalgam, these axes are the j's above?
                // for current node, these axes are the position of each source node in the vector of neighbours
                xarray<cd> target = gamma[current_index];

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

            // return std::real(sum(amalgam * conj(amalgam))(0));

            amalgam = amalgam * conj(amalgam);

            xstrided_slice_vector slices;
            for (uint32_t i = 0; i < architecture->size(); i++) {
                if (i == qubit) {
                    slices.push_back(value);
                }
                else {
                    slices.push_back(all());
                }
            }

            amalgam = strided_view(amalgam, slices);

            return std::real(sum(amalgam)(0));
        }

        // measures the qubit, setting its value based on the probability that it should be 0 or 1
        void measure(uint32_t qubit) {
            double p0 = get_qubit_probability(qubit, 0);
            double rand_val = uniform_rand_in_range(0.0, 1.0);
            if (rand_val <= p0) {
                // then we have measured a 0, so update gamma accordingly
                std::vector<double> p0_v({p0});
                Gate update_gate(&update_to_0, true, &p0_v);
                apply_single_qubit_gate(qubit, update_gate);
            }
            else {
                // otherwise we have measured a 1, so update gamma accordingly
                std::vector<double> p1_v({1 - p0});
                Gate update_gate(&update_to_1, true, &p1_v);
                apply_single_qubit_gate(qubit, update_gate);
            }
        }
        
};

#endif