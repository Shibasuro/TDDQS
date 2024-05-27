#ifndef TDDCIRC
#define TDDCIRC
#include "tdd_arch.hpp"
#include "circuit.hpp"
#include <string>

// n qubit initial TDD state?
// That method would be necessary for a state evolution approach


// What about an arbitrary contraction order approach?
// Where we apply an arbitrary number of single and two qubit gates,
// applying these contractions in an arbitrary order

// Need to connect gates together by the indices which are alike in order to contract them
// this is actually quite challenging with this arbitrary axis contraction instead of index set contraction

// Different cases:
// single qubit gate contracted with single qubit gate - resulting shape is single qubit gate
// two qubit gate contracted with single qubit gate - resulting shape is two qubit gate
// two qubit gate contracted with two qubit gate (on one index) - resulting shape is three qubit gate?
// - this makes contraction of this form the most messy - hard to keep track of correct axes
// two qubit gate contracted with two qubit gate (on both indices) - resulting shape is two qubit gate

// Information we can reasonably expect to store
// TDDs for each gate - can store gate type and convert to TDD
// which qubits each gate should apply to
// some kind of ordering on gate applications when they are applied to the same qubits


// maybe can store the axis indices of edges? when representing the quantum circuit as a tensor network
// and then after every contraction, update the axis indices of the remaining edges
// So we represent the quantum circuit as a graph, where edges go from the earlier gate to the latter gate
// storing the axis of the edge in the first gate and the axis of the edge in the second gate
// Can be done with a contract gates method, which contracts all edges between two gates 
// updating the axis indices of remaining edges accordingly

// each edge can also be given a number (so that contraction order can be experimented with?)

// I can make use of my original instruction/gate model here from TNs
// at least to represent the circuit

// May require some modifications to actually contract the circuit however


// TODO construct a graph of tdds/gates to allow for arbitrary contraction order


// For now, use the state evolution approach (as this would be the approach when using SVD anyways)


// for now we will represent the initial state as a rank n TDD
class TDD_Circuit {
    private:
        uint32_t num_qubits;
        TDD state;
        std::vector<Instruction> instructions;
        // stores current axes to contract over for each qubit
        // currently not needed to be updated as we are updating in sequence?
        std::vector<uint16_t> contraction_axes;
    
        virtual void initialise_state(std::string bitstring = "", bool zero_init = true) {
            // for now this is just a rank n TDD, all qubits initialised to state 0
            svector<size_t> shape;
            xstrided_slice_vector slice;
            for (uint32_t i = 0; i < num_qubits; i++) {
                shape.push_back(2);
                if (zero_init || bitstring[i] == '0') {
                    slice.push_back(0);
                }
                else {
                    slice.push_back(1);
                }
                contraction_axes.push_back(i);
            }
            xarray<cd> tensor_state = zeros<cd>(shape);
            strided_view(tensor_state, slice) = cd(1,0);
            
            state = convert_tensor_to_TDD(tensor_state);
        }

        virtual void apply_instruction(Instruction instr) {
            if (instr.get_type() == Instr_type::GATE) {
                // then we are applying a gate
                if (instr.is_single_qubit_gate()) {
                    // then it is single qubit gate
                    xarray<cd> gate = instr.get_gate().get_gate();
                    TDD gate_TDD = convert_tensor_to_TDD(gate);
                    uint32_t target = instr.get_q1();
                    uint16_t target_axis = contraction_axes[target];

                    // this should leave axis in correct place
                    state = contract_tdds(state, gate_TDD, {target_axis}, {1});
                }
                else {
                    // otherwise its a two qubit gate
                    xarray<cd> gate = instr.get_gate().get_gate();
                    gate.reshape({2, 2, 2, 2});
                    // might need to swapaxes here to reshape from ijmn to imjn
                    // so that contraction below works better
                    gate = swapaxes(gate, 1, 2);

                    TDD gate_TDD = convert_tensor_to_TDD(gate);
                    uint32_t q1 = instr.get_q1();
                    uint32_t q2 = instr.get_q2();
                    uint16_t target_axis1 = contraction_axes[q1];
                    uint16_t target_axis2 = contraction_axes[q2];
                    // for contract_tdds, both sets of axes are assumed to be in ascending order
                    // this is not problematic this time, but may be problematic if 
                    // contraction axes order gets skewed in the future - TODO
                    if (q1 < q2) {
                        // bmcn, imjn -> bicj 
                        state = contract_tdds(state, gate_TDD, {target_axis1, target_axis2}, {1,3});
                    }
                    else {
                        // bmcn, imjn -> bicj
                        state = contract_tdds(state, gate_TDD, {target_axis2, target_axis1}, {1,3});
                    }
                }
            }
        }

    public:
        TDD_Circuit() {
            //do nothing
        }
        TDD_Circuit(uint32_t qubits) {
            num_qubits = qubits;
            initialise_state();
        }
        TDD_Circuit(uint32_t qubits, std::string bitstring) {
            num_qubits = qubits;
            initialise_state(bitstring, false);
        }

        void add_instruction(Instruction instr) {
            instructions.push_back(instr);
        }

        void simulate() {
            uint32_t i = 0;
            for (Instruction instr : instructions) {
                std::cout << "applying instr: " << i++ << std::endl;
                apply_instruction(instr);
            }
        }

        virtual cd get_amplitude(xarray<size_t> indices) {
            return state.get_value(indices);
        }

        void x(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&pauli_x_gate, true), q1));
        }

        void y(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&pauli_y_gate, true), q1));
        }

        void z(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&pauli_z_gate, true), q1));
        }

        void h(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&hadamard_gate, true), q1));
        }

        void s(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&phase_gate, true), q1));
        }

        void t(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&t_gate, true), q1));
        }

        void sx(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&sx_gate, true), q1));
        }

        void tdg(uint32_t q1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&t_dagger_gate, true), q1));
        }

        void rz(uint32_t q1, double theta) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&rz_gate, true, {theta}), q1));
        }

        void cx(uint32_t c1, uint32_t t1) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&controlled_not_gate, false), c1, t1));
        }

        void cu1(uint32_t c1, uint32_t t1, double lambda) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&controlled_u1_gate, false, {lambda}), c1, t1));
        }

        void swap(uint32_t q1, uint32_t q2) {
            add_instruction(Instruction(Instr_type::GATE, Gate(&swap_gate, false), q1, q2));
        }
        
        void toffoli(uint32_t c1, uint32_t c2, uint32_t t1) {
            h(t1);
            cx(c2, t1);
            tdg(t1);
            cx(c1, t1);
            t(t1);
            cx(c2, t1);
            tdg(t1);
            cx(c1, t1);
            t(t1);
            t(c2);
            h(t1);
            cx(c1, c2);
            t(c1);
            tdg(c2);
            cx(c1, c2);
        }
        void print_num_gates() {
            std::cout << instructions.size() << std::endl;
        }
};

class MPS_Circuit : public TDD_Circuit {
    private:
        uint32_t num_qubits;
        std::vector<TDD> state;
        std::vector<Instruction> instructions;
        // Lambda stores the connections between each qubit
        // for this need to update two qubit gate, amplitude, statevector etc
        // measurement probability
        std::vector<xarray<cd>> lambda;
        std::default_random_engine generator;

        uint32_t max_chi = std::numeric_limits<uint32_t>::max();

        double uniform_rand_in_range(double min, double max) {
            std::uniform_real_distribution<double> distribution(min, max);
            return distribution(generator);
        }

        void initialise_state(std::string bitstring = "", bool zero_init = true) override {
            // MPS State of N TDDs, all qubits initialised to state 0
            for (uint32_t i = 0; i < num_qubits; i++) {
                xarray<cd> tensor_state = zeros<cd>({2});
                if (zero_init || bitstring[i] == '0') {
                    tensor_state[0] = cd(1, 0);
                }
                else {
                    tensor_state[1] = cd(1,0);
                }
                std::vector<size_t> shape = {2,1};
                // this is because the edge nodes in MPS only have one adjacent node
                if (i != 0 && i != num_qubits - 1) {
                    shape.push_back(1);
                }
                tensor_state.reshape(shape);
                state.push_back(convert_tensor_to_TDD(tensor_state));
                if (i < num_qubits - 1) {
                    // initially just 1 by 1 identity
                    lambda.push_back({1.0});
                }
            }
            generator.seed(time(NULL));
        }

        void apply_instruction(Instruction instr) override {
            if (instr.get_type() == Instr_type::GATE) {
                // then we are applying a gate
                if (instr.is_single_qubit_gate()) {
                    // then it is single qubit gate, so this is same as original case, just contract
                    xarray<cd> gate = instr.get_gate().get_gate();
                    TDD gate_TDD = convert_tensor_to_TDD(gate);
                    uint32_t target = instr.get_q1();

                    // only need to update the qubit the gate is being applied to
                    state[target] = contract_tdds(gate_TDD, state[target], {1}, {0});
                }
                else {
                    // otherwise its a two qubit gate
                    // Note that it is assumed that two qubit gates are applied to adjacent qubits
                    xarray<cd> gate = instr.get_gate().get_gate();
                    gate.reshape({2, 2, 2, 2});

                    uint32_t q1 = instr.get_q1();
                    uint32_t q2 = instr.get_q2();
                    // make sure we always use the lower qubit number q1, and adjust the gate
                    // accordingly to make sure gate is still applied correctly?
                    if (q2 < q1) {
                        uint32_t temp = q2;
                        q2 = q1;
                        q1 = temp;
                        // if this is the case, just swap m,n and i,j in ijmn?
                        // that way it is effectively the same gate when we flip q1 and q2
                        gate = swapaxes(gate, {2}, {3});
                        gate = swapaxes(gate, {0}, {1});
                    }
                    TDD gate_TDD = convert_tensor_to_TDD(gate);

                    // now need to contract the two relevant MPS states state[q1] and state[q2]
                    // also contracting them with the gate itself
                    // THIS CAN BE DONE IN TDD space

                    // 1. Contract two relevant parts of MPS with gate

                    // either q1 < q2 or q2 < q1
                    // can also have q1/q2 = 0 or q1/q2 = num_qubits - 1
                    // in which case there is only one possible bond index
                    // for these purposes, will consider first index as being left one, second index
                    // as being right one (where q0 is on the far left), and then single bonds will 
                    // be determined by qubit number

                    // index will be either 1 or 2 as it is MPS, each is either {2,1} or {2,1,1}
                    uint16_t q1_to_q2_bond_index;
                    // q2 will always use bond index one in this case (as it is going left)
                    uint16_t q2_to_q1_bond_index = 1;
                    if (q1 == 0) {
                        q1_to_q2_bond_index = 1;
                    }
                    else {
                        q1_to_q2_bond_index = 2;
                    }
                    
                    std::vector<size_t> shape1 = state[q1].get_shape();
                    std::vector<size_t> shape2 = state[q2].get_shape();

                    // Record the old dimensions that will not be contracted
                    uint32_t old_chi1 = 1;
                    uint32_t old_chi2 = 1;
                    if (shape1.size() == 3) {
                        if (q1_to_q2_bond_index == 1) {
                            old_chi1 = shape1[2];
                        }
                        else {
                            old_chi1 = shape1[1];
                        }
                    }

                    if (shape2.size() == 3) {
                        // always the case as q1 < q2
                        old_chi2 = shape2[2];
                    }
                    xarray<cd> temp1 = convert_TDD_to_tensor(state[q1]);
                    state[q1] = convert_tensor_to_TDD(temp1);
                    xarray<cd> temp2 = convert_TDD_to_tensor(state[q2]);
                    state[q2] = convert_tensor_to_TDD(temp2);
                    // absorb lambdas
                    if (q1 > 0) {
                        xarray<cd> left = diag(lambda[q1 - 1]);
                        TDD lambda_left = convert_tensor_to_TDD(left);
                        state[q1] = contract_tdds(state[q1], lambda_left, {1}, {1});
                    }
                    if (q2 < num_qubits - 1) {
                        xarray<cd> right = diag(lambda[q2]);
                        TDD lambda_right = convert_tensor_to_TDD(right);
                        state[q2] = contract_tdds(state[q2], lambda_right, {2}, {0});
                    }
                    xarray<cd> bond = diag(lambda[q1]);
                    TDD lambda_bond = convert_tensor_to_TDD(bond);
                    state[q1] = contract_tdds(state[q1], lambda_bond, {q1_to_q2_bond_index}, {0});

                    // intermediate seems to have wrong norm, is this due to contractions with lambdas?
                    // or due to contractions with each other?
                    temp1 = convert_TDD_to_tensor(state[q1]);
                    state[q1] = convert_tensor_to_TDD(temp1);
                    temp2 = convert_TDD_to_tensor(state[q2]);
                    state[q2] = convert_tensor_to_TDD(temp2);

                    // TODO? More efficient method might be to support reshaping of TDDs - not clear how to achieve this
                    // Contract the two parts of the MPS state
                    TDD intermediate = contract_tdds(state[q1], state[q2], {q1_to_q2_bond_index}, {q2_to_q1_bond_index});
                    // mab, nbc -> manc
                    // or have mb, nbc -> mnc


                    // Contract intermediate with gate (ijmn) (important thing is to contract over m and n)
                    // gate contraction index is the same as the q1_to_q2_bond_index
                    TDD theta = contract_tdds(intermediate, gate_TDD, {0, q1_to_q2_bond_index}, {2, 3});
                    // manc, ijmn -> ijac
                    // mnc, ijmn -> ijc
                    // effectively want final output to be iajc or i(1)jc
                    // so we need to do reshapes/swapaxes

                    // 2. Convert TDD to tensor for SVD

                    xarray<cd> theta_tensor = convert_TDD_to_tensor(theta);
                    // now that it is a tensor, we can apply the necessary reshape/axis swaps to get ibjc
                    // the reshapes handle the case where old_chi1 or old_chi2 is 1 before doing the axis swap
                    if (q1_to_q2_bond_index == q2_to_q1_bond_index) {
                        // then ijc, reshape to ij(1)c
                        theta_tensor.reshape({2, 2, old_chi1, old_chi2});
                    }
                    // swap (1) and j or j and a to get required form
                    theta_tensor = swapaxes(theta_tensor, 1, 2);

                    // flatten to square matrix for SVD
                    theta_tensor.reshape({2 * old_chi1, 2 * old_chi2});

                    // 3. SVD 
                    // SVD returns unitary matrices as U and V
                    std::tuple<xarray<cd>, xarray<cd>, xarray<cd>> svd = linalg::svd(theta_tensor);
                    xarray<cd> u = get<0>(svd);
                    xarray<cd> s = get<1>(svd);
                    xarray<cd> v = get<2>(svd);

                    // 4. SVD Culling and Renormalisation (outputs new u, s, v)
                    // remove values that come from floating point errors;
                    double double_error = 1e-16;
                    s = filter(s, real(s) > double_error);
                    // Remove smallest schmidt coefficients such that sum of squares is less
                    // than truncation threshold
                    double truncation_threshold = 1e-16;
                    // TODO s_norm manages to become != 1 at some points for some reason
                    std::cout << "s_norm: " << std::real(sum(s*s)(0)) << std::endl;
                    uint32_t new_chi = s.size();
                    double sum_squares = 0;
                    for (uint32_t i = new_chi - 1; i > 0; i--) {
                        if ((sum_squares + std::norm(s(i))) < truncation_threshold) {
                            if (i == 1) {
                                new_chi = 1;
                            }
                            sum_squares += std::norm(s(i));
                        }
                        else {
                            new_chi = i + 1;
                            break;
                        }
                    }

                    // if this would push us over the limit, then we restrict to max_chi at cost of fidelity
                    if (new_chi > max_chi) {
                        new_chi = max_chi;
                    }
                    std::cout << "new_chi: " << new_chi << std::endl;
                    xarray<cd> u_prime = view(u, all(), range(0, new_chi));
                    xarray<cd> s_prime = view(s, range(0, new_chi));
                    xarray<cd> v_prime = view(v, range(0, new_chi), all());

                    // renormalise s_prime
                    s_prime *= pow( sum(s * s) / sum(s_prime * s_prime), 0.5);

                    // 5. Calculate new tensors for q1 and q2
                    xarray<cd> q1_prime = u_prime;//linalg::dot(u_prime, diag(s_prime));
                    lambda[q1] = s_prime;
                    xarray<cd> q2_prime = v_prime;
                    
                    // these matrices come out as (2*old_chi1, new_chi)
                    // and (new_chi, 2*old_chi2)

                    if (shape1.size() > 2) {
                        q1_prime.reshape({2, old_chi1, new_chi});
                    }

                    q2_prime = swapaxes(q2_prime, 0, 1);
                    if (shape2.size() > 2) {
                        q2_prime.reshape({2, old_chi2, new_chi});
                        q2_prime = swapaxes(q2_prime, 1, 2);
                    }

                    if (q1 > 0) {
                        xarray<cd> lambda_left = diag(1.0 / lambda[q1 - 1]);
                        q1_prime = linalg::tensordot(q1_prime, lambda_left, {1}, {1});
                        q1_prime = swapaxes(q1_prime, {1}, {2});
                    }
                    if (q2 < num_qubits - 1) {
                        xarray<cd> lambda_right = diag(1.0 / lambda[q2]);
                        q2_prime = linalg::tensordot(q2_prime, lambda_right, {2}, {0});
                    }

                    // 6. Convert back to TDD and update the state
                    state[q1] = convert_tensor_to_TDD(q1_prime);
                    state[q2] = convert_tensor_to_TDD(q2_prime);
                    
                }
            }
            else if (instr.get_type() == Instr_type::MEASUREMENT) {
                uint32_t target = instr.get_q1();
                measure(target);
            }
        }
    public:
        MPS_Circuit(uint32_t qubits) {
            TDD_Circuit();
            num_qubits = qubits;
            initialise_state();
        }

        MPS_Circuit(uint32_t qubits, std::string bitstring) {
            TDD_Circuit();
            num_qubits = qubits;
            initialise_state(bitstring, false);
        }

        // this constructor provides an option to restrict the maximum bond dimension at the cost of fidelity
        MPS_Circuit(uint32_t qubits, std::string bitstring, uint32_t max_bd) {
            TDD_Circuit();
            num_qubits = qubits;
            initialise_state(bitstring, false);
            max_chi = max_bd;
        }

        void print_mps_state() {
            for (uint32_t i = 0; i < num_qubits; i++) {
                std::cout << "state for qubit " << i << ": " << std::endl;
                xarray<cd> temp = convert_TDD_to_tensor(state[i]);
                std::cout << temp << std::endl;
                state[i] = convert_tensor_to_TDD(temp);
            }
        }

        // This is just to delete all the TDDs and see if any nodes/edges are leftover - which
        // would suggest somewhere cleanup is failing
        void cleanup() {
            for (uint32_t i = 0; i < num_qubits; i++) {
                state[i].cleanup();
            }
        }

        uint32_t get_num_qubits() {
            return num_qubits;
        }

        cd get_amplitude(xarray<size_t> indices) override {
            // these indices should all be 0 or 1 (as they are the physical indices
            // and thus dimension 2)
            TDD amalgam = state[0].get_child_TDD(indices[0]);
            for (size_t i = 1; i < num_qubits; i++) {
                // always absorb left lambda first
                xarray<cd> left = diag(lambda[i - 1]);
                TDD lambda_left = convert_tensor_to_TDD(left);
                // leave as false as we do not want to delete the old state[i]
                TDD next_state = contract_tdds(state[i], lambda_left, {1}, {1}, 0, false);
                lambda_left.cleanup();
                TDD next_to_contract = next_state.get_child_TDD(indices[i]);
                amalgam = contract_tdds(amalgam, next_to_contract, {0}, {0}, 0, false);
                next_state.cleanup();
            }
            return amalgam.get_weight();
        }

        // this is incorrect but also inefficient - TODO delete when no longer useful
        double get_inefficient_qubit_probability(uint16_t qubit, uint32_t val) {
            // old, inefficient method
            // this is not efficient as we lose the compression of MPS
            // (2,a) (2,a,b) (2,b,c) (2,c)
            // (2,2,b) (2,b,c) (2,c)
            // (2,2,2,c) (2,c)
            // (2,2,2,2)
            TDD amalgam = state[0];
            for (uint16_t i = 1; i < num_qubits; i++) {
                // always absorb left lambda first
                xarray<cd> left = diag(lambda[i - 1]);
                TDD lambda_left = convert_tensor_to_TDD(left);
                // leave as false as we do not want to delete the old state[i]
                TDD next_to_contract = contract_tdds(state[i], lambda_left, {1}, {1}, 0, false);
                lambda_left.cleanup();
                amalgam = contract_tdds(amalgam, next_to_contract, {i}, {1}, 0, false);
            }
            // this gives the statevector in amalgam
            // now compute sum of relevant amplitudes
            return amalgam.get_probability_sum(qubit, val);
        }

        // NOTE THAT THIS DOES NOT CLEANUP NODES AND EDGES - this is fine though as 
        // if we convert to statevector we lose all compression anyway
        xarray<cd> get_statevector() {
            TDD amalgam = state[0];
            for (uint16_t i = 1; i < num_qubits; i++) {
                xarray<cd> left = diag(lambda[i - 1]);
                TDD lambda_left = convert_tensor_to_TDD(left);
                // leave as false as we do not want to delete the old state[i]
                TDD next_to_contract = contract_tdds(state[i], lambda_left, {1}, {1}, 0, false);
                lambda_left.cleanup();
                amalgam = contract_tdds(amalgam, next_to_contract, {i}, {1}, 0, false);
            }
            // statevector is in amalgam, just convert to tensor
            return convert_TDD_to_tensor(amalgam);
        }

        // val can be either 0 or 1
        // MORE EFFICIENT METHOD HERE
        // TODO can be extended to support arbitrary expectation values (so long as they are separable)
        double get_qubit_probability(uint16_t qubit, uint32_t val) {
            // calculate initial term, contracting from the right
            // TODO investigate whether this is sufficiently efficient (for higher bond dimension its slow)

            // TODO need to absorb lambdas here somewhere
            TDD q_current;
            if (qubit == num_qubits - 1) {
                q_current = kronecker_conjugate(state[num_qubits - 1].get_child_TDD(val));
            }
            else {
                std::vector<TDD> tdds;
                TDD child0 = state[num_qubits - 1].get_child_TDD(0);
                TDD child1 = state[num_qubits - 1].get_child_TDD(1);
                tdds.push_back(kronecker_conjugate(child0));
                tdds.push_back(kronecker_conjugate(child1));
                q_current = add_tdds(tdds);
            }
            for (uint32_t j = 0; j < num_qubits - 1; j++) {
                uint32_t i = num_qubits - 2 - j;
                TDD q_temp;
                TDD current_state = state[i];
                //absorb right lambda into the state, since i will be between 0 and num_qubits - 2
                xarray<cd> right = diag(lambda[i]);
                TDD lambda_right = convert_tensor_to_TDD(right);
                // dont delete current_state here as we need to preserve state[q2]
                uint16_t bond_index = 2;
                if (i == 0) {
                    bond_index = 1;
                }
                TDD next_state = contract_tdds(current_state, lambda_right, {bond_index}, {0}, 0, false);
                // but we can cleanup lambda_right here
                lambda_right.cleanup();
                if (i == qubit) {
                    // then we can index by val
                    q_temp = kronecker_conjugate(next_state.get_child_TDD(val));
                }
                else {
                    std::vector<TDD> tdds;
                    TDD child0 = next_state.get_child_TDD(0);
                    TDD child1 = next_state.get_child_TDD(1);
                    tdds.push_back(kronecker_conjugate(child0));
                    tdds.push_back(kronecker_conjugate(child1));
                    q_temp = add_tdds(tdds);
                }
                // clean up next_state
                next_state.cleanup();
                if (i > 0) {
                    q_current = contract_tdds(q_temp, q_current, {1, 3}, {0, 1});
                }
                else {
                    q_current = contract_tdds(q_temp, q_current, {0, 1}, {0, 1});
                }
            }
            // now q_current should contain the final weight
            double probability = std::real(q_current.get_weight());
            return probability;
        }

        // TODO deprecate as this is inefficient
        std::vector<double> get_qubit_probabilities(std::vector<uint16_t> qubits, std::vector<uint32_t> vals) {
            TDD amalgam = state[0];
            for (uint16_t i = 1; i < num_qubits; i++) {
                TDD next_to_contract = state[i];
                amalgam = contract_tdds(amalgam, next_to_contract, {i}, {1}, 0, false);
            }
            std::vector<double> probabilities;
            for (uint32_t i = 0; i < qubits.size(); i++) {
                probabilities.push_back(amalgam.get_probability_sum(qubits[i], vals[i]));
            }
            return probabilities;
        }

        void measure(uint16_t qubit) {
            double p0 = get_qubit_probability(qubit, 0);
            double rand_val = uniform_rand_in_range(0.0, 1.0);
            if (rand_val <= p0) {
                // then we have measured a 0, so update gamma accordingly
                std::vector<double> p0_v({p0});
                Gate update_gate(&update_to_0, true, p0_v);
                apply_instruction(Instruction(Instr_type::GATE, update_gate, qubit));
            }
            else {
                // otherwise we have measured a 1, so update gamma accordingly
                std::vector<double> p1_v({1 - p0});
                Gate update_gate(&update_to_1, true, p1_v);
                apply_instruction(Instruction(Instr_type::GATE, update_gate, qubit));
            }
        }
};



#endif 