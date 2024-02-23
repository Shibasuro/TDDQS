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
        std::vector<uint8_t> contraction_axes;
    
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
                    uint8_t target_axis = contraction_axes[target];

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
                    uint8_t target_axis1 = contraction_axes[q1];
                    uint8_t target_axis2 = contraction_axes[q2];
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
            for (Instruction instr : instructions) {
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
};

class MPS_Circuit : public TDD_Circuit {
    private:
        uint32_t num_qubits;
        std::vector<TDD> state;
        std::vector<Instruction> instructions;
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
            }
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
                    state[target] = contract_tdds(state[target], gate_TDD, {0}, {1});
                }
                else {
                    // otherwise its a two qubit gate
                    // Note that it is assumed that two qubit gates are applied to adjacent qubits
                    xarray<cd> gate = instr.get_gate().get_gate();
                    gate.reshape({2, 2, 2, 2});

                    TDD gate_TDD = convert_tensor_to_TDD(gate);
                    uint32_t q1 = instr.get_q1();
                    uint32_t q2 = instr.get_q2();

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
                    uint8_t q1_to_q2_bond_index;
                    uint8_t q2_to_q1_bond_index;
                    uint8_t gate_contraction_index = 1;
                    if (q1 < q2) {
                        if (q1 == 0) {
                            q1_to_q2_bond_index = 1;
                        }
                        else {
                            q1_to_q2_bond_index = 2;
                            gate_contraction_index = 2;
                        }
                        // q2 will always use bond index one in this case (as it is going left)
                        q2_to_q1_bond_index = 1;
                    }
                    else if (q2 < q1) {
                        //q1 will always use bond index one in this case (as it is going left)
                        q1_to_q2_bond_index = 1;
                        if (q2 == 0) {
                            q2_to_q1_bond_index = 1;
                        }
                        else {
                            q2_to_q1_bond_index = 2;
                        }
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
                        if (q2_to_q1_bond_index == 1) {
                            old_chi2 = shape2[2];
                        }
                        else {
                            old_chi2 = shape2[1];
                        }
                    }

                    // TODO? More efficient method might be to support reshaping of TDDs - not clear how to achieve this

                    // Contract the two parts of the MPS state
                    TDD intermediate = contract_tdds(state[q1], state[q2], {q1_to_q2_bond_index}, {q2_to_q1_bond_index});
                    // mab, ncd ->
                    // possibilities (1,1), (1,2), (2,1)
                    // mnbd, mncb, mand

                    // Contract intermediate with gate (ijmn) (important thing is to contract over m and n)
                    TDD theta = contract_tdds(intermediate, gate_TDD, {0, gate_contraction_index}, {2, 3});
                    // mnbd, ijmn -> ijbd
                    // mncb, ijmn -> ijcb
                    // mand, ijmn -> ijad
                    // effectively want final output to be ibjc, ibjd or iajd

                    // 2. Convert TDD to tensor for SVD

                    xarray<cd> theta_tensor = convert_TDD_to_tensor(theta);
                    // now that it is a tensor, we can apply the necessary reshape to get ibjc or equivalent
                    if (q1_to_q2_bond_index == q2_to_q1_bond_index) {
                        // then ijbd, reshape to ibjd
                        theta_tensor.reshape({2, 2, old_chi1, old_chi2});
                    }
                    else if (q2_to_q1_bond_index == 2) {
                        // then ijcb, need to reshape to ibjc
                        theta_tensor.reshape({2, 2, old_chi2, old_chi1});
                        theta_tensor = swapaxes(theta_tensor, 2, 3);
                    }
                    else {
                        // then ijad, reshape to iajd
                        theta_tensor.reshape({2, 2, old_chi1, old_chi2});
                    }
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
                    double double_error = 0.00001;
                    filtration(u, real(u * conj(u)) < double_error) = 0;
                    xarray<cd> temp_s = filter(s, real(s) > double_error);
                    filtration(v, real(v * conj(v)) < double_error) = 0;

                    uint32_t new_chi = temp_s.size();
                    xarray<cd> u_prime = view(u, all(), range(0, new_chi));
                    xarray<cd> s_prime = view(s, range(0, new_chi));
                    xarray<cd> v_prime = view(v, range(0, new_chi), all());

                    // renormalise s_prime
                    s_prime *= pow( sum(s * s) / sum(s_prime * s_prime), 0.5);

                    // 5. Calculate new tensors for q1 and q2
                    xarray<cd> q1_prime = linalg::dot(u_prime, diag(s_prime));
                    xarray<cd> q2_prime = v_prime;

                    // apply reshape to ensure we have (2, new_chi, other) etc.

                    if (shape1.size() == 2) {
                        q1_prime.reshape({2, new_chi});
                    }
                    else {
                        if (q1_to_q2_bond_index == 1) {
                            q1_prime.reshape({2, new_chi, shape1[2]});
                        }
                        else {
                            q1_prime.reshape({2, shape1[1], new_chi});
                        }
                    }

                    if (shape2.size() == 2) {
                        q2_prime.reshape({2, new_chi});
                    }
                    else {
                        if (q2_to_q1_bond_index == 1) {
                            q2_prime.reshape({2, new_chi, shape2[2]});
                        }
                        else {
                            q2_prime.reshape({2, shape2[1], new_chi});
                        }
                    }

                    // 6. Convert back to TDD and update the state
                    state[q1] = convert_tensor_to_TDD(q1_prime);
                    state[q2] = convert_tensor_to_TDD(q2_prime);
                    
                }
            }
        }
    public:
        MPS_Circuit(uint32_t qubits) {
            TDD_Circuit();
            num_qubits = qubits;
            initialise_state();
        }

        uint32_t get_num_qubits() {
            return num_qubits;
        }

        cd get_amplitude(xarray<size_t> indices) override {
            // TODO implement the contraction of the MPS TDD state
            // these indices should all be 0 or 1 (as they are the physical indices
            // and thus dimension 2)
            TDD amalgam = state[0].get_child_TDD(indices[0]);
            for (size_t i = 1; i < num_qubits; i++) {
                TDD next_to_contract = state[i].get_child_TDD(indices[i]);
                amalgam = contract_tdds(amalgam, next_to_contract, {0}, {0});
            }
            return amalgam.get_weight();
        }
};



#endif 