#ifndef TDDCIRC
#define TDDCIRC
#include "tdd_arch.hpp"
#include "circuit.hpp"

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
    
        void initialise_state() {
            // for now this is just a rank n TDD, all qubits initialised to state 0
            svector<size_t> shape;
            xstrided_slice_vector slice;
            for (uint32_t i = 0; i < num_qubits; i++) {
                shape.push_back(2);
                slice.push_back(0);
                contraction_axes.push_back(i);
            }
            xarray<cd> tensor_state = zeros<cd>(shape);
            strided_view(tensor_state, slice) = cd(1,0);
            
            state = convert_tensor_to_TDD(tensor_state);
        }

        void apply_instruction(Instruction instr) {
            if (instr.get_type() == Instr_type::GATE) {
                // then we are applying a gate
                if (instr.is_single_qubit_gate()) {
                    // then it is single qubit gate
                    xarray<cd> gate = instr.get_gate().get_gate();
                    TDD gate_TDD = convert_tensor_to_TDD(gate);
                    uint32_t target = instr.get_q1();
                    uint8_t target_axis = contraction_axes[target];

                    // this should leave axis in correct place
                    // something might still be wrong with this
                    // WHAT APPEARS TO BE HAPPENING
                    // the new axis ends up at the end - thus why everything cancels with target axis 2
                    // HOWEVER this issue does not occur if the initial state is 1???
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
                        // imjn, bmcn -> ibjc // which is what we want
                        state = contract_tdds(gate_TDD, state, {1,3}, {target_axis1, target_axis2});
                    }
                    else {
                        // imjn, bmcn -> ibjc // which is what we want
                        state = contract_tdds(gate_TDD, state, {1,3}, {target_axis2, target_axis1});
                    }

                }
            }
        }

    public:
        TDD_Circuit(uint32_t qubits) {
            num_qubits = qubits;
            initialise_state();
        }

        void add_instruction(Instruction instr) {
            instructions.push_back(instr);
        }

        void simulate() {
            for (Instruction instr : instructions) {
                apply_instruction(instr);
            }
        }

        cd get_amplitude(xarray<size_t> indices) {
            return state.get_value(indices);
        }
};



#endif 