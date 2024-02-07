#ifndef TDDARCH
#define TDDARCH
#include <vector>
#include <iostream>
#include <queue>
#include "xtensor/xarray.hpp"
#include <xtensor/xstrided_view.hpp>

typedef std::complex<double> cd;
using namespace xt;

class TDD_Node {
    private:
        uint8_t axis_index;
        // how to identify parents to propagate weights upwards?
        std::vector<TDD_Node *> parents;
        // how to group identical successor weight pairs?
        std::vector<TDD_Node> successors;
        std::vector<cd> weights;
        
        // Top level hash table to store all nodes? 
        // TODO: NEEDS MORE THOUGHT
        // Need to ensure nodes contain sufficient information to propagate weights upwards
        // Requires
        // -- reference to parents
        // -- identification for which weights to update in parent data
        // Need to ensure successors with the same weight and target node can be merged
        // And that successors which are the same node but with different weight can also be supported
        // Requires
        // -- Way to keep track of which indices correspond to the same target
        // -- Way to ensure only one instance of the target node exists (or can be merged in reduction process)


    
    public:
        void initialise_vectors(size_t axis_dimension) {
            for (size_t i = 0; i < axis_dimension; i++) {
                TDD_Node succ(axis_index + 1, this);
                successors.push_back(succ);
                weights.push_back(1);
            }
        }

        TDD_Node(){}
        TDD_Node(uint8_t axis, TDD_Node *parent) {
            axis_index = axis;
            parents.push_back(parent);
        }
        TDD_Node(uint8_t axis, size_t axis_dim) {
            axis_index = axis;
            initialise_vectors(axis_dim);
        }

        TDD_Node * get_successor_ref(size_t index) {
            return &successors[index];
        }

        cd get_weight(size_t index) {
            return weights[index];
        }

        void set_terminal_weight(cd weight) {
            //weights.push_back(weight);
            // need to carry out normalisation here, so propagate this new weight up to it's parents?
        }
        
        bool is_terminal() {
            return successors.empty();
        }
        
        // TODO: need to update this propagation to trim more than 1 index if nodes have been merged
        cd get_value (xarray<uint32_t> indices) {
            if (is_terminal()) {
                return 1;
            }
            if (indices.size() == 0) {
                // this would mean an illegal index set (too short)
                return -2;
            }
            uint32_t current_index = indices[0];
            if (weights[current_index] == cd{0,0}) {
                return 0;
            }
            xarray<uint32_t> new_indices = view(indices, range(1, indices.size()));
            return weights[current_index] * successors[current_index].get_value(new_indices);
        }
};

class TDD {
    private: 
        TDD_Node root;
        cd in_weight;
    public:
        TDD() {}
        TDD(TDD_Node r, cd weight) {
            root = r;
            in_weight = weight;
        }
        // TODO: need to update this propagation to trim more than 1 index if nodes have been merged
        cd get_value(xarray<uint32_t> indices) {
            if (root.is_terminal()) {
                return in_weight;
            }
            if (in_weight == cd{0,0}) {
                return 0;
            }
            xarray<uint32_t> new_indices = view(indices, range(1, indices.size()));
            return in_weight * root.get_weight(indices[0]) * root.get_successor_ref(indices[0])->get_value(new_indices);
        }
        

};

// iterative implementation to convert an arbitrary tensor into a TDD
TDD convert_tensor_to_TDD(xarray<cd> tensor) {
    svector<size_t> shape = tensor.shape();
    TDD_Node root(0, shape[0]);
    cd weight = 1;
    std::queue<xarray<cd>> tensors_to_process;
    tensors_to_process.push(tensor);

    std::queue<TDD_Node *> nodes_to_process;
    nodes_to_process.push(&root);

    uint32_t count = 1; 
    while (!tensors_to_process.empty()) {
        xarray<cd> current_tensor = tensors_to_process.front();
        tensors_to_process.pop();
        svector<size_t> current_shape = current_tensor.shape();
        size_t current_dimension = current_shape[0];

        TDD_Node *current_node = nodes_to_process.front();
        nodes_to_process.pop();
        
        if (current_tensor.size() == 1) {
            current_node->set_terminal_weight(current_tensor[0]);
        }
        else {
            current_node->initialise_vectors(current_dimension);
            // create new tensor views and tdd_nodes to process based on value of current axis index
            xt::xstrided_slice_vector sv;
            for (size_t i = 0; i < current_shape.size(); i++) {
                sv.push_back(xt::all());
            }
            for (size_t i = 0; i < current_dimension; i++) {
                sv[0] = i;
                xarray<cd> new_tensor = strided_view(current_tensor, sv);
                tensors_to_process.push(new_tensor);

                nodes_to_process.push(current_node->get_successor_ref(i));
                count++;
            }
        }
    }
    std::cout << "Nodes created: " << count << std::endl;

    return TDD(root, weight);
}

#endif