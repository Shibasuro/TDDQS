#ifndef TDDARCH
#define TDDARCH
#include <vector>
#include <iostream>
#include <queue>
#include <functional>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <math.h>
#include "xtensor/xarray.hpp"
#include <xtensor/xstrided_view.hpp>
#include "utils.hpp"

// FIXED buffer overflow - specifically on large circuits such as qft15/qft16
// This is because of the potentialy for high reference counts for node and edge duplication
// expanded to uint32_t to fix

typedef std::complex<double> cd;
using namespace xt;

class TDD_Node;

class TDD_Edge {
    private:
        const TDD_Node *target;
        cd weight;

    public:
        TDD_Edge(const TDD_Node *t, cd w) {
            target = t;
            weight = w;
        }
        const TDD_Node *get_target() const {
            return target;
        }

        cd get_weight() const {
            return weight;
        }

        bool operator==(const TDD_Edge &other) const {
            return (target == other.get_target() && is_approx_equal(weight, other.get_weight()));
        }
};

class TDD_Node {
    private:
        uint16_t axis_index;
        std::vector<const TDD_Edge *> successors;

        // trade off between maintaining full vector of successors (dimension * 64 bits)
        // and maintaining a vector of indices (mapping real index to successor index)
        // which would require space d * MAX_INDEX_SIZE + d' * 64 (assuming 64 bit pointers)
        // MAX_INDEX_SIZE would be the number of bits required to store the max dimension (probably 16 bits)
        // d' is the number of distinct successors
        // d' = (p/q)d, p < q
        // so space is 16d + 64(p/q)d vs 64d
        // (p/q) < 3/4
        // exact memory saving = 16 * (3d - 4d') bits, this means we do not save memory if d' > 3/4d
        // worst case cost is then 5/4 * original cost (when d = d')
        // this requires quite a bit of redundancy, but also enables more redundancy exploitation

        // unlocks the potential for index vectors and successor vectors to be hash mapped

        // alternative approach would require the below:
        // std::vector<uint16_t> successor_indices;
        // std::vector<const TDD_Edge *> successors;
        // in addition to updating any weight retrieval, successor addition, successor retrieval
        // and deletion
        // TODO further investigation


        // RR5 - Reduction rule 5 involves eliminating redundant edges 
        // this approach would reduce the number of pointers stored
        // duplicate edges are already handled by the edge map

    public:

        TDD_Node(){}
        TDD_Node(uint16_t axis) {
            axis_index = axis;
        }

        const TDD_Edge *get_successor_ref(size_t index) const {
            return successors[index];
        }
        
        bool is_terminal() const {
            return successors.empty();
        }

        uint16_t get_axis_index() const {
            return axis_index;
        }

        const TDD_Edge *add_successor(const TDD_Node *t, cd w);

        std::vector<const TDD_Edge *> get_successors() const {
            return successors;
        }
        
        void set_successors(std::vector<const TDD_Edge *> succs) {
            successors = succs;
        }

        size_t get_dimension() const {
            return successors.size();
        }

        cd get_weight(size_t index) const {
            if (is_terminal()) {
                return cd(1,0);
            }
            return successors[index]->get_weight();
        }

        void clear_successors() const;

        void delete_edges() const;

        void cleanup() const;

        void cleanup_duplicates() const;

        void duplicate() const;
        
        cd get_value(xarray<size_t> indices) const {
            if (is_terminal()) {
                return cd(1,0);
            }
            if (indices.size() == 0) {
                // this would mean an illegal index set (too short)
                // TODO HANDLE ELEGANTLY INSTEAD OF RETURNING WEIRD VALUE
                return cd(-2,0);
            }
            size_t current_index = indices[axis_index];
            if (get_weight(current_index) == cd{0,0}) {
                return cd(0,0);
            }
            const TDD_Node *next = get_successor_ref(current_index)->get_target();
            return get_weight(current_index) * next->get_value(indices);
        }

        double get_probability_sum(uint16_t qubit, uint32_t val, std::vector<size_t> shape) const {
            if (is_terminal()) {
                return 1;
            }

            uint16_t axis_index = get_axis_index();
            double prob_sum = 0;
            if (axis_index == qubit) {
                const TDD_Node *next = get_successor_ref(val)->get_target();
                cd weight = get_weight(val);
                double w = std::real(std::conj(weight) * weight);
                double prod = 1;
                std::vector<size_t> new_shape = std::vector<size_t>(shape.begin() + 1, shape.end());
                if (next->is_terminal()) {
                    for (uint32_t i = 1; i < shape.size(); i++) {
                        prod *= shape[i];
                    }
                }
                else {
                    uint16_t next_axis_index = next->get_axis_index();
                    uint16_t diff = next_axis_index - axis_index;
                    // account for skipped axes
                    for (uint32_t i = 1; i < diff; i++) {
                        prod *= shape[i];
                    }
                    new_shape = std::vector<size_t>(shape.begin() + diff, shape.end());
                }
                return w * prod * next->get_probability_sum(qubit, val, new_shape);
            }
            for (uint32_t i = 0; i < get_dimension(); i++) {
                const TDD_Node *next = get_successor_ref(i)->get_target();
                cd weight = get_weight(i);
                double w = std::real(std::conj(weight) * weight);
                std::vector<size_t> new_shape = std::vector<size_t>(shape.begin() + 1, shape.end());
                double prod = 1;
                if (next->is_terminal()) {
                    for (uint32_t i = 1; i < shape.size(); i++) {
                        // if we skip the qubit, need to account for that
                        if (axis_index + i != qubit) {
                            prod *= shape[i];
                        }
                    }
                }
                else {
                    uint16_t next_axis_index = next->get_axis_index();
                    uint16_t diff = next_axis_index - axis_index;
                    // account for skipped axes
                    for (uint32_t i = 1; i < diff; i++) {
                        // if we skip the valued qubit, then don't multiply by that layer
                        if (axis_index + i != qubit) {
                            prod *= shape[i];
                        }
                    }
                    new_shape = std::vector<size_t>(shape.begin() + diff, shape.end());
                }
                prob_sum += w * prod * next->get_probability_sum(qubit, val, new_shape);
            }

            return prob_sum;
        }

        bool operator==(const TDD_Node &other) const {
            if (axis_index != other.get_axis_index()){
                return false;
            }
            if (get_dimension() != other.get_dimension()) {
                return false;
            }
            for (size_t i = 0; i < get_dimension(); i++) {
                if (get_successor_ref(i) != other.get_successor_ref(i)) {
                    return false;
                }
            }
            return true;
        }
};

template <>
struct std::hash<TDD_Edge> {
    std::size_t operator()(const TDD_Edge& t) const {
        return ((hash<double>()(t.get_weight().real()) ^ (hash<double>()(t.get_weight().imag()) << 1) >> 1) ^ (hash<const TDD_Node *>()(t.get_target())) << 1);
    }
};
template <>
struct std::hash<TDD_Node> {
    std::size_t operator()(const TDD_Node& t) const {
        size_t total = 0;
        for (size_t i = 0; i < t.get_dimension(); i++) {
            size_t x = hash<const TDD_Edge *>()(t.get_successor_ref(i)) << (i % 3);
            total ^= x + 0x9e3779b9 + (total << 6) + (total >> 2);
        }
        return total ^ (hash<uint16_t>()(t.get_axis_index()) << 1);
        // size_t seed = t.get_dimension();
        // for (size_t i = 0; i < t.get_dimension(); i++) {
        //     const TDD_Edge* edge = t.get_successor_ref(i);
        //     size_t x = (size_t)(edge);
        //     x = ((x >> 16) ^ x) * 0x45d9f3b;
        //     x = ((x >> 16) ^ x) * 0x45d9f3b;
        //     x = (x >> 16) ^ x;
        //     seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        // }
        // return seed ^ (hash<uint16_t>()(t.get_axis_index()) << 1);
    }
};

// map to store all the edges and nodes that are defined throughout the process
class TDD_Map {
    private:
        // maps contain the item itself, and maps to the reference count
        std::unordered_map<TDD_Edge, uint32_t> edge_map;
        std::unordered_map<TDD_Node, uint32_t> node_map;

        // unique terminal node is defined here, -1 is chosen as axis index to ensure it is always last
        TDD_Node terminal_node = TDD_Node(-1);

        uint64_t max_nodes = 0;
        uint64_t max_edges = 0;

        uint64_t max_memory_usage = 0;
        uint64_t approximate_memory_usage = 0;
        double time = 0;

        void check_nodes() {
            if (node_map.size() > max_nodes) {
                max_nodes = node_map.size();
            }
            if (approximate_memory_usage > max_memory_usage) {
                max_memory_usage = approximate_memory_usage;
            }
        }
        void check_edges() {
            if (edge_map.size() > max_edges) {
                max_edges = edge_map.size();
            }
            if (approximate_memory_usage > max_memory_usage) {
                max_memory_usage = approximate_memory_usage;
            }
        }

    public:
        // for adding new nodes and edges to the maps, returns pointer to node or edge
        const TDD_Node *add_node(TDD_Node node) {
            if (node.is_terminal()) {
                return get_terminal_node();
            }
            check_nodes();
            auto pr = node_map.emplace(node, 1);
            size_t dim = node.get_dimension();
            auto it = pr.first;
            if (!pr.second) {
                // if it already exists, increment refcount unless it is terminal;
                if (!node.is_terminal()) {
                    it->second++;
                }
            }
            else {
                // otherwise its a new node, increase estimated memory
                // one uint16_t for axis_index and then a vector of pointers
                // vector of pointers has 3 pointers for vector, + dim pointers for contents
                approximate_memory_usage += 16;
                approximate_memory_usage += 3 * 64;
                approximate_memory_usage += dim * 64;
            }

            return &(it->first);
        }

        const TDD_Edge *add_edge(TDD_Edge edge) {
            check_edges();
            add_node(*(edge.get_target()));
            auto t1 = std::chrono::high_resolution_clock::now();
            auto pr = edge_map.emplace(edge, 1);
            auto it = pr.first;
            if (!pr.second) {
                // if it already exists, increment refcount;
                it->second++;
            }
            else {
                // if its a new edge, then increase memory usage
                // one 64 bit pointer and one complex double
                approximate_memory_usage += 64;
                approximate_memory_usage += 128;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            inc_time(ms_double.count());

            return &(it->first);
        }
        
        // for removing references
        void remove_node_ref(const TDD_Node *node, bool del = true) {
            check_nodes();
            TDD_Node temp = *node;
            size_t dim = temp.get_dimension();
            // if node is terminal, do not change refcount
            if (temp.is_terminal()) {
                return;
            }
            auto it = node_map.find(temp);
            if (it != node_map.end()) {
                it->second -= 1;
                if (it->second == 0 && del) {
                    // then remove the node
                    node_map.erase(it);
                    // also delete the nodes estimated memory usage
                    // one uint16_t for axis_index and then a vector of pointers
                    // vector of pointers has 3 pointers for vector, + dim pointers for contents
                    approximate_memory_usage -= 16;
                    approximate_memory_usage -= 3 * 64;
                    approximate_memory_usage -= dim * 64;
                }
            }
        }

        void remove_edge_ref(const TDD_Edge *edge, bool remove_node = true) {
            check_edges();
            TDD_Edge temp = *edge;
            if (remove_node) {
                remove_node_ref(temp.get_target());
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            auto it = edge_map.find(temp);
            if (it != edge_map.end()) {
                it->second -= 1;
                if (it->second == 0) {
                    // then remove the edge
                    edge_map.erase(it);
                    // delete from estimated memory usage as well
                    // one 64 bit pointer and one complex double
                    approximate_memory_usage -= 64;
                    approximate_memory_usage -= 128;
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            inc_time(ms_double.count());
        }
        
        // for this I need to make sure the weight is directly propagated upwards in recursive def
        const TDD_Node *get_terminal_node() {
            auto pr = node_map.emplace(terminal_node, 0);
            auto it = pr.first;
            return &(it->first);
        }

        size_t num_unique_nodes() {
            return node_map.size();
        }

        size_t num_unique_edges() {
            return edge_map.size();
        }

        uint64_t peak_nodes() {
            return max_nodes;
        }

        uint64_t peak_edges() {
            return max_edges;
        }

        void inc_time(double t) {
            time += t;
        }

        void print_time() {
            std::cout << "Time taken for specific task: " << time << "ms" << std::endl;
        }

        void reset() {
            node_map.clear();
            edge_map.clear();
        }
        
        void print_maps() {
            std::cout << "node_map" << std::endl;
            for (const auto &[k, v] : node_map) {
                if (k.is_terminal()) {
                    std::cout << "terminal" << std::endl;
                }
                std::cout << "refs: " << v << std::endl;
            }
            std::cout << "edge_map" << std::endl;
            for (const auto &[k, v] : edge_map) {
                if (k.get_target()->is_terminal()) {
                    std::cout << "terminal" << std::endl;
                }
                std::cout << "refs: " << v << std::endl;
            }
        }
        // would be nice to be able to estimate memory cost here, idk how though
        // since nodes can have different numbers of successors, just measuring number of nodes/edges isnt a fair assessment
        uint64_t get_max_memory_usage() {
            return max_memory_usage;
        }
        
};

extern TDD_Map cache_map;

inline const TDD_Edge *TDD_Node::add_successor(const TDD_Node *t, cd w) {
    TDD_Edge new_edge(t, w);
    const TDD_Edge *new_edge_ptr = cache_map.add_edge(new_edge);
    successors.push_back(new_edge_ptr);
    return new_edge_ptr;
}

inline void TDD_Node::clear_successors() const {
    for (size_t i = 0; i < successors.size(); i++) {
        cache_map.remove_edge_ref(successors[i]);
    }
}

inline void TDD_Node::delete_edges() const {
    for (size_t i = 0; i < successors.size(); i++) {
        cache_map.remove_edge_ref(successors[i], false);
    }
}

inline void TDD_Node::cleanup() const {
    if (is_terminal()) {
        return;
    }
    for (size_t i = 0; i < successors.size(); i++) {
        const TDD_Node *next = get_successor_ref(i)->get_target();
        next->cleanup();
        cache_map.remove_edge_ref(successors[i]);
    }
}

inline void TDD_Node::cleanup_duplicates() const {
    if (is_terminal()) {
        return;
    }
    // delete all the successors except the first one which we preserve
    cache_map.remove_edge_ref(successors[0]);
    for (size_t i = 1; i < successors.size(); i++) {
        const TDD_Node *next = get_successor_ref(i)->get_target();
        next->cleanup();
        cache_map.remove_edge_ref(successors[i]);
    }
}

inline void TDD_Node::duplicate() const {
    for (size_t i = 0; i < successors.size(); i++) {
        const TDD_Node *next = get_successor_ref(i)->get_target();
        cache_map.add_edge(*get_successor_ref(i));
        next->duplicate();
    }
}

class TDD {
    private: 
        const TDD_Node *root;
        cd in_weight;
        uint16_t axis_offset = 0;
        // shape is needed for contraction (keeps track of shape TDD should represent)
        // TODO if it turns out there are very high rank tensors (many many dimensions)
        // then may be worth converting from a vector to a deque or list
        std::vector<size_t> shape;
    public:
        TDD() {}
        TDD(const TDD_Node *r, cd weight, std::vector<size_t> s = {}) {
            root = r;
            in_weight = weight;
            shape = s;
        }
        TDD(const TDD_Node *r, cd weight, std::vector<size_t> s, uint16_t offset) {
            root = r;
            in_weight = weight;
            shape = s;
            axis_offset = offset;
        }

        uint16_t get_axis_offset() const {
            return axis_offset;
        }

        const TDD_Node *get_root() const {
            return root;
        }

        cd get_weight() const {
            return in_weight;
        }

        std::vector<size_t> get_shape() const {
            return shape;
        }
        
        void multiply_weight(cd multiplier) {
            in_weight *= multiplier;
        }

        void swap_successors(size_t index1, size_t index2) {
            if (root->get_axis_index() > 0) {
                // if successors are all the same, swapping does nothing
                return;
            }
            TDD_Node new_node(0);
            size_t dimension = root->get_dimension();
            cd normalisation_weight = 0;
            for (size_t i = 0; i < dimension; i++) {
                size_t current_index = i;
                // apply the swap
                if (i == index1) {
                    current_index = index2;
                }
                else if (i == index2) {
                    current_index = index1;
                }
                const TDD_Edge *old_edge = root->get_successor_ref(current_index);
                cd weight = old_edge->get_weight();
                if (normalisation_weight != cd(0,0)) {
                    weight = weight / normalisation_weight;
                }
                else {
                    normalisation_weight = weight;
                    weight = 1;
                }
                new_node.add_successor(old_edge->get_target(), weight);
                cache_map.remove_edge_ref(old_edge);
            }
            // this will never affect reduction, may only affect normalisation
            in_weight *= normalisation_weight;
            cache_map.remove_node_ref(root);
            root = cache_map.add_node(new_node);
        }

        // returns first nonzero index, or if none is found then out of bounds index
        size_t get_first_nonzero_index() const {
            for (size_t i = 0; i < shape.size(); i++) {
                if (shape[i] != 0) {
                    return i;
                }
            }
            return shape.size();
        }

        cd get_value(xarray<size_t> indices) const {
            if (root->is_terminal()) {
                return in_weight;
            }
            if (in_weight == cd{0,0}) {
                return 0;
            }
            uint16_t axis_index = root->get_axis_index();
            const TDD_Node *next = root->get_successor_ref(indices[axis_index])->get_target();
            return in_weight * root->get_weight(indices[axis_index]) * next->get_value(indices);
        }

        double get_probability_sum(uint16_t qubit, uint32_t val) const {
            double w = std::real(std::conj(in_weight) * in_weight);
            if (root->is_terminal()) {
                return 1;
            }
            return w * root->get_probability_sum(qubit, val, get_shape());
        }

        bool operator==(const TDD &other) const {
            return (is_approx_equal(in_weight, other.get_weight()) && root == other.get_root());
        }

        void cleanup() const {
            get_root()->cleanup();
            cache_map.remove_node_ref(get_root());
        }

        // only used for getting child with a physical index
        TDD get_child_TDD(size_t index) {
            // we only want to move one forward (as we are just indexing through the physical index,
            // and want to leave the rest intact)
            std::vector<size_t> new_shape = std::vector<size_t>(shape.begin() + 1, shape.end());
            if (root->is_terminal() || root->get_axis_index() > 0) {
                // if root is terminal or already past the physical index, just return
                // progressing forward by one index in the shape
                // with an axis offset to account for shape shift
                return TDD(root, in_weight, new_shape, 1);
            }
            // otherwise root is not terminal and axis index is 0, thus we need to shift to the successor
            const TDD_Edge* successor = root->get_successor_ref(index);
            const TDD_Node* new_node = successor->get_target();
            cd edge_weight = successor->get_weight();
            cd new_in_weight = in_weight * edge_weight;
            // include an axis offset to account for shape shift
            return TDD(new_node, new_in_weight, new_shape, 1);
        }

};

std::vector<size_t> get_shape_as_vector(xarray<cd> &tensor) {
    std::vector<size_t> shape;
    svector<size_t> true_shape = tensor.shape();
    for (size_t i = 0; i < true_shape.size(); i++) {
        shape.push_back(true_shape[i]);
    }

    return shape;
}

// TODO does clear_successors need to be reimplemented or replaced with cleanup
// as it currently only removes the direct children (does not clean up anything further down)


// recursive implementation to convert arbitrary tensor into a TDD
TDD convert_tensor_to_TDD(xarray<cd> &tensor, uint16_t axis = 0) {

    std::vector<size_t> new_shape = get_shape_as_vector(tensor);

    if (tensor.size() == 1) {
        // if terminal node, then just return trivial TDD with in_weight equal to weight
        // Shape can be left empty at this stage?
        return TDD(cache_map.get_terminal_node(), tensor[0], new_shape);
    }

    size_t dimension = tensor.shape()[0];
    TDD_Node new_node(axis);
    
    xstrided_slice_vector sv;
    for (size_t i = 0; i < tensor.shape().size(); i++) {
        sv.push_back(all());
    }

    // fill up successors of current node, applying normalisation in the process
    cd weight = 1;
    cd normalisation_weight = 0;
    std::set<const TDD_Edge *> new_edge_set;
    for (size_t i = 0; i < dimension; i++) {
        sv[0] = i;
        xarray<cd> new_tensor = strided_view(tensor, sv);

        TDD child = convert_tensor_to_TDD(new_tensor, axis + 1);

        const TDD_Node *next_node = child.get_root();
        cache_map.remove_node_ref(next_node, false);

        cd next_weight = child.get_weight();

        // apply normalisation while iterating through successors
        if (normalisation_weight != cd(0,0)) {
            next_weight = next_weight / normalisation_weight;
        }
        else if (!is_approx_equal(next_weight, cd(0,0))) {
            normalisation_weight = next_weight;
            next_weight = 1;
        }
        // use of the unordered_map to store new nodes and edges implements RR4
        // as each unique node is only stored once
        // also stores new edge in a map to check if edges are all the same
        new_edge_set.insert(new_node.add_successor(next_node, next_weight));
    }
    weight *= normalisation_weight;

    // apply reductions
    // RR2 - redirect weight 0 edge to terminal 
    // if weight is 0, then node is just the terminal node with in_weight 0
    if (weight == cd(0,0)) {
        // in this case, we also need to delete subnodes as we are replacing current node with terminal
        // removing edge reference should also eliminate references to the node
        new_node.cleanup();
        return TDD(cache_map.get_terminal_node(), 0, new_shape);
    }

    // RR3 - need to check new node to see if all successors are the same and have the same weights
    if (new_edge_set.size() == 1) {
        // if all the successors are the same, then that means we do not need this node, instead
        // direct the tdd to the successor node with the in weight
        // that means the other successors need to be deleted as well (they are identical to the first anyway)
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.cleanup_duplicates();

        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, weight, new_shape);
    }

    // otherwise, the new node is now reduced and we can add it to the map
    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);

    return TDD(new_node_ptr, weight, new_shape);
}

// TODO can change set checking to ensure all the same to stop adding them once there are
// two distinct elements in the set (at that point there is a guarantee they wont all be the same,
// so can save on memory by no longer adding to the set)
// assumes all TDDs have the same index set and axis (i.e. same shape)
// Addition is a shape preserving operation
// worst case time complexity is O(tdds.size() * product of shape elements)
// TODO investigate why add_tdds is so slow - may just be due to the overheads?
TDD add_tdds(std::vector<TDD> &tdds, bool first = true) {
    std::set<const TDD_Node *> root_set;
    cd weight_sum = 0;
    uint16_t min_axis_index = tdds[0].get_root()->get_axis_index();
    size_t dimension = tdds[0].get_root()->get_dimension();
    for (TDD tdd : tdds) {
        const TDD_Node *current_root = tdd.get_root();

        // set to keep track of whether they are all equal
        root_set.insert(current_root);

        // sum in weights in case all the roots are the same
        weight_sum += tdd.get_weight();

        // find minimum axis index to use as next level of resulting TDD
        uint16_t axis_index = current_root->get_axis_index();
        if (axis_index < min_axis_index) {
            min_axis_index = axis_index;
            dimension = current_root->get_dimension();
        }
    }

    // check if they are all pointing to the same root
    if (root_set.size() == 1) {
        // if so, sum of the TDDs simply requires adding the weights
        // cleanup all of the other tdds
        for (size_t i = 1; i < tdds.size(); i++) {
            tdds[i].cleanup();
        }
        // shape should also be unchanged
        return TDD(tdds[0].get_root(), weight_sum, tdds[0].get_shape(), tdds[0].get_axis_offset());
    }

    // otherwise start generating new node, with the minimum axis index
    TDD_Node new_node(min_axis_index);
    cd weight = 1;
    // since addition preserves shape, we can set the shape as being the shape of the first TDD
    // it is only crucial to preserve the shape at the top level, as the value of shape is not actually
    // used for addition, its just necessary to maintain the correct shape for contraction
    std::vector<size_t> shape = tdds[0].get_shape();

    // compute successors, normalising in the process
    cd normalisation_weight = 0;
    std::set<const TDD_Edge *> new_edge_set;
    for (size_t i = 0; i < dimension; i++) {
        std::vector<TDD> sub_tdds;
        // calculate correct sub_tdds for each tdd being added
        // indexed over the current index
        for (size_t j = 0; j < tdds.size(); j++) {
            const TDD_Node *root = tdds[j].get_root();
            // if the axis index is equal to the currently processed one
            // then we can index it directly
            if (root->get_axis_index() == min_axis_index) {
                const TDD_Edge *edge = root->get_successor_ref(i);
                const TDD_Node *node = edge->get_target();
                cd edge_weight = tdds[j].get_weight() * edge->get_weight();
                // shape isnt relevant for children, only important to preserve it at the top level
                sub_tdds.push_back(TDD(node, edge_weight));
            }
            // otherwise we are on a later axis index, and can just return the current TDD
            else {
                sub_tdds.push_back(tdds[j]);
            }
        }

        TDD child = add_tdds(sub_tdds, false);

        const TDD_Node *next_node = child.get_root();
        cache_map.remove_node_ref(next_node, false);

        cd next_weight = child.get_weight();

        // apply normalisation while iterating through successors
        if (normalisation_weight != cd(0,0)) {
            next_weight = next_weight / normalisation_weight;
        }
        else if (!is_approx_equal(next_weight, cd(0,0))) {
            normalisation_weight = next_weight;
            next_weight = 1;
        }
        new_edge_set.insert(new_node.add_successor(next_node, next_weight));
    }

    //we can now clean up the summands used at this step (any of those with axis_index == min_axis_index)
    for (size_t i = 0; i < tdds.size(); i++) {
        const TDD_Node *root = tdds[i].get_root();
        // only clean up if we actually used the index at this stage
        if (root->get_axis_index() == min_axis_index) {
            // only delete the edges at this stage in case the next case is base case
            root->delete_edges();
            cache_map.remove_node_ref(root);
            // this means that some nodes don't get deleted though (i.e. if next case
            // is not base case)
        }
        // clean up top level tdds if its the top level call
        else if (first) {
            cache_map.remove_node_ref(root);
        }
    }

    // apply remaining reductions
    weight *= normalisation_weight;
    // RR2 - see convert to tdd for more info
    if (weight == cd(0,0)) {
        new_node.cleanup();
        return TDD(cache_map.get_terminal_node(), 0, shape, tdds[0].get_axis_offset());
    }

    // RR3 - see convert to tdd for more info
    if (new_edge_set.size() == 1) {
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.cleanup_duplicates();
        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, weight, shape, tdds[0].get_axis_offset());
    }

    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
    return TDD(new_node_ptr, weight, shape, tdds[0].get_axis_offset());
}

// axes are assumed to both be in ascending order, and same length
// each axis at the same index should have the same dimension

// resulting shape order will be indices before first contraction index for first TDD,
// followed by indices before first contraction index for second TDD, and so on
// e.g. T1 (a, b, c) T2 (d, b, e) contracted on d results in T (a, d, c, e)

TDD contract_tdds(TDD &first, TDD &second, std::vector<uint16_t> first_axes, std::vector<uint16_t> second_axes, uint16_t axis = 0, bool clear = true, bool kc = false) {

    std::vector<size_t> f_shape = first.get_shape();
    std::vector<size_t> s_shape = second.get_shape();

    uint16_t first_axis_offset = first.get_axis_offset();
    uint16_t second_axis_offset = second.get_axis_offset();

    uint16_t first_axis = first.get_first_nonzero_index();
    uint16_t second_axis = second.get_first_nonzero_index();
    
    // check if both TDDs are trivial (i.e. root is terminal)
    if (first.get_root()->is_terminal() && second.get_root()->is_terminal()) {
        // compute the new weight post contraction
        double contraction_product = 1;
        for (size_t i = 0; i < first_axes.size(); i++) {
            contraction_product *= f_shape[first_axes[i]];
        }
        cd new_weight = first.get_weight() * second.get_weight() * contraction_product;

        // compute new shape (ordered by first TDDs axes first, followed by second TDDs axes)
        // should be first TDDs axes before first removed axes,
        // second TDDs axes before second_removed_axes[0] and so on
        std::vector<size_t> new_shape;
        size_t remaining = first_axes.size();
        size_t i = first_axis;
        size_t j = second_axis;
        size_t first_axis_index = 0;
        size_t second_axis_index = 0;
        while (i < f_shape.size() || j < s_shape.size()) {
            if (second_axis_index == remaining) {
                // then we have reached the end, just loop through i and j until the end is reached
                if (i < f_shape.size()) {
                    new_shape.push_back(f_shape[i]);
                    i++;
                }
                else {
                    new_shape.push_back(s_shape[j]);
                    j++;
                }
            }
            else if (first_axis_index == second_axis_index) {
                if (i < first_axes[first_axis_index]) {
                    new_shape.push_back(f_shape[i]);
                    i++;
                }
                else {
                    first_axis_index++;
                    i++;
                }
            }
            else {
                if (j < second_axes[second_axis_index]) {
                    new_shape.push_back(s_shape[j]);
                    j++;
                }
                else {
                    second_axis_index++;
                    j++;
                }
            }
        }

        return TDD(cache_map.get_terminal_node(), new_weight, new_shape);
    }

    const TDD_Node *f_root = first.get_root();
    const TDD_Node *s_root = second.get_root();

    // start generating new node, with the next axis index
    TDD_Node new_node(axis);
    cd weight = 1;

    // Track new axis index in axis parameter

    size_t dimension;

    // temporary indexing scheme to avoid verbose if statements
    size_t indexing_scheme;
    // indexing_scheme values = 
    // 0 - Index Conditionally and Contract
    // 1 - Index First
    // 2 - Index Second

    // evaluate possible axis combinations to retrieve dimension
    if (first_axes.empty()) {
        // if we are done with contraction, we just need to prioritise left before right
        if (!f_root->is_terminal()) {
            dimension = f_shape[first_axis];
            indexing_scheme = 1;
        }
        else {
            dimension = s_shape[second_axis];
            indexing_scheme = 2;
        }
    }
    else if (first_axis == first_axes[0] && second_axis == second_axes[0]) {
        // contract in this case
        dimension = f_shape[first_axis];
        indexing_scheme = 0;
    }
    else if (first_axis < first_axes[0]) {
        // prioritise incrementing first axis always
        dimension = f_shape[first_axis];
        indexing_scheme = 1;
    }
    else {
        //otherwise increment second axis
        dimension = s_shape[second_axis];
        indexing_scheme = 2;
    }

    // shape to construct
    // (Dimension, Child_Shape)
    // Otherwise addition handles the shape
    std::vector<size_t> shape;
    shape.push_back(dimension);

    // compute successors, normalising in the process
    cd normalisation_weight = 0;
    std::set<const TDD_Edge *> new_edge_set;
    std::vector<TDD> new_tdds;
    for (size_t i = 0; i < dimension; i++) {

        const TDD_Node *first_succ_node = f_root;
        const TDD_Node *second_succ_node = s_root;
        cd first_succ_weight = first.get_weight();
        cd second_succ_weight = second.get_weight();
        std::vector<size_t> first_succ_shape = f_shape;
        std::vector<size_t> second_succ_shape = s_shape;

        std::vector<uint16_t> new_first_axes = first_axes;
        std::vector<uint16_t> new_second_axes = second_axes;
        uint16_t new_axis = axis;

        // need to select new successors depending on indexing_scheme
        // also determines which node pointers to progress
        switch(indexing_scheme) {
            case 0:
                // Index Both and Contract
                // should only occur if both axes are equal
                if (first_axis >= (f_root->get_axis_index() - first_axis_offset) && !f_root->is_terminal()) {
                    first_succ_node = f_root->get_successor_ref(i)->get_target();
                    first_succ_weight *= f_root->get_successor_ref(i)->get_weight();
                    // we progress by one index, but leave axis the same to account for 
                    // lost contraction axis
                }
                if (second_axis >= (s_root->get_axis_index() - second_axis_offset) && !s_root->is_terminal()) {
                    second_succ_node = s_root->get_successor_ref(i)->get_target();
                    second_succ_weight *= s_root->get_successor_ref(i)->get_weight();
                    // we progress by one index, but leave axis the same to account for 
                    // lost contraction axis
                }

                // correct shapes through logical removal
                // basically eliminates all parts of the shape that have now been accounted for
                first_succ_shape[first_axis] = 0;
                second_succ_shape[second_axis] = 0;

                // drop first value as we are contracting that axis at this step
                new_first_axes = std::vector<uint16_t>(first_axes.begin() + 1, first_axes.end());
                new_second_axes = std::vector<uint16_t>(second_axes.begin() + 1, second_axes.end());
                break;
            case 1:
                // Index First
                if (first_axis >= (f_root->get_axis_index() - first_axis_offset) && !f_root->is_terminal()) {
                    first_succ_node = f_root->get_successor_ref(i)->get_target();
                    first_succ_weight *= f_root->get_successor_ref(i)->get_weight();
                }
                // increment by 1 as we progress forward one index
                new_axis += 1;
                // correct shapes through logical removal
                first_succ_shape[first_axis] = 0;
                break;
            case 2:
                // Index Second
                if (second_axis >= (s_root->get_axis_index() - second_axis_offset) && !s_root->is_terminal()) {
                    second_succ_node = s_root->get_successor_ref(i)->get_target();
                    if (kc) {
                        second_succ_weight *= std::conj(s_root->get_successor_ref(i)->get_weight());
                    }
                    else {
                        second_succ_weight *= s_root->get_successor_ref(i)->get_weight();
                    }
                }
                /// increment by 1 as we progress forward one index
                new_axis += 1;
                // correct shapes through logical removal
                second_succ_shape[second_axis] = 0;
                break;
        }

        TDD first_successor = TDD(first_succ_node, first_succ_weight, first_succ_shape, first_axis_offset);
        TDD second_successor = TDD(second_succ_node, second_succ_weight, second_succ_shape, second_axis_offset);

        TDD child = contract_tdds(first_successor, second_successor, new_first_axes, new_second_axes, new_axis, false, kc);

        // if we are contracting (scheme 0) then we only need to store the new tdds and add them
        if (indexing_scheme == 0) {
            new_tdds.push_back(child);
        }
        else {
            const TDD_Node *next_node = child.get_root();
            cache_map.remove_node_ref(next_node, false);

            cd next_weight = child.get_weight();

            std::vector<size_t> child_shape = child.get_shape();
            // if its the first child, use it to construct the shape
            if (i == 0) {
                for (size_t a : child_shape) {
                    shape.push_back(a);
                }
            }

            // apply normalisation while iterating through successors
            if (normalisation_weight != cd(0,0)) {
                next_weight = next_weight / normalisation_weight;
            }
            else if (!is_approx_equal(next_weight, cd(0,0))) {
                normalisation_weight = next_weight;
                next_weight = 1;
            }

            new_edge_set.insert(new_node.add_successor(next_node, next_weight));
        }
    }

    // although the below is effective, it also does not clean up as dynamically as we would like
    // TODO this may result in higher maximum node counts during contraction
    // so it is important to look into making cleanup finer grained (i.e. clean up during execution)
    if (clear) {
        // then we are in initial call, clear the input TDDs
        first.cleanup();
        second.cleanup();
    }


    if (indexing_scheme == 0) {
        // if we are contracting an index at this step, then return the sum of all the contracted TDDs
        // should already be reduced thanks to add_tdds automatically reducing
        // this will automatically clean up the summands too
        return add_tdds(new_tdds);
    }

    // reduce the TDD
    weight *= normalisation_weight;
    // RR2 - see convert to tdd for more info
    if (weight == cd(0,0)) {
        new_node.cleanup();
        return TDD(cache_map.get_terminal_node(), 0, shape);
    }

    // RR3 - see convert to tdd for more info
    if (new_edge_set.size() == 1) {
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.cleanup_duplicates();
        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, weight, shape);
    }

    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
    return TDD(new_node_ptr, weight, shape);
}

TDD apply_lambda_left(TDD &tdd, xarray<cd> &lambda_local) {
    cd new_in_weight = tdd.get_weight();
    uint32_t first_dim = 2;
    if (tdd.get_root()->get_axis_index() != 0) {
        // then the 0 and 1 children are the same, this doesn't get changed by this operation
        first_dim = 1;
    }
    TDD_Node new_root(0);
    cd global_normalisation_weight = 0;
    for (uint32_t i = 0; i < first_dim; i++) {
        // need to get 0 and 1 children and treat separately
        // unless first_dim is 1, i.e. 0-child == 1-child
        const TDD_Node* i_state;
        cd old_weight;
        if (first_dim == 1) {
            i_state = tdd.get_root();
            old_weight = 1;
        }
        else {
            i_state = tdd.get_root()->get_successor_ref(i)->get_target();
            old_weight = tdd.get_root()->get_successor_ref(i)->get_weight();
        }
        if (old_weight == cd(0,0)) {
            new_root.add_successor(cache_map.get_terminal_node(), cd(0,0));
            continue;
        }
        TDD_Node new_i_child(1);
        // need to do this for all items in state.get_shape()[1] as we are absorbing from left
        // no need to change the first edge though
        TDD_Node new_node(1);
        cd normalisation_weight = 0;
        if (i_state->get_axis_index() != 1) {
            new_node.set_successors(i_state->get_successors());
            normalisation_weight = lambda_local(0);
        }
        std::set<const TDD_Edge *> new_edge_set;
        for (uint32_t j = 0; j < tdd.get_shape()[1]; j++) {
            if (i_state->get_axis_index() == 1) {
                const TDD_Edge* old_edge = i_state->get_successor_ref(j);
                // we multiply on this axis for left contraction
                cd new_edge_weight = old_edge->get_weight() * lambda_local(j);
                if (normalisation_weight != cd(0,0)) {
                    new_edge_weight = new_edge_weight / normalisation_weight;
                }
                else if (!is_approx_equal(new_edge_weight, cd(0,0))) {
                    normalisation_weight = lambda_local(j);
                    new_edge_weight = 1;
                }
                new_edge_set.insert(new_i_child.add_successor(old_edge->get_target(), new_edge_weight));
                cache_map.remove_edge_ref(old_edge);
            }
            else {
                // otherwise we skipped this axis, so need to make new node with weight from lambda
                cd new_edge_weight = lambda_local(j) / normalisation_weight;
                const TDD_Node* new_node_ptr = cache_map.add_node(new_node);
                cache_map.remove_node_ref(new_node_ptr, false);
                new_edge_set.insert(new_i_child.add_successor(new_node_ptr, new_edge_weight));
            }
        }
        if (global_normalisation_weight != cd(0,0)) {
            normalisation_weight = normalisation_weight / global_normalisation_weight;
        }
        else {
            global_normalisation_weight = normalisation_weight;
            normalisation_weight = 1;
        }
        old_weight *= normalisation_weight;
        if (i_state->get_axis_index() != 1) {
            i_state->cleanup();
        }
        // delete the old i_state
        cache_map.remove_node_ref(i_state);
        if (new_edge_set.size() == 1) {
            // if all the successors are the same, then that means we do not need this node, instead
            // direct the tdd to the successor node with the in weight
            // that means the other successors need to be deleted as well (they are identical to the first anyway)
            const TDD_Node next_node = *(new_i_child.get_successor_ref(0)->get_target());
            // delete the successors
            new_i_child.cleanup_duplicates();

            const TDD_Node *new_node_ptr = cache_map.add_node(next_node);
            // then new_node_ptr is the new_node to pass up to the parent
            if (first_dim == 1) {
                cache_map.remove_node_ref(tdd.get_root());
                return TDD(new_node_ptr, new_in_weight * global_normalisation_weight, tdd.get_shape());
            }
            cache_map.remove_node_ref(new_node_ptr, false);
            new_root.add_successor(new_node_ptr, old_weight);
        }
        else {
            const TDD_Node *new_node_ptr = cache_map.add_node(new_i_child);
            if (first_dim == 1) {
                cache_map.remove_node_ref(tdd.get_root());
                return TDD(new_node_ptr, new_in_weight * global_normalisation_weight, tdd.get_shape());
            }
            cache_map.remove_node_ref(new_node_ptr, false);
            new_root.add_successor(new_node_ptr, old_weight);
        }
    }
    // clean up old tdd
    tdd.get_root()->delete_edges();
    cache_map.remove_node_ref(tdd.get_root());
    // final reduction
    if (new_root.get_successor_ref(0) == new_root.get_successor_ref(1)) {
        const TDD_Node next_node = *(new_root.get_successor_ref(0)->get_target());
        // delete the successors
        new_root.cleanup_duplicates();

        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);
        return TDD(new_node_ptr, new_in_weight * global_normalisation_weight, tdd.get_shape());
    }
    const TDD_Node* new_node_ptr = cache_map.add_node(new_root);
    return TDD(new_node_ptr, new_in_weight * global_normalisation_weight, tdd.get_shape());
}

// this is hardcoded specifically for the case of applying lambda to the right side of an MPS TDD
// we can do this by copying all the nodes down to the bottom level, then multiplying by lambdas
// and propagating the normalisation all the way back up
// this means we can cleanup at the end as we fully replace the tdd
// This still offers time saving over regular contraction for lambda application,
// as we can keep lambda as just the diagonal values, eliminating conversion to TDD
// and additionally we do not need to make any calls to add_tdds
// This case is still slower than left lambda application though however
// note we can assume the state is of shape (2,x1,x2), as the case where qubit is 0 is equivalent
// to left lambda contraction, and we do not absorb from the right on the rightmost qubit
// NOTE: this can also be extended to any last axis application
// could possibly be extended to applying lambdas to arbitrary layers as well
TDD apply_lambda_right(TDD &tdd, xarray<cd> &lambda_local, uint16_t axis = 0, bool cleanup = true) {
    const TDD_Node* root = tdd.get_root();
    uint16_t axis_index = root->get_axis_index();
    std::vector<size_t> shape = tdd.get_shape();
    cd in_weight = tdd.get_weight();
    // if weight is 0, then node is just the terminal node with in_weight 0
    if (in_weight == cd(0,0)) {
        // in this case, the root is terminal with weight 0
        // thus applying lambda would not do anything anyway
        return TDD(cache_map.get_terminal_node(), 0, shape);
    }
    size_t dimension = shape[axis];
    TDD_Node new_node(axis);
    cd normalisation_weight = 0;
    if (axis == 2 && axis_index != 2) {
        normalisation_weight = lambda_local(0);
    }
    std::set<const TDD_Edge*> new_edge_set;
    for (size_t i = 0; i < dimension; i++) {
        if (axis == 2 && axis_index == 2) {
            // then we are at the level we need to apply lambdas and we have existing edges
            // to multiply up by lambda
            const TDD_Edge* old_edge = root->get_successor_ref(i);
            cd new_edge_weight = old_edge->get_weight() * lambda_local(i);
            if (normalisation_weight != cd(0,0)) {
                new_edge_weight = new_edge_weight / normalisation_weight;
            }
            else if (!is_approx_equal(new_edge_weight, cd(0,0))) {
                normalisation_weight = lambda_local(i);
                new_edge_weight = 1;
            }
            // this is always an edge to the terminal node
            // TODO can remove the old edge here as part of finer grained removal
            // cache_map.remove_edge_ref(old_edge);
            new_edge_set.insert(new_node.add_successor(cache_map.get_terminal_node(), new_edge_weight));
        }
        else if (axis == 2) {
            // otherwise axis is 2, but the TDD skips axis 2 (i.e. all the successors were the same)
            // so we need to construct new edges to terminal nodes
            cd new_edge_weight = lambda_local(i) / normalisation_weight;
            new_edge_set.insert(new_node.add_successor(cache_map.get_terminal_node(), new_edge_weight));
        }
        else {
            // otherwise we are on axis 0 or 1, so progress tdd as necessary and make recursive call
            const TDD_Node* child = root;
            cd child_weight = 1;
            if (axis_index == axis) {
                // progress tdd
                const TDD_Edge* temp = root->get_successor_ref(i);
                child = temp->get_target();
                child_weight = temp->get_weight();
            }
            TDD old_child(child, child_weight, shape);
            TDD new_child = apply_lambda_right(old_child, lambda_local, axis + 1, false);
            const TDD_Node *new_child_root = new_child.get_root();
            cd new_weight = new_child.get_weight();
            // update normalisation weight if necessary
            if (normalisation_weight != cd(0,0)) {
                new_weight = new_weight / normalisation_weight;
            }
            else if (!is_approx_equal(new_weight, cd(0,0))) {
                normalisation_weight = new_weight;
                new_weight = 1;
            }
            cache_map.remove_node_ref(new_child_root, false);
            new_edge_set.insert(new_node.add_successor(new_child_root, new_weight));
        }    
    }
    if (cleanup) {
        // delete the old tdd towards end of top level call
        // TODO make cleanup more efficient? - as you go along
        // that would save having to do a second traversal, does not affect time complexity
        tdd.cleanup();
    }
    // now do usual reduction and normalisation

    // RR3 - need to check new node to see if all successors are the same and have the same weights
    if (new_edge_set.size() == 1) {
        // if all the successors are the same, then that means we do not need this node, instead
        // direct the tdd to the successor node with the in weight
        // that means the other successors need to be deleted as well (they are identical to the first anyway)
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.cleanup_duplicates();

        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, in_weight * normalisation_weight, shape);
    }

    // otherwise, the new node is now reduced and we can add it to the map
    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
    return TDD(new_node_ptr, in_weight * normalisation_weight, shape);
}

// convert TDD to a tensor
xarray<cd> convert_TDD_to_tensor(TDD tdd, bool cleanup = true) {
    uint16_t axis_offset = tdd.get_axis_offset();
    xarray<cd> tensor = zeros<cd>(tdd.get_shape());
    const TDD_Node* root = tdd.get_root();
    if (root->is_terminal()) {
        tensor = full_like(tensor, tdd.get_weight());
        tdd.cleanup();
        return tensor;
    }

    size_t target_dimension = tdd.get_shape().size();
    // process nodes iteratively
    std::queue<const TDD_Node*> nodes_to_process;
    std::queue<cd> cumulative_weights;
    std::queue<xstrided_slice_vector> index_sets;
    nodes_to_process.push(root);
    cumulative_weights.push(tdd.get_weight());
    xstrided_slice_vector initial_index_set;
    // account for initial skipped layers
    for (uint16_t i = 0; i < (root->get_axis_index() - axis_offset); i++) {
        initial_index_set.push_back(all());
    }
    index_sets.push(initial_index_set);
    while (!nodes_to_process.empty()) {
        const TDD_Node* current_node = nodes_to_process.front();
        nodes_to_process.pop();
        cd cumulative_weight = cumulative_weights.front();
        cumulative_weights.pop();
        xstrided_slice_vector index_set = index_sets.front();
        index_sets.pop();

        size_t dimension = current_node->get_dimension();
        for (size_t i = 0; i < dimension; i++) {
            const TDD_Edge* new_edge = current_node->get_successor_ref(i);
            const TDD_Node* new_node = new_edge->get_target();
            // update cumulative weights and index set
            cd new_weight = new_edge->get_weight() * cumulative_weight;
            xstrided_slice_vector new_index_set = index_set;
            new_index_set.push_back(i);

            if (new_node->is_terminal()) {
                // if new node is terminal, then we want to fully populate all items it refers to
                // we don't need to push anything then, and can now just fully populate
                while (new_index_set.size() < target_dimension) {
                    new_index_set.push_back(all());
                }
                auto tr = strided_view(tensor, new_index_set);
                tr = new_weight;
            }
            else {
                // need to also push back skipped layers
                uint32_t axis_diff = new_node->get_axis_index() - current_node->get_axis_index();
                for (uint32_t i = 1; i < axis_diff; i++) {
                    new_index_set.push_back(all());
                }
                // otherwise keep traversing
                nodes_to_process.push(new_node);
                cumulative_weights.push(new_weight);
                index_sets.push(new_index_set);
            }
        }
    }

    // clean up the tdd, could this be more efficient by deleting as we go along above?
    if (cleanup) {
        tdd.cleanup();
    }
    return tensor;
}

// uses contract_tdds to compute kronecker product of tdd and tdd_conjugate
TDD kronecker_conjugate(TDD tdd) {
    TDD tdd2 = TDD(tdd.get_root(), std::conj(tdd.get_weight()), tdd.get_shape(), tdd.get_axis_offset());
    return contract_tdds(tdd, tdd2, {}, {}, 0, false, true);
}

// TODO for these multiply functions, look into adding 0 shortcircuiting?

// contract two vectors together efficiently (should be same length vectors)
// axis input here is whatever the axis index would be in the main computation
TDD multiply_vectors(TDD &v1, TDD &v2, uint16_t axis1 = 0, uint16_t axis2 = 0) {
    cd sum = 0;
    size_t dim = v1.get_shape()[0];
    for (size_t i = 0; i < dim; i++) {
        cd w1 = v1.get_weight();
        cd w2 = v2.get_weight();
        if (v1.get_root()->get_axis_index() == axis1) {
            // then its not already terminal, so can index it
            w1 *= v1.get_root()->get_successor_ref(i)->get_weight();
        }
        if (v2.get_root()->get_axis_index() == axis2) {
            // then its not already terminal, so can index it
            w2 *= v2.get_root()->get_successor_ref(i)->get_weight();
        }
        sum += w1 * w2;
    }
    return TDD(cache_map.get_terminal_node(), sum, {});
}

// contract vector with matrix
TDD multiply_v_m(TDD &v, TDD &m, uint16_t axis1 = 0, uint16_t axis2 = 0) {
    std::vector<TDD> tdds_to_add;
    size_t dim = v.get_shape()[0];
    for (size_t i = 0; i < dim; i++) {
        cd w1 = v.get_weight();
        if (v.get_root()->get_axis_index() == axis1) {
            // then its not already terminal, so can index it
            w1 *= v.get_root()->get_successor_ref(i)->get_weight();
        }
        const TDD_Node* m2 = m.get_root();
        cd w2 = m.get_weight();
        if (m2->get_axis_index() == axis2) {
            // then can index it
            const TDD_Edge* edge = m2->get_successor_ref(i);
            m2 = edge->get_target();
            w2 *= edge->get_weight();
        }
        // TODO find faster way than simply duplicating m2 and overwriting the axis_index
        size_t dim2 = m.get_shape()[1];
        // if we call from multiply_matrices then we want axis_index = 2
        // if not, then axis_index = 1
        // i.e. we want the new axis_index to be = axis1
        if (m2->is_terminal()) {
            tdds_to_add.push_back(TDD(m2, w1 * w2, {dim2}));
        }
        else {
            TDD_Node m2_prime(axis1);
            m2_prime.set_successors(m2->get_successors());
            const TDD_Node* m2pp = cache_map.add_node(m2_prime);
            for (size_t j = 0; j < dim2; j++) {
                cache_map.add_edge(*(m2_prime.get_successor_ref(j)));
            }
            tdds_to_add.push_back(TDD(m2pp, w1 * w2, {dim2}));
        }
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    TDD temp = add_tdds(tdds_to_add, false);
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    // cache_map.inc_time(ms_double.count());
    return temp;
    // return add_tdds(tdds_to_add, false);
}

// for this we just require m1 is a matrix (m2 can be matrix or vector?)
// for contracting matrix-vector or matrix-matrix
TDD multiply_matrices(TDD &m1, TDD &m2, uint16_t axis = 0) {
    TDD_Node new_node(axis);
    cd in_weight = m1.get_weight();
    size_t dim = m1.get_shape()[0];
    cd normalisation_weight = 0;
    std::set<const TDD_Edge*> new_edge_set;
    std::vector<size_t> f_shape = m1.get_shape();
    // construct the shape
    std::vector<size_t> shape{dim};
    for (size_t i = 0; i < dim; i++) {
        const TDD_Node *m1_child = m1.get_root();
        cd m1_child_weight = 1;
        if (m1_child->get_axis_index() == axis) {
            // then can index m1_child
            const TDD_Edge *edge = m1_child->get_successor_ref(i);
            m1_child = edge->get_target();
            m1_child_weight *= edge->get_weight();
        }
        std::vector<size_t> m1_child_shape(f_shape.begin() + 1, f_shape.end());
        TDD m1_vector(m1_child, m1_child_weight, m1_child_shape);
        TDD new_child;
        if (m2.get_shape().size() == 2) {
            // m2 is a matrix, then we can call multiply_v_m
            new_child = multiply_v_m(m1_vector, m2, axis + 1, axis);
        }
        else {
            // m2 is a vector, we can call multiply_vectors
            new_child = multiply_vectors(m1_vector, m2, axis + 1, axis);
        }
        // update normalisation weight and add new edge

        const TDD_Node *new_child_root = new_child.get_root();
        cd new_weight = new_child.get_weight();
        // push back the child shape onto the shape
        if (i == 0) {
            std::vector<size_t> child_shape = new_child.get_shape();
            for (uint32_t j = 0; j < child_shape.size(); j++) {
                shape.push_back(child_shape[j]);
            }
        }
        // update normalisation weight if necessary
        if (normalisation_weight != cd(0,0)) {
            new_weight = new_weight / normalisation_weight;
        }
        else if (!is_approx_equal(new_weight, cd(0,0))) {
            normalisation_weight = new_weight;
            new_weight = 1;
        }
        cache_map.remove_node_ref(new_child_root, false);
        new_edge_set.insert(new_node.add_successor(new_child_root, new_weight));
    }
    // reduce and normalise
    in_weight *= normalisation_weight;
    // RR2 - see convert to tdd for more info
    if (in_weight == cd(0,0)) {
        new_node.cleanup();
        return TDD(cache_map.get_terminal_node(), 0, shape);
    }
    // RR3 - need to check new node to see if all successors are the same and have the same weights
    if (new_edge_set.size() == 1) {
        // if all the successors are the same, then that means we do not need this node, instead
        // direct the tdd to the successor node with the in weight
        // that means the other successors need to be deleted as well (they are identical to the first anyway)
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.cleanup_duplicates();

        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, in_weight, shape);
    }

    // otherwise, the new node is now reduced and we can add it to the map
    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
    return TDD(new_node_ptr, in_weight, shape);
}

// specialised algorithm for contracting two adjacent MPS TDDs
// takes mab, nbc -> mnac or mnc (c can still be 1 though)
TDD contract_MPS_tdds(TDD &mps1, TDD &mps2, bool cleanup = true) {
    TDD_Node new_node(0);
    cd in_weight = 1;
    cd normalisation_weight = 0;
    std::set<const TDD_Edge*> new_edge_set;
    std::vector<size_t> shape{4};
    std::vector<size_t> fshape = mps1.get_shape();
    std::vector<size_t> sshape = mps2.get_shape();
    // TODO could make use of equality of different cases to skip recomputing the same matrix products
    for (uint32_t i = 0; i < 2; i++) {
        const TDD_Node *root1 = mps1.get_root();
        cd weight1 = mps1.get_weight();
        std::vector<size_t> shape1(fshape.begin() + 1, fshape.end());
        if (root1->get_axis_index() == 0) {
            // then we can index it 
            const TDD_Edge *edge = root1->get_successor_ref(i);
            root1 = edge->get_target();
            weight1 *= edge->get_weight();
        }
        TDD child1 = TDD(root1, weight1, shape1);
        for (uint32_t j = 0; j < 2; j++) {
            const TDD_Node *root2 = mps2.get_root();
            cd weight2 = mps2.get_weight();
            std::vector<size_t> shape2(sshape.begin() + 1, sshape.end());
            if (root2->get_axis_index() == 0) {
                // then we can index it 
                const TDD_Edge *edge = root2->get_successor_ref(j);
                root2 = edge->get_target();
                weight2 *= edge->get_weight();
            }
            TDD child2 = TDD(root2, weight2, shape2);
            // then we want to add a child to the new_root for each mps1[i] * mps2[j]
            TDD new_child;
            // either vector-matrix, matrix-matrix, or matrix-vector (or vector-vector if only 2 qubits total)
            if (shape1.size() == 1 && shape2.size() == 1) {
                // vector-vector - outputs scalar
                new_child = multiply_vectors(child1, child2, 1, 1);
            }
            else if (shape1.size() == 1 && shape2.size() == 2) {
                // vector-matrix - outputs vector
                new_child = multiply_v_m(child1, child2, 1, 1);
            }
            else {
                // matrix-vector or matrix-matrix
                new_child = multiply_matrices(child1, child2, 1);
            }
            // update normalisation weight and add new edge
            const TDD_Node *new_child_root = new_child.get_root();
            cd new_weight = new_child.get_weight();
            // push back the child shape onto the shape
            if (i == 0 && j == 0) {
                std::vector<size_t> child_shape = new_child.get_shape();
                for (uint32_t j = 0; j < child_shape.size(); j++) {
                    shape.push_back(child_shape[j]);
                }
            }
            // update normalisation weight if necessary
            if (normalisation_weight != cd(0,0)) {
                new_weight = new_weight / normalisation_weight;
            }
            else if (!is_approx_equal(new_weight, cd(0,0))) {
                normalisation_weight = new_weight;
                new_weight = 1;
            }
            cache_map.remove_node_ref(new_child_root, false);
            new_edge_set.insert(new_node.add_successor(new_child_root, new_weight));    
        }
    }
    if (cleanup) {
        mps1.cleanup();
        mps2.cleanup();
    }
    // apply reduction and normalisation
    in_weight *= normalisation_weight;
    // RR2 - see convert to tdd for more info
    if (in_weight == cd(0,0)) {
        new_node.cleanup();
        return TDD(cache_map.get_terminal_node(), 0, shape);
    }
    // RR3 - need to check new node to see if all successors are the same and have the same weights
    if (new_edge_set.size() == 1) {
        // if all the successors are the same, then that means we do not need this node, instead
        // direct the tdd to the successor node with the in weight
        // that means the other successors need to be deleted as well (they are identical to the first anyway)
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.cleanup_duplicates();

        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, in_weight, shape);
    }

    // otherwise, the new node is now reduced and we can add it to the map
    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
    return TDD(new_node_ptr, in_weight, shape);
}

// efficiently apply dim x dim gate to a tdd which is dim x anything
TDD apply_gate(xarray<cd> &gate, TDD &tdd, uint32_t dim) {
    TDD_Node new_node(0);
    cd normalisation_weight = 0;
    cd in_weight = tdd.get_weight();
    std::vector<size_t> shape = tdd.get_shape();
    std::set<const TDD_Edge *> new_edge_set;
    // we are generating 2 or 4 children at this stage
    for (uint32_t i = 0; i < dim; i++) {
        std::vector<TDD> tdds_to_add;
        // we are indexing the tdd state here to sum the matrices/vectors
        for (uint32_t j = 0; j < dim; j++) {
            const TDD_Node *summand = tdd.get_root();
            cd s_weight = 1;
            if (summand->get_axis_index() == 0) {
                // then we can index it 
                const TDD_Edge *edge = summand->get_successor_ref(j);
                summand = edge->get_target();
                s_weight *= edge->get_weight();
            }
            // shape doesnt matter going into add_tdds as it is shape preserving, but do it just in case
            // we change add_tdds in the future
            // duplicate summand in the cache map to ensure we don't remove the original
            summand->duplicate();
            cache_map.add_node(*summand);
            std::vector s_shape(shape.begin() + 1, shape.end());
            tdds_to_add.push_back(TDD(summand, gate(i,j) * s_weight, s_shape));
        }
        // auto t1 = std::chrono::high_resolution_clock::now();
        TDD new_child = add_tdds(tdds_to_add, false);
        // auto t2 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        // cache_map.inc_time(ms_double.count());

        const TDD_Node *new_child_root = new_child.get_root();
        cd new_weight = new_child.get_weight();

        // update normalisation weight if necessary
        if (normalisation_weight != cd(0,0)) {
            new_weight = new_weight / normalisation_weight;
        }
        else if (!is_approx_equal(new_weight, cd(0,0))) {
            normalisation_weight = new_weight;
            new_weight = 1;
        }
        cache_map.remove_node_ref(new_child_root, false);
        new_edge_set.insert(new_node.add_successor(new_child_root, new_weight));

    }
    tdd.cleanup();
    // apply reduction and normalisation
    in_weight *= normalisation_weight;
    // RR2 - see convert to tdd for more info
    if (in_weight == cd(0,0)) {
        new_node.cleanup();
        return TDD(cache_map.get_terminal_node(), 0, shape);
    }
    // RR3 - need to check new node to see if all successors are the same and have the same weights
    if (new_edge_set.size() == 1) {
        // if all the successors are the same, then that means we do not need this node, instead
        // direct the tdd to the successor node with the in weight
        // that means the other successors need to be deleted as well (they are identical to the first anyway)
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.cleanup_duplicates();

        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, in_weight, shape);
    }

    // otherwise, the new node is now reduced and we can add it to the map
    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
    return TDD(new_node_ptr, in_weight, shape);
}

// efficient algorithm for applying a two qubit gate (4x4 matrix)
// takes 4,4 and 4,xi1,xi2 -> 4,xi1,xi2
// TODO add swap functionality here and move gate processing to here from apply_instruction
TDD apply_two_qubit_gate(xarray<cd> &gate, TDD &tdd) {
    // TODO get more efficient way of checking gate equality
    // perhaps gate numbers for example
    // TODO add hardcoded gates here too
    // if ( it is a cnot gate ) {
    //     // then we just have to swap 10 and 11
    //     tdd.swap_successors(2,3);
    //     return tdd;
    // }
    // else if (it is a swap gate) {
    //     // then we swap 10 and 01
    //     tdd.swap_successors(1, 2);
    //     return tdd;
    // }
    return apply_gate(gate, tdd, 4);
}

// efficiently apply single qubit gate
// 2,2 and 2,xi1,xi2 -> 2,xi1,xi2
TDD apply_single_qubit_gate(xarray<cd> &gate, TDD &tdd) {
    // TODO add hardcoded gates here too
    return apply_gate(gate, tdd, 2);
}
// BELOW IS NOT ACTUALLY USED ANYWHERE, however it could be necessary for doing SVD directly on 
// TDD operation
// Swapping axes is a fairly efficient operation, but reshape would be less efficient than
// directly reshaping a tensor thus it seems it is not viable to do the direct SVD on a TDD

// swap adjacent axes i.e. if swapping a, b sends iabj -> ibaj
// we assume second_axis = first_axis + 1
TDD swap_adjacent_axes(TDD tdd, uint16_t first_axis, uint16_t second_axis, bool first = true) {
    // first traverse the TDD to reach the first_axis
    const TDD_Node* root = tdd.get_root();
    std::vector<size_t> shape = tdd.get_shape();
    std::vector<size_t> final_shape = shape;
    if (first) {
        final_shape[first_axis] = shape[second_axis];
        final_shape[second_axis] = shape[first_axis];
    }
    uint16_t axis_index = root->get_axis_index();
    if (axis_index > second_axis || root->is_terminal()) {
        // then we know the values are unaffected by swapping the two axes,
        // just need to swap the shape
        // as we have skipped both axes that needed to be swapped
        return TDD(root, tdd.get_weight(), final_shape);
    }
    else if (axis_index == first_axis) {
        // if the axis_index == first_axis, then we must swap the first two axes
        // this is certainly the hardest case
        cd in_weight = tdd.get_weight();
        TDD_Node new_node(first_axis);
        uint16_t target_dimension = shape[second_axis]; // the top level node will have dimension of the second node
        uint16_t target_child_dimension = shape[first_axis]; // basically swapping the dimensions
        uint16_t current_dimension = shape[first_axis];
        std::set<const TDD_Edge *> new_edge_set;
        // for the purposes of return, may need to alter shape
        if (first) {
            shape = final_shape;
        }
        for (size_t i = 0; i < target_dimension; i++) {
            // need to make a new node for each of these
            TDD_Node new_child(second_axis);

            // need to compute the in_weight for each of these
            // this node corresponds to second_axis index = i
            // so lets have the in_weight be the non zero weight with min first_axis and second_axis index = i ?
            // we choose the weight this way as this allows us to automatically normalise
            cd weight = cd(0,0);
            for (size_t j = 0; j < current_dimension; j++) {
                const TDD_Edge* edge = root->get_successor_ref(j);
                const TDD_Node* succ = edge->get_target();
                if (succ->is_terminal() || succ->get_axis_index() > second_axis) {
                    // then we treat weight as being 1
                    weight = 1;
                }
                else {
                    // otherwise we are exactly at the second axis, so we get the weight directly
                    weight = succ->get_successor_ref(i)->get_weight();
                }
                if (weight != cd(0,0)) {
                    break;
                }
            }
            // if the weight is approximately 0 (i.e. from floating point errors)
            // then just point at the 0 terminal
            if (is_approx_equal(weight, cd(0,0))) {
                new_edge_set.insert(new_node.add_successor(cache_map.get_terminal_node(), cd(0,0)));
                continue;
            }
            // so we now have the new weight going into the new child
            for (size_t j = 0; j < target_child_dimension; j++) {
                // here we are computing the out_weight, and then making an edge to the old successor
                const TDD_Edge* edge = root->get_successor_ref(j);
                const TDD_Node* succ = edge->get_target();
                cd edge_weight = edge->get_weight();
                cd target_weight = edge_weight;
                const TDD_Node* target_node;
                if (succ-> is_terminal() || succ->get_axis_index() > second_axis) {
                    // effectively treat the weight as being 1
                    target_node = succ;
                }
                else {
                    target_weight *= succ->get_successor_ref(i)->get_weight();
                    target_node = succ->get_successor_ref(i)->get_target();
                }
                cd out_weight = cd(0,0);
                if (weight != cd(0,0)) {
                    out_weight = target_weight / weight;
                    // in case floating point errors mean they are not quite equal, 
                    // make sure we set to 1 for normalisation
                    if (is_approx_equal(target_weight, weight)) {
                        out_weight = cd(1,0);
                    }
                }
                new_child.add_successor(target_node, out_weight);
            }
            const TDD_Node* new_child_ptr = cache_map.add_node(new_child);
            new_edge_set.insert(new_node.add_successor(new_child_ptr, weight));
        }
        // I believe the above is normalised by design

        // we can clean up the old nodes and edges here now, as we do not need them any more
        for (size_t i = 0; i < current_dimension; i++) {
            const TDD_Node* successor = root->get_successor_ref(i)->get_target();
            // delete the old edges (without deleting the nodes they point to as those are preserved)
            successor->delete_edges();
        }
        // delete the original root and its successors
        root->clear_successors();
        cache_map.remove_node_ref(root);

        // RR3 - only need to apply reduction rule here
        if (new_edge_set.size() == 1) {
            const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
            // delete the successors
            new_node.clear_successors();
            const TDD_Node *new_node_ptr = cache_map.add_node(next_node);
            return TDD(new_node_ptr, in_weight, shape);
        }

        const TDD_Node* new_node_ptr = cache_map.add_node(new_node);
        return TDD(new_node_ptr, in_weight, shape);
    }
    else if (axis_index == second_axis) {
        // otherwise, first_axis does not do anything, so just need to update axis index of node?
        TDD_Node new_node(first_axis);
        std::vector<const TDD_Edge *> successors = root->get_successors();
        new_node.set_successors(successors);
        // now we just need to remove the old node, and add the new one?
        cache_map.remove_node_ref(root);
        const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
        return TDD(new_node_ptr, tdd.get_weight(), shape);
    }
    // otherwise we have axis_index < first_axis
    // so just progress forwards

    size_t dimension = shape[axis_index];
    TDD_Node new_node(axis_index);
    cd weight = tdd.get_weight();
    // compute successors
    std::set<const TDD_Edge *> new_edge_set;
    for (size_t i = 0; i < dimension; i++) {
        const TDD_Edge* successor_edge = root->get_successor_ref(i);
        const TDD_Node* successor_node = successor_edge->get_target();
        cd successor_weight = successor_edge->get_weight();

        // pass through the shape unchanged as it doesn't matter here
        TDD current_child(successor_node, successor_weight, shape);
        
        TDD child = swap_adjacent_axes(current_child, first_axis, second_axis, false);
        // TODO evaluate whether it is necessary to re-reduce (I believe renormalisation is unnecessary here)
        const TDD_Node *next_node = child.get_root();
        cache_map.remove_node_ref(next_node, false);

        cd next_weight = child.get_weight();

        new_edge_set.insert(new_node.add_successor(next_node, next_weight));
    }

    // RR3 - only need to apply reduction rule here
    if (new_edge_set.size() == 1) {
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.clear_successors();
        const TDD_Node *new_node_ptr = cache_map.add_node(next_node);

        return TDD(new_node_ptr, weight, shape);
    }

    const TDD_Node *new_node_ptr = cache_map.add_node(new_node);
    return TDD(new_node_ptr, weight, shape);
}

// TDD reshape(TDD tdd, std::vector<size_t> new_shape) {

// }

#endif