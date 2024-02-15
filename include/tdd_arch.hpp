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
            return (target == other.get_target() && weight == other.get_weight());
        }
};

class TDD_Node {
    private:
        uint8_t axis_index;
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


        // Reduction rule 5 involves eliminating redundant edges 
        // this approach would reduce the number of pointers stored
        // duplicate edges are already handled by the edge map

    public:

        TDD_Node(){}
        TDD_Node(uint8_t axis) {
            axis_index = axis;
        }

        const TDD_Edge *get_successor_ref(size_t index) const {
            return successors[index];
        }
        
        bool is_terminal() const {
            return successors.empty();
        }

        uint8_t get_axis_index() const {
            return axis_index;
        }

        const TDD_Edge *add_successor(const TDD_Node *t, cd &w);

        std::vector<const TDD_Edge *> get_successors() const {
            return successors;
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

        void cleanup() const;
        
        cd get_value (xarray<size_t> indices) const {
            if (is_terminal()) {
                return cd(1,0);
            }
            if (indices.size() == 0) {
                // this would mean an illegal index set (too short)
                // TODO HANDLE ELEGANTLY INSTEAD OF RETURNING WEIRD VALUE
                return cd(-2,0);
            }
            size_t current_index = indices[0];
            if (get_weight(current_index) == cd{0,0}) {
                return cd(0,0);
            }
            const TDD_Node *next = get_successor_ref(current_index)->get_target();
            xarray<size_t> new_indices;
            // if next is not terminal then node merge is relevant
            if (!next->is_terminal()) {
                // account for node merge by tracking the axis index
                uint8_t index_diff = next->get_axis_index() - get_axis_index();
                new_indices = view(indices, range(index_diff, indices.size()));
            }
            return get_weight(current_index) * next->get_value(new_indices);
        }

        bool operator==(const TDD_Node &other) const {
            return (axis_index == other.get_axis_index() && successors == other.get_successors());
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
            total ^= hash<const TDD_Edge *>()(t.get_successor_ref(i)) << (i % 3);
            total = total >> (i % 3);
        }
        return total ^ (hash<uint8_t>()(t.get_axis_index()) << 1);
    }
};

// map to store all the edges and nodes that are defined throughout the process
class TDD_Map {
    private:
        // maps contain the item itself, and maps to the reference count
        std::unordered_map<TDD_Edge, uint16_t> edge_map;
        std::unordered_map<TDD_Node, uint16_t> node_map;

        // unique terminal node is defined here, -1 is chosen as axis index to ensure it is always last
        TDD_Node terminal_node = TDD_Node(-1);

    public:
        // for adding new nodes and edges to the maps, returns pointer to node or edge
        const TDD_Node *add_node(TDD_Node node) {
            auto pr = node_map.emplace(node, 1);
            auto it = pr.first;
            if (!pr.second) {
                // if it already exists, increment refcount unless it is terminal;
                if (!node.is_terminal()) {
                    it->second++;
                }
            }

            return &(it->first);
        }

        const TDD_Edge *add_edge(TDD_Edge edge) {
            add_node(*(edge.get_target()));
            auto pr = edge_map.emplace(edge, 1);
            auto it = pr.first;
            if (!pr.second) {
                // if it already exists, increment refcount;
                it->second++;
            }

            return &(it->first);
        }
        
        // for removing references
        void remove_node_ref(const TDD_Node *node, bool del = true) {
            TDD_Node temp = *node;
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
                }
            }
        }

        void remove_edge_ref(const TDD_Edge *edge) {
            TDD_Edge temp = *edge;
            remove_node_ref(temp.get_target());
            auto it = edge_map.find(temp);
            if (it != edge_map.end()) {
                it->second -= 1;
                if (it->second == 0) {
                    // then remove the node
                    edge_map.erase(it);
                }
            }
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
        
        // would be nice to be able to estimate memory cost here, idk how though
        // since nodes can have different numbers of successors, just measuring number of nodes/edges isnt a fair assessment

};

extern TDD_Map cache_map;

inline const TDD_Edge *TDD_Node::add_successor(const TDD_Node *t, cd &w) {
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

class TDD {
    private: 
        const TDD_Node *root;
        cd in_weight;
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

        const TDD_Node *get_root() const {
            return root;
        }

        cd get_weight() const {
            return in_weight;
        }

        std::vector<size_t> get_shape() const {
            return shape;
        }

        cd get_value(xarray<size_t> indices) const {
            if (root->is_terminal()) {
                return in_weight;
            }
            if (in_weight == cd{0,0}) {
                return 0;
            }
            const TDD_Node *next = root->get_successor_ref(indices[0])->get_target();
            xarray<size_t> new_indices;
            if (!next->is_terminal()) {
                // account for node merge by tracking the axis index
                uint32_t index_diff = next->get_axis_index() - root->get_axis_index();
                new_indices = view(indices, range(index_diff, indices.size()));
            }
            return in_weight * root->get_weight(indices[0]) * next->get_value(new_indices);
        }

        bool operator==(const TDD &other) const {
            return (in_weight == other.get_weight() && root == other.get_root());
        }

        void cleanup() const {
            get_root()->cleanup();
            cache_map.remove_node_ref(get_root());
        }

};

// recursive implementation to convert arbitrary tensor into a TDD
TDD convert_tensor_to_TDD(xarray<cd> &tensor, uint8_t axis = 0) {

    if (tensor.size() == 1) {
        // if terminal node, then just return trivial TDD with in_weight equal to weight
        // Shape can be left empty at this stage?
        return TDD(cache_map.get_terminal_node(), tensor[0]);
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
    std::vector<size_t> new_shape;
    for (size_t i = 0; i < dimension; i++) {
        sv[0] = i;
        xarray<cd> new_tensor = strided_view(tensor, sv);

        TDD child = convert_tensor_to_TDD(new_tensor, axis + 1);
        if (i == 0) {
            // then take the shape and append the new dimension to the front
            new_shape = child.get_shape();
            new_shape.insert(new_shape.begin(), dimension);
        }

        const TDD_Node *next_node = child.get_root();
        cache_map.remove_node_ref(next_node, false);

        cd next_weight = child.get_weight();

        // apply normalisation while iterating through successors
        if (normalisation_weight != cd(0,0)) {
            next_weight = next_weight / normalisation_weight;
        }
        // TODO need notion of approximately equal 0 here to account for floating point errors
        else if (next_weight != cd(0,0)) {
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
    // TODO? does this need approximate equality?
    if (weight == cd(0,0)) {
        // in this case, we also need to delete subnodes as we are replacing current node with terminal
        // removing edge reference should also eliminate references to the node
        new_node.clear_successors();
        return TDD(cache_map.get_terminal_node(), 0, new_shape);
    }

    // RR3 - need to check new node to see if all successors are the same and have the same weights
    if (new_edge_set.size() == 1) {
        // if all the successors are the same, then that means we do not need this node, instead
        // direct the tdd to the successor node with the in weight
        // that means the other successors need to be deleted as well (they are identical to the first anyway)
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.clear_successors();

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
TDD add_tdds(std::vector<TDD> &tdds, bool first = true) {
    std::vector<const TDD_Node *> roots;
    std::set<const TDD_Node *> root_set;
    cd weight_sum = 0;
    uint8_t min_axis_index = tdds[0].get_root()->get_axis_index();
    size_t dimension = tdds[0].get_root()->get_dimension();
    for (TDD tdd : tdds) {
        const TDD_Node *current_root = tdd.get_root();
        roots.push_back(current_root);

        // set to keep track of whether they are all equal
        root_set.insert(current_root);

        // sum in weights in case all the roots are the same
        weight_sum += tdd.get_weight();

        // find minimum axis index to use as next level of resulting TDD
        uint8_t axis_index = current_root->get_axis_index();
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
        return TDD(roots[0], weight_sum, tdds[0].get_shape());
    }

    // otherwise start generating new node, with the minimum axis index
    TDD_Node new_node(min_axis_index);
    cd weight = 1;
    // since addition preserves shape, we can set the shape as being the shape of the first TDD
    // it is only crucial to preserve the shape at the top level, as the value of shape is not actually
    // used, its just necessary to maintain the correct shape
    std::vector<size_t> shape = tdds[0].get_shape();

    // compute successors, normalising in the process
    cd normalisation_weight = 0;
    std::set<const TDD_Edge *> new_edge_set;
    for (size_t i = 0; i < dimension; i++) {
        std::vector<TDD> sub_tdds;
        // calculate correct sub_tdds for each tdd being added
        // indexed over the current index
        for (size_t j = 0; j < roots.size(); j++) {
            const TDD_Node *root = roots[j];
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
        // TODO need notion of approximately equal 0 here to account for floating point errors
        else if (next_weight != cd(0,0)) {
            normalisation_weight = next_weight;
            next_weight = 1;
        }

        new_edge_set.insert(new_node.add_successor(next_node, next_weight));
    }

    // we can now clean up the summands used at this step (any of those with axis_index == min_axis_index)
    for (size_t i = 0; i < roots.size(); i++) {
        const TDD_Node *root = roots[i];
        // only clean up if we actually used the index at this stage
        if (root->get_axis_index() == min_axis_index) {
            root->clear_successors();
        }
        // clean up top level tdds if its the top level call
        if (first) {
            cache_map.remove_node_ref(root);
        }
    }

    // apply remaining reductions
    weight *= normalisation_weight;
    // RR2 - see convert to tdd for more info
    if (weight == cd(0,0)) {
        new_node.clear_successors();
        return TDD(cache_map.get_terminal_node(), 0, shape);
    }

    // RR3 - see convert to tdd for more info
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

// axes are assumed to both be in ascending order, and same length
// each axis at the same index should have the same dimension

// resulting shape order will be indices before first contraction index for first TDD,
// followed by indices before first contraction index for second TDD, and so on
// e.g. T1 (a, b, c) T2 (d, b, e) contracted on d results in T (a, d, c, e)

// also TODO: apply cleanup of unused nodes and edges during execution

TDD contract_tdds(TDD &first, TDD &second, std::vector<uint8_t> first_axes, std::vector<uint8_t> second_axes, uint8_t axis = 0) {

    std::vector<size_t> f_shape = first.get_shape();
    std::vector<size_t> s_shape = second.get_shape();
    // check if both TDDs are trivial (i.e. root is terminal)
    if (first.get_root()->is_terminal() && second.get_root()->is_terminal()) {
        // compute the new weight post contraction
        double contraction_product = 1;
        std::set<uint8_t> removed_f_axes;
        std::set<uint8_t> removed_s_axes;
        for (size_t i = 0; i < first_axes.size(); i++) {
            contraction_product *= f_shape[first_axes[i]];
            removed_f_axes.insert(first_axes[i]);
            removed_s_axes.insert(second_axes[i]);
        }
        cd new_weight = first.get_weight() * second.get_weight() * contraction_product;

        // compute new shape (ordered by first TDDs axes first, followed by second TDDs axes)
        std::vector<size_t> new_shape;
        for (uint8_t i = 0; i < f_shape.size(); i++) {
            // zero check is to make sure we do not include logically removed parts of the shape
            if (removed_f_axes.find(i) == removed_f_axes.end() && f_shape[i] != 0) {
                new_shape.push_back(f_shape[i]);
            }
        }
        for (uint8_t i = 0; i < s_shape.size(); i++) {
            if (removed_s_axes.find(i) == removed_s_axes.end() && s_shape[i] != 0) {
                new_shape.push_back(s_shape[i]);
            }
        }

        return TDD(first.get_root(), new_weight, new_shape);
    }

    const TDD_Node *f_root = first.get_root();
    const TDD_Node *s_root = second.get_root();

    uint8_t first_axis = f_root->get_axis_index();
    uint8_t second_axis = s_root->get_axis_index();
    // if either node is terminal, overwrite the axis index with the length of the shape (as it isnt really valid to index over)
    if (f_root->is_terminal()) {
        first_axis = f_shape.size() - 1;
    }
    if (s_root->is_terminal()) {
        second_axis = s_shape.size() - 1;
    }
    // should I be taking axis number differently here?
    // could it be the first non-zero part of the shape instead?
    // this may be less efficient but also correct?
    // also should not affect efficiency very much actually, as depth is assumed to not be too high
    // uint8_t first_axis = f_root->get_axis_index();
    // uint8_t second_axis = s_root->get_axis_index();

    std::cout << "first_axis: " << (size_t)first_axis << " second_axis: " << (size_t)second_axis << std::endl;

    // start generating new node, with the minimum axis index
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
    else if (f_root->is_terminal()) {
        // then second one is not terminal, but may require contraction?
        if (second_axis >= second_axes[0]) {
            // contract in this case
            dimension = s_shape[second_axis];
            indexing_scheme = 0;
        }
        else {
            dimension = s_shape[second_axis];
            indexing_scheme = 2;
        }
    }
    else if (s_root->is_terminal()) {
        // then first is not terminal
        if (first_axis >= first_axes[0]) {
            dimension = f_shape[first_axis];
            indexing_scheme = 0;
        }
        else {
            dimension = f_shape[first_axis];
            indexing_scheme = 1;
        }
    }
    else if (first_axis >= first_axes[0] && second_axis >= second_axes[0]) {
        // Only case where contraction is necessary
        // need to purge first axis index for both
        dimension = f_shape[first_axis];
        indexing_scheme = 0;
    }
    else if (first_axis < first_axes[0]) {
        // successors can also be directly extracted for contraction, however we do not reduce the number
        // of axes left to contract
        // For this one we take first_axis as successor to index through
        dimension = f_shape[first_axis];
        indexing_scheme = 1;
    }
    else /* second_axis < second_axes[0] */ {
        // for this one we take second_axis as successor to index through
        dimension = s_shape[second_axis];
        indexing_scheme = 2;
    }
    std::cout << "scheme: " << indexing_scheme << std::endl;

    // shape to construct
    // Case 1 - (Dimension, Newly Skipped axes of first, Child_Shape)
    // Case 2 - (Dimension, Newly Skipped axes of second, Child_Shape) 
    // For cases 0 and 3, addition takes care of shape considerations
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

        std::vector<uint8_t> new_first_axes = first_axes;
        std::vector<uint8_t> new_second_axes = second_axes;
        uint8_t new_axis = axis;

        // need to select new successors depending on indexing_scheme
        // also determines how which axes to skip ahead
        // TODO Another Edge Case, what if we are in indexing_scheme 0, but have not
        // processed all indices prior to the contraction number
        // do we really want to set all of the shape to 0?
        switch(indexing_scheme) {
            case 0:
                // Index Conditionally and Contract
                // only bump up the ones which are equal
                if (first_axis == first_axes[0]) {
                    first_succ_node = f_root->get_successor_ref(i)->get_target();
                    first_succ_weight *= f_root->get_successor_ref(i)->get_weight();
                    // increment new_axis to account for skipped axes
                    if (first_succ_node->is_terminal()) {
                        new_axis += f_shape.size() - first_axis - 1;
                    }
                    else {
                        new_axis += first_succ_node->get_axis_index() - first_axis;
                    }
                    // decrement by 1 to account for contracted axis
                    new_axis -= 1;
                }
                if (second_axis == second_axes[0]) {
                    second_succ_node = s_root->get_successor_ref(i)->get_target();
                    second_succ_weight *= s_root->get_successor_ref(i)->get_weight();
                    // increment new_axis to account for skipped axes
                    if (second_succ_node->is_terminal()) {
                        new_axis += s_shape.size() - second_axis - 1;
                    }
                    else {
                        new_axis += second_succ_node->get_axis_index() - second_axis;
                    }
                    // decrement by 1 to account for contracted axis
                    new_axis -= 1;
                }

                // correct shapes through logical removal
                // basically eliminates all parts of the shape that have now been accounted for
                for (size_t j = 0; j <= first_axes[0]; j++) {
                    first_succ_shape[j] = 0;
                }
                for (size_t j = 0; j <= second_axes[0]; j++) {
                    second_succ_shape[j] = 0;
                }
                break;
            case 1:
                {
                    // Index First
                    first_succ_node = f_root->get_successor_ref(i)->get_target();
                    first_succ_weight *= f_root->get_successor_ref(i)->get_weight();
                    // increment by number of axes we progress by 
                    // ALSO NEED TO ENSURE NEW SUCCESSORS ARE NOT TERMINAL
                    size_t first_limit = first_succ_node->get_axis_index();
                    if (first_succ_node->is_terminal()) {
                        first_limit = f_shape.size() - 1;
                    }
                    new_axis += first_limit - first_axis;
                    // correct shapes through logical removal
                    for (size_t j = 0; j <= first_axis; j++) {
                        first_succ_shape[j] = 0;
                    }
                    // if its the first child, then build the shape of the result
                    if (i == 0) {
                        for (size_t j = first_axis + 1; j < first_limit; j++) {
                            shape.push_back(f_shape[j]);
                        }
                    }
                }
                break;
            case 2:
                {
                    // Index Second
                    second_succ_node = s_root->get_successor_ref(i)->get_target();
                    second_succ_weight *= s_root->get_successor_ref(i)->get_weight();
                    // increment by number of axis we progress by
                    // ALSO NEED TO ENSURE NEW SUCCESSORS ARE NOT TERMINAL
                    size_t second_limit = second_succ_node->get_axis_index();
                    if (second_succ_node->is_terminal()) {
                        second_limit = s_shape.size() - 1;
                    }
                    new_axis += second_limit - second_axis;
                    // correct shapes through logical removal
                    for (size_t j = 0; j <= second_axis; j++) {
                        second_succ_shape[j] = 0;
                    }
                    // if its the first child, then build the shape of the result
                    if (i == 0) {
                        for (size_t j = second_axis + 1; j < second_limit; j++) {
                            shape.push_back(s_shape[j]);
                        }
                    }
                }
                break;
        }

        if (indexing_scheme == 0) {
            // drop first value as we are contracting that axis at this step
            new_first_axes = std::vector<uint8_t>(first_axes.begin() + 1, first_axes.end());
            new_second_axes = std::vector<uint8_t>(second_axes.begin() + 1, second_axes.end());
        }
        TDD first_successor = TDD(first_succ_node, first_succ_weight, first_succ_shape);
        TDD second_successor = TDD(second_succ_node, second_succ_weight, second_succ_shape);

        TDD child = contract_tdds(first_successor, second_successor, new_first_axes, new_second_axes, new_axis);
        new_tdds.push_back(child);

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
        // TODO need notion of approximately equal 0 here to account for floating point errors
        else if (next_weight != cd(0,0)) {
            normalisation_weight = next_weight;
            next_weight = 1;
        }

        new_edge_set.insert(new_node.add_successor(next_node, next_weight));
    }

    if (indexing_scheme == 0) {
        // if we are contracting an index at this step, then return the sum of all the contracted TDDs
        // should already be reduced thanks to add_tdds automatically reducing
        // this will automatically clean up the summands too
        return add_tdds(new_tdds);
    }

    // otherwise, clean up the recursive elements and reduce the TDD
    // TODO Clean up

    // Begin reduction
    weight *= normalisation_weight;
    // RR2 - see convert to tdd for more info
    if (weight == cd(0,0)) {
        new_node.clear_successors();
        return TDD(cache_map.get_terminal_node(), 0, shape);
    }

    // RR3 - see convert to tdd for more info
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

#endif