#ifndef TDDARCH
#define TDDARCH
#include <vector>
#include <iostream>
#include <queue>
#include <functional>
#include <set>
#include <unordered_map>
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

        void clear_successors();
        
        cd get_value (xarray<size_t> indices) const {
            if (is_terminal()) {
                return cd(1,0);
            }
            if (indices.size() == 0) {
                // this would mean an illegal index set (too short)
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

        // unique terminal node is defined here
        TDD_Node terminal_node = TDD_Node();

    public:
        // for adding new nodes and edges to the maps, returns pointer to node or edge
        const TDD_Node *add_node(TDD_Node node, bool instantiating) {
            auto pr = node_map.emplace(node, 0);
            auto it = pr.first;
            if (!pr.second && !instantiating) {
                // if it already exists, increment refcount;
                it->second++;
            }

            return &(it->first);
        }

        const TDD_Edge *add_edge(TDD_Edge edge) {
            add_node(*(edge.get_target()), false);
            auto pr = edge_map.emplace(edge, 1);
            auto it = pr.first;
            if (!pr.second) {
                // if it already exists, increment refcount;
                it->second++;
            }

            return &(it->first);
        }
        
        // for removing references
        void remove_node_ref(const TDD_Node *node) {
            TDD_Node temp = *node;
            auto it = node_map.find(temp);
            if (it != node_map.end()) {
                it->second -= 1;
                if (it->second == 0) {
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
        TDD_Node *get_terminal_node() {
            return &terminal_node;
        }

        size_t num_unique_nodes() {
            return node_map.size();
        }

        size_t num_unique_edges() {
            return edge_map.size();
        }

};

extern TDD_Map cache_map;

inline const TDD_Edge *TDD_Node::add_successor(const TDD_Node *t, cd &w) {
    TDD_Edge new_edge(t, w);
    const TDD_Edge *new_edge_ptr = cache_map.add_edge(new_edge);
    successors.push_back(new_edge_ptr);
    return new_edge_ptr;
}

inline void TDD_Node::clear_successors() {
    for (size_t i = 0; i < successors.size(); i++) {
        cache_map.remove_edge_ref(successors[i]);
    }
}

class TDD {
    private: 
        const TDD_Node *root;
        cd in_weight;
    public:
        TDD() {}
        TDD(const TDD_Node *r, cd weight) {
            root = r;
            in_weight = weight;
        }

        const TDD_Node *get_root() const {
            return root;
        }
        cd get_weight() const {
            return in_weight;
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

};

// recursive implementation to convert arbitrary tensor into a TDD
TDD convert_tensor_to_TDD(xarray<cd> tensor, uint8_t axis = 0) {

    if (tensor.size() == 1) {
        // if terminal node, then just return trivial TDD with in_weight equal to weight
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
    for (size_t i = 0; i < dimension; i++) {
        sv[0] = i;
        xarray<cd> new_tensor = strided_view(tensor, sv);

        TDD child = convert_tensor_to_TDD(new_tensor, axis + 1);

        const TDD_Node *next_node = child.get_root();

        cd next_weight = child.get_weight();

        // apply normalisation while iterating through successors
        if (normalisation_weight != cd(0,0)) {
            next_weight = next_weight / normalisation_weight;
        }
        // need notion of approximately equal 0 here to account for floating point errors
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
    // does this need approximate equality?
    if (weight == cd(0,0)) {
        // in this case, we also need to delete subnodes as we are replacing current node with terminal
        // removing edge reference should also eliminate references to the node
        new_node.clear_successors();
        return TDD(cache_map.get_terminal_node(), 0);
    }

    // RR3 - need to check new node to see if all successors are the same and have the same weights
    if (new_edge_set.size() == 1) {
        // if all the successors are the same, then that means we do not need this node, instead
        // direct the tdd to the successor node with the in weight
        // that means the other successors need to be deleted as well (they are identical to the first anyway)
        const TDD_Node next_node = *(new_node.get_successor_ref(0)->get_target());
        // delete the successors
        new_node.clear_successors();

        const TDD_Node *new_node_ptr = cache_map.add_node(next_node, true);

        return TDD(new_node_ptr, weight);
    }

    // otherwise, the new node is now reduced and we can add it to the map
    const TDD_Node *new_node_ptr = cache_map.add_node(new_node, true);

    return TDD(new_node_ptr, weight);
}

#endif