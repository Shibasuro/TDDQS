#ifndef TNARCH
#define TNARCH
#include <vector>
#include <utility>

using namespace std;

class TN_Node {
    private:
        uint32_t nindex;
        vector<uint32_t> neighbours;
    public:
        TN_Node(const uint32_t node_index) {
            nindex = node_index;
        }
        add_neighbour(const uint32_t neighbour) {
            neighbours.push_back(neighbour)
        }
        uint32_t degree() {
            return neighbours.size();
        }
}

class TN_Arch {
    private:
        uint32_t nnodes;
        vector<TN_Node> nodes;
        compute_neighbours(const vector<pair<uint32_t, uint32_t>> edges) {
            for (uint32_t i = 0; i < nnodes; i++) {
                TN_Node new_node(i);
                nodes[i] = new_node;
            }
            for (pair<uint32_t, uint32_t> e : edges) {
                uint32_t i = e.first;
                uint32_t j = e.second;
                nodes[i].add_neighbour(j);
                nodes[j].add_neighbour(i);
            }
        }
    public:
        TN_Arch(const uint32_t num_nodes, const vector<pair<uint32_t, uint32_t>> edges) {
            nnodes = num_nodes;
            compute_neighbours(edges);
        }
        vector<TN_Node> get_nodes() {
            return nodes;
        }

}

#endif