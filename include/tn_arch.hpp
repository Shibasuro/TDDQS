#ifndef TNARCH
#define TNARCH
#include <vector>
#include <utility>

class TN_Node {
    private:
        uint32_t nindex;
        std::vector<uint32_t> neighbours;
    public:
        TN_Node(const uint32_t &node_index) {
            nindex = node_index;
        }
        void add_neighbour(const uint32_t &neighbour) {
            neighbours.push_back(neighbour);
        }
        uint32_t degree() {
            return neighbours.size();
        }
        std::vector<uint32_t> get_neighbours() {
            return neighbours;
        }
        uint32_t get_index(uint32_t target_node) {
            auto it = find(neighbours.begin(), neighbours.end(), target_node);
            if (it != neighbours.end()) {
                return it - neighbours.begin();
            }
            return degree();
        }
};

class TN_Arch {
    private:
        uint32_t nnodes;
        std::vector<TN_Node> nodes;
        void compute_neighbours(const std::vector<std::pair<uint32_t, uint32_t>> &edges) {
            for (uint32_t i = 0; i < nnodes; i++) {
                TN_Node new_node(i);
                nodes.push_back(new_node);
            }
            for (std::pair<uint32_t, uint32_t> e : edges) {
                uint32_t i = e.first;
                uint32_t j = e.second;
                nodes[i].add_neighbour(j);
                nodes[j].add_neighbour(i);
            }
        }
    public:
        TN_Arch() {
            
        }
        TN_Arch(const uint32_t &num_nodes, const std::vector<std::pair<uint32_t, uint32_t>> &edges) {
            nnodes = num_nodes;
            compute_neighbours(edges);
        }
        std::vector<TN_Node> get_nodes() {
            return nodes;
        }
        TN_Node get_node(uint32_t nindex) {
            return nodes[nindex];
        }
        uint32_t size() {
            return nodes.size();
        }
        

};

TN_Arch MPS_Arch(uint32_t num_nodes) {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t i = 0; i < num_nodes - 1; i++) {
        edges.push_back(std::pair<uint32_t, uint32_t>(i, i + 1));
    }
    return TN_Arch(num_nodes, edges);
}

#endif