// FILEPATH: src/memory_management/SeamGraph.h
#ifndef SEAMGRAPH_H
#define SEAMGRAPH_H

#include <vector>
#include <memory>

class SeamGraph {
public:
    struct SeamEdge {
        int start;
        int end;
        float weight;
    };

private:
    std::vector<SeamEdge> edges;
    bool seamResolved = false;

public:
    void AddEdge(int start, int end, float weight) {
        seamResolved = false;
        edges.push_back({start, end, weight});
    }

    void ResolveSeams() { /* Seam resolution logic here */ }
};

#endif // SEAMGRAPH_H
