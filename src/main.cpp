/**
 * Batch Edge Update Simulator for APSP in MPC
 * 
 * This program simulates the round complexity of processing a batch
 * of edge deletions using the algorithm described in:
 * "Batch Edge Updates for Fully Dynamic All-Pairs Shortest Paths in the MPC Model"
 * 
 * It compares batch processing against sequential processing and outputs
 * the simulated round counts and speedup.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include <unordered_set>
#include <queue>
#include <cassert>

// ----------------------------------------------------------------------
// MPC Model Parameters
// ----------------------------------------------------------------------
const double DELTA = 0.5;           // memory per machine O(n^delta)
const int SORT_ROUNDS = 2;          // O(1/delta) -> constant rounds
const int BROADCAST_ROUNDS = 2;
const int PREFIX_MIN_ROUNDS = 2;

// ----------------------------------------------------------------------
// Round Cost Functions (as derived in the paper)
// ----------------------------------------------------------------------

/**
 * Round cost for matrix multiplication on (min,+) semiring.
 * Lemma 2 in paper: O(1 + ceil((1 - delta/2)/delta)) rounds.
 */
int matrixMultiplyRounds() {
    return 1 + static_cast<int>(std::ceil((1.0 - DELTA / 2.0) / DELTA));
}

/**
 * Simulates the round count for Algorithm 2 (BatchDelete) on a single edge.
 * This is the baseline sequential cost.
 */
int sequentialDeleteRounds(int n, int m) {
    int rounds = 0;
    
    // Step 1: Identify deleted tree edges (local check) - O(1)
    rounds += 1;
    
    // Step 2: Split tree (Euler tour split) - O(1/delta)
    rounds += BROADCAST_ROUNDS;
    
    // Step 3: Find replacement edge (sort + prefix min) - O(1/delta)
    rounds += SORT_ROUNDS + PREFIX_MIN_ROUNDS;
    
    // Step 4: Update tree - O(1/delta)
    rounds += BROADCAST_ROUNDS;
    
    // Step 5: Update distance matrix (matrix multiplication)
    // For sequential, we do this once per deletion.
    int logn = static_cast<int>(std::ceil(std::log2(n)));
    rounds += logn * matrixMultiplyRounds();
    
    // Step 6: Cascading effects (for single deletion, log 1 = 0)
    
    return rounds;
}

/**
 * Simulates the round count for Algorithm 2 (BatchDelete) on a batch of k edges.
 * The key insight: parallel replacement edge search handles all k deletions
 * simultaneously in O(1/delta) rounds.
 */
int batchDeleteRounds(int n, int k, int m) {
    int rounds = 0;
    int logn = static_cast<int>(std::ceil(std::log2(n)));
    int logk = static_cast<int>(std::ceil(std::log2(k + 1))); // log k for cascading
    
    // Step 1: Identify deleted tree edges (parallel local check) - O(1)
    rounds += 1;
    
    // Step 2: Split trees (parallel Euler tour splits) - O(1/delta)
    rounds += BROADCAST_ROUNDS;
    
    // Step 3: Parallel replacement edge search
    // - Local scan: O(1)
    // - MPC Sort of O(k*m) items: O(1/delta)
    // - Parallel prefix minima: O(1/delta)
    rounds += 1 + SORT_ROUNDS + PREFIX_MIN_ROUNDS;
    
    // Step 4: Update trees (parallel Euler tour merges) - O(1/delta)
    rounds += BROADCAST_ROUNDS;
    
    // Step 5: Update distance matrix
    // Recompute affected trees using restricted Bellman-Ford (parallel across roots)
    // and matrix multiplication. The paper states this is O(log n * (1/delta + log n))
    // For batch, we do this once for all affected roots.
    rounds += logn * (BROADCAST_ROUNDS + matrixMultiplyRounds());
    
    // Step 6: Cascading effects - at most O(log k) iterations
    rounds += logk * (BROADCAST_ROUNDS + SORT_ROUNDS + PREFIX_MIN_ROUNDS);
    
    return rounds;
}

/**
 * Generate an Erdős–Rényi random graph G(n, p).
 * Returns adjacency list and edge list.
 */
struct Graph {
    int n;
    int m;
    std::vector<std::vector<int>> adj;
    std::vector<std::pair<int,int>> edges;
};

Graph generateErdosRenyi(int n, double p, std::mt19937& gen) {
    Graph G;
    G.n = n;
    G.adj.resize(n);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (dis(gen) < p) {
                G.adj[u].push_back(v);
                G.adj[v].push_back(u);
                G.edges.emplace_back(u, v);
            }
        }
    }
    G.m = static_cast<int>(G.edges.size());
    return G;
}

/**
 * Compute a spanning forest of the graph using BFS.
 * Returns a list of tree edges.
 */
std::vector<std::pair<int,int>> computeSpanningForest(const Graph& G) {
    std::vector<bool> visited(G.n, false);
    std::vector<std::pair<int,int>> treeEdges;
    
    for (int start = 0; start < G.n; ++start) {
        if (visited[start]) continue;
        
        std::queue<int> q;
        q.push(start);
        visited[start] = true;
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v : G.adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    treeEdges.emplace_back(u, v);
                    q.push(v);
                }
            }
        }
    }
    return treeEdges;
}

/**
 * Run a single trial for given n and k.
 * Returns (batch_rounds, sequential_rounds).
 */
std::pair<int, int> runTrial(int n, int k, std::mt19937& gen) {
    // Generate graph with average degree d = 5 => p = d / n
    double p = 5.0 / n;
    Graph G = generateErdosRenyi(n, p, gen);
    
    // Compute spanning forest (simulate preprocessing)
    std::vector<std::pair<int,int>> treeEdges = computeSpanningForest(G);
    
    // If not enough tree edges, reduce k
    int actual_k = std::min(k, static_cast<int>(treeEdges.size()));
    if (actual_k == 0) {
        return {0, 0};  // degenerate case
    }
    
    // Randomly select k tree edges to delete
    std::shuffle(treeEdges.begin(), treeEdges.end(), gen);
    // (We don't actually delete them; the simulation only needs n, k, m)
    
    int batch_rounds = batchDeleteRounds(n, actual_k, G.m);
    int seq_rounds = actual_k * sequentialDeleteRounds(n, G.m);
    
    return {batch_rounds, seq_rounds};
}

int main() {
    // ------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------
    const int N = 5000;                        // number of vertices
    const std::vector<int> K_VALUES = {1, 10, 50, 100, 200, 500};
    const int TRIALS = 10;                     // number of independent trials
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // ------------------------------------------------------------------
    // Output header
    // ------------------------------------------------------------------
    std::cout << "Batch Edge Deletion Simulation for APSP in MPC\n";
    std::cout << "================================================\n";
    std::cout << "n = " << N << ", trials = " << TRIALS << "\n";
    std::cout << "MPC parameter delta = " << DELTA << "\n\n";
    
    std::cout << std::left << std::setw(10) << "k"
              << std::setw(15) << "Batch Rounds"
              << std::setw(18) << "Sequential Rounds"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(55, '-') << "\n";
    
    // Also save to CSV file
    std::ofstream csv_file("simulation_results.csv");
    csv_file << "k,BatchRounds,SequentialRounds,Speedup\n";
    
    // ------------------------------------------------------------------
    // Run experiments
    // ------------------------------------------------------------------
    for (int k : K_VALUES) {
        long long total_batch = 0;
        long long total_seq = 0;
        
        for (int t = 0; t < TRIALS; ++t) {
            auto [batch_r, seq_r] = runTrial(N, k, gen);
            total_batch += batch_r;
            total_seq += seq_r;
        }
        
        double avg_batch = static_cast<double>(total_batch) / TRIALS;
        double avg_seq = static_cast<double>(total_seq) / TRIALS;
        double speedup = avg_seq / avg_batch;
        
        std::cout << std::left << std::setw(10) << k
                  << std::setw(15) << std::fixed << std::setprecision(1) << avg_batch
                  << std::setw(18) << std::fixed << std::setprecision(1) << avg_seq
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x\n";
        
        csv_file << k << ","
                 << std::fixed << std::setprecision(1) << avg_batch << ","
                 << std::fixed << std::setprecision(1) << avg_seq << ","
                 << std::fixed << std::setprecision(2) << speedup << "\n";
    }
    
    csv_file.close();
    
    std::cout << "\nResults saved to simulation_results.csv\n";
    
    return 0;
}