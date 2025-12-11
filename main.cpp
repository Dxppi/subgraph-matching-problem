#include <iostream>
#include <chrono>
#include <iomanip>
#include "Algorithms.hpp"
#include "Matrix.hpp"
#include <LAGraph.h>
#include <LAGraphX.h>
#include <GraphBLAS.h>


Matrix create_complete_graph(size_t n) {
    Matrix adj(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) adj.set(i, j, 1);
        }
    }
    return adj;
}

int main() {
    size_t n = 5;
    Matrix adj(n);
    
    for (size_t i = 0; i < n; ++i) 
        for (size_t j = 0; j < n; ++j) 
            if (i != j) adj.set(i, j, 1);

    std::cout << "--- Graph: K5 (Complete graph with 5 vertices) ---" << std::endl;
    
    std::cout << "3-Cliques (k=3): " << solve_k_clique(adj, 3u) << " (Expected: 10)" << std::endl;
    std::cout << "4-Cliques (k=4): " << solve_k_clique(adj, 4u) << " (Expected: 5)" << std::endl;
    std::cout << "5-Cliques (k=5): " << solve_k_clique(adj, 5u) << " (Expected: 1)" << std::endl;

    Matrix adj_noise(6);
    
    for (size_t i = 0; i < 5; ++i) 
        for (size_t j = 0; j < 5; ++j) 
            if (i != j) adj_noise.set(i, j, 1);
    
    adj_noise.set(0, 5, 1); adj_noise.set(5, 0, 1);

    std::cout << "\n--- Graph: K5 + 1 loose edge (6 vertices) ---" << std::endl;
    
    std::cout << "5-Cliques (Alg 5 prune): " << solve_k_clique(adj_noise, 5u) << " (Expected: 1)" << std::endl;
    
    SolverConfig config_alg6;
    config_alg6.prep = SolverConfig::ALG_6;
    std::cout << "5-Cliques (Alg 6 prune): " << solve_k_clique(adj_noise, 5u, config_alg6) << " (Expected: 1)" << std::endl;
    
    // ----------------------------------------------------
    // СРАВНИТЕЛЬНЫЙ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ
    // ----------------------------------------------------
    size_t test_k = 4u;
    Matrix K_test = create_complete_graph(6); 

    SolverConfig parallel_config;
    parallel_config.parallel_enabled = true; 

    SolverConfig sequential_config;
    sequential_config.parallel_enabled = false; 

    std::cout << "\n--- Performance Comparison (K6, k=" << test_k << ") ---" << std::endl;

    auto start_seq = std::chrono::high_resolution_clock::now();
    count_t count_seq = solve_k_clique(K_test, test_k, sequential_config);
    auto end_seq = std::chrono::high_resolution_clock::now();
    auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq).count();

    auto start_par = std::chrono::high_resolution_clock::now();
    count_t count_par = solve_k_clique(K_test, test_k, parallel_config);
    auto end_par = std::chrono::high_resolution_clock::now();
    auto duration_par = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par).count();

    std::cout << "k=" << test_k << " count: " << count_par << " (Expected: 15)" << std::endl;
    std::cout << "Sequential Time: " << duration_seq << " ms" << std::endl;
    std::cout << "Parallel Time:   " << duration_par << " ms" << std::endl;
    
    if (duration_par > 0) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Speedup:         " << (double)duration_seq / duration_par << "x" << std::endl;
    }

    return 0;
}