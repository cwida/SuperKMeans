#include "superkmeans/superkmeans.h"
#include <iostream>
#include <vector>

int main() {
    const size_t n = 1000;
    const size_t d = 32;
    const size_t k = 10;

    std::vector<float> data(n * d);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data)
        v = dist(rng);

    skmeans::SuperKMeans<> skm(k, d);
    auto centroids = skm.Train(data.data(), n);
    auto assignments = skm.Assign(data.data(), centroids.data(), n, k);

    if (assignments.size() != n) {
        std::cerr << "FAIL: expected " << n << " assignments, got " << assignments.size()
                  << std::endl;
        return 1;
    }

    std::cout << "FetchContent integration test: PASSED" << std::endl;
    return 0;
}
