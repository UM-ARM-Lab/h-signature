#include <h_signature/h_signature.h>
#include <iostream>
#include <chrono>

int main() {
    Loop loop(3, 5);
    loop << 0, 1, 1, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 1, 0;
    Loop obs_loop(3, 5);
    obs_loop << 0.5, 0.5, 0.5, 0.5, 0.5,
                -0.5, -0.5, 0.5, 0.5, -0.5,
                0.5, 0.5, 1.5, 1.5, 0.5;
    Skeleton skeleton = {{"obs1", obs_loop}};
    // time it
    auto start = std::chrono::high_resolution_clock::now();
    HSignature h_signature = get_h_signature(loop, skeleton);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "H-signature: " << h_signature << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    return 0;
}