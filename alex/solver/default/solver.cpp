#include <iostream>
#include "../../Eigen/Dense"

int main() {
    Eigen::Matrix2f m;
    m << 1, 2,
         3, 4;
    std::cout << m << std::endl;
    return 0;
}