//
// Created by Joy on 2020/06/28.
//

#ifndef O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
#define O3_SIGMA_MODEL_IMPURE_TENSOR_HPP

#include "tensor.hpp"
#include <vector>

class ImpureTensor {
public:
    int distance{0};
    bool isMerged{false};
    std::vector<double> corrs;
    Tensor tensors[3];

    ImpureTensor();

    ImpureTensor(int D);

    ImpureTensor(int Dx, int Dy, int Dx_max, int Dy_max);

    ImpureTensor(int d, ImpureTensor &T);

    ImpureTensor(ImpureTensor &rhs);

    ~ImpureTensor();

    ImpureTensor &operator=(const ImpureTensor &rhs);
};

#endif //O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
