#ifndef O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
#define O3_SIGMA_MODEL_IMPURE_TENSOR_HPP

#include "tensor.hpp"
#include <vector>

class ImpureTensor {
public:
    int distance{0};
    int mergeIndex{0};
    bool isMerged{false};
    std::vector<double> corrs;
    Tensor tensors[3];

    ImpureTensor();

    ImpureTensor(int D, int N);

    ImpureTensor(int D, int D_max, int N);

    ImpureTensor(int Di, int Dj, int Dk, int Dl, int N);

    ImpureTensor(int Di, int Dj, int Dk, int Dl, int D_max, int N);

    ImpureTensor(int d, ImpureTensor &T);

    ImpureTensor(ImpureTensor &rhs);

    ~ImpureTensor();

    ImpureTensor &operator=(const ImpureTensor &rhs);

    static int normalization(Tensor &T, ImpureTensor &originIMT);

    static int normalization(Tensor &T, ImpureTensor &originIMT, std::vector<ImpureTensor> &IMTs);
};

#endif //O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
