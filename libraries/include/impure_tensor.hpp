#ifndef O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
#define O3_SIGMA_MODEL_IMPURE_TENSOR_HPP

#include "tensor.hpp"
#include <vector>

#define DIMENSION 3

template<class Tensor>
class BaseImpureTensor {
public:
    int distance{0};
    bool isMerged{false};
    bool isImpure{false};
    std::vector<double> corrs;
    Tensor tensors[DIMENSION];

    BaseImpureTensor() {
        tensors[0] = Tensor();
        tensors[1] = Tensor();
        tensors[2] = Tensor();
    }

    BaseImpureTensor(int D) {
        tensors[0] = Tensor(D);
        tensors[1] = Tensor(D);
        tensors[2] = Tensor(D);
    }

    BaseImpureTensor(int D, int D_max) {
        tensors[0] = Tensor(D, D_max);
        tensors[1] = Tensor(D, D_max);
        tensors[2] = Tensor(D, D_max);
    }

    BaseImpureTensor(int Di, int Dj, int Dk, int Dl) {
        tensors[0] = Tensor(Di, Dj, Dk, Dl);
        tensors[1] = Tensor(Di, Dj, Dk, Dl);
        tensors[2] = Tensor(Di, Dj, Dk, Dl);
    }

    BaseImpureTensor(int Di, int Dj, int Dk, int Dl, int D_max) {
        tensors[0] = Tensor(Di, Dj, Dk, Dl, D_max);
        tensors[1] = Tensor(Di, Dj, Dk, Dl, D_max);
        tensors[2] = Tensor(Di, Dj, Dk, Dl, D_max);
    }

    BaseImpureTensor(int d, BaseImpureTensor<Tensor> &T) {
        this->distance = d;
        tensors[0] = Tensor(T.tensors[0]);
        tensors[1] = Tensor(T.tensors[1]);
        tensors[2] = Tensor(T.tensors[2]);
    }

    BaseImpureTensor(BaseImpureTensor<Tensor> &rhs) {
        distance = rhs.distance;
        corrs.clear();
        tensors[0] = rhs.tensors[0];
        tensors[1] = rhs.tensors[1];
        tensors[2] = rhs.tensors[2];
    }

    ~BaseImpureTensor() {
        corrs.clear();
    }

    BaseImpureTensor<Tensor> &operator=(const BaseImpureTensor<Tensor> &rhs) {
        distance = rhs.distance;
        tensors[0] = rhs.tensors[0];
        tensors[1] = rhs.tensors[1];
        tensors[2] = rhs.tensors[2];
        return *this;
    }

    static int normalization(Tensor &T, BaseImpureTensor<Tensor> &originIMT);

    static int normalization(Tensor &T, BaseImpureTensor<Tensor> &originIMT, std::vector<BaseImpureTensor<Tensor>> &IMTs);
};

#endif //O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
