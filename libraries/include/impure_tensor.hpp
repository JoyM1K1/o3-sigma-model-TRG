#ifndef O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
#define O3_SIGMA_MODEL_IMPURE_TENSOR_HPP

#include "tensor.hpp"
#include <vector>

#define REP(i, N) for (int i = 0; i < (N); ++i)

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
        for (auto &tensor : tensors) tensor = Tensor();
    }

    explicit BaseImpureTensor(int D) {
        for (auto &tensor : tensors) tensor = Tensor(D);
    }

    BaseImpureTensor(int D, int D_max) {
        for (auto &tensor : tensors) tensor = Tensor(D, D_max);
    }

    BaseImpureTensor(int Di, int Dj, int Dk, int Dl) {
        for (auto &tensor : tensors) tensor = Tensor(Di, Dj, Dk, Dl);
    }

    BaseImpureTensor(int Di, int Dj, int Dk, int Dl, int D_max) {
        for (auto &tensor : tensors) tensor = Tensor(Di, Dj, Dk, Dl, D_max);
    }

    BaseImpureTensor(int d, BaseImpureTensor<Tensor> &T) {
        distance = d;
        REP(i, DIMENSION) tensors[i] = Tensor(T.tensors[i]);
    }

    BaseImpureTensor(BaseImpureTensor<Tensor> &rhs) {
        distance = rhs.distance;
        corrs.clear();
        REP(i, DIMENSION) tensors[i] = Tensor(rhs.tensors[i]);
    }

    ~BaseImpureTensor() {
        corrs.clear();
    }

    BaseImpureTensor<Tensor> &operator=(const BaseImpureTensor<Tensor> &rhs) {
        distance = rhs.distance;
        REP(i, DIMENSION) tensors[i] = rhs.tensors[i];
        return *this;
    }
};

#endif //O3_SIGMA_MODEL_IMPURE_TENSOR_HPP
