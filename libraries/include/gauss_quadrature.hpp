#ifndef O3_SIGMA_MODEL_GAUSS_QUADRATURE_HPP
#define O3_SIGMA_MODEL_GAUSS_QUADRATURE_HPP

#include "tensor.hpp"
#include "impure_tensor.hpp"

namespace GaussQuadrature {
    void initTensor(const double &K, const int &n_node, const int &D_cut, Tensor &T);
    void initTensorWithImpure(const double &K, const int &n_node, const int &D_cut, const int &D, Tensor &T, ImpureTensor &IMT);
}

#endif //O3_SIGMA_MODEL_GAUSS_QUADRATURE_HPP
