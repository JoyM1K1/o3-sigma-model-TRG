#ifndef O3_SIGMA_MODEL_SPHERICALHARMONICS_HPP
#define O3_SIGMA_MODEL_SPHERICALHARMONICS_HPP

#include "tensor.hpp"
#include "impure_tensor.hpp"

namespace SphericalHarmonics {
    void initTensor(const double &K, const int &l_max, Tensor &T);
    void initTensorWithImpure(const double &K, const int &l_max, Tensor &T, ImpureTensor &IMT);
}

#endif //O3_SIGMA_MODEL_SPHERICALHARMONICS_HPP
