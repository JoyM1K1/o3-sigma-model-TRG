#ifndef O3_SIGMA_MODEL_SPHERICALHARMONICS_HPP
#define O3_SIGMA_MODEL_SPHERICALHARMONICS_HPP

#include "tensor.hpp"
#include "impure_tensor.hpp"
#include "TRG.hpp"
#include "HOTRG.hpp"

namespace SphericalHarmonics {
    void init_tensor(const double &K, const int &l_max, BaseTensor &T);

    template<class Tensor>
    void init_tensor_with_impure(const double &K, const int &l_max, Tensor &T, BaseImpureTensor<Tensor> &IMT);

    extern template void init_tensor_with_impure(const double &K, const int &l_max, TRG::Tensor &T, BaseImpureTensor<TRG::Tensor> &IMT);

    extern template void init_tensor_with_impure(const double &K, const int &l_max, HOTRG::Tensor &T, BaseImpureTensor<HOTRG::Tensor> &IMT);
}

#endif //O3_SIGMA_MODEL_SPHERICALHARMONICS_HPP
