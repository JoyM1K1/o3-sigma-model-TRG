#ifndef O3_SIGMA_MODEL_GAUSS_QUADRATURE_HPP
#define O3_SIGMA_MODEL_GAUSS_QUADRATURE_HPP

#include "tensor.hpp"
#include "impure_tensor.hpp"
#include "TRG.hpp"
#include "HOTRG.hpp"

namespace GaussQuadrature {
    void init_tensor(const double &K, const int &n_node, const int &D_cut, BaseTensor &T);

    template<class Tensor>
    void init_tensor_with_impure(const double &K, const int &n_node, const int &D_cut, const int &D, Tensor &T, BaseImpureTensor<Tensor> &IMT);

    extern template void
    init_tensor_with_impure(const double &K, const int &n_node, const int &D_cut, const int &D, TRG::Tensor &T, BaseImpureTensor<TRG::Tensor> &IMT);

    extern template void
    init_tensor_with_impure(const double &K, const int &n_node, const int &D_cut, const int &D, HOTRG::Tensor &T, BaseImpureTensor<HOTRG::Tensor> &IMT);
}

#endif //O3_SIGMA_MODEL_GAUSS_QUADRATURE_HPP
