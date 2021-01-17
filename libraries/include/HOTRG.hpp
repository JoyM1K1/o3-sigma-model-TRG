#ifndef O3_SIGMA_MODEL_HOTRG_HPP
#define O3_SIGMA_MODEL_HOTRG_HPP

#include <string>
#include "tensor.hpp"
#include "impure_tensor.hpp"

namespace HOTRG {
    class Tensor : public BaseTensor {
    public:
        Tensor() : BaseTensor() {};

        Tensor(int D_cut) : BaseTensor(D_cut) {};

        Tensor(int D, int D_max) : BaseTensor(D, D_max) {};

        Tensor(int Di, int Dj, int Dk, int Dl) : BaseTensor(Di, Dj, Dk, Dl) {};

        Tensor(int Di, int Dj, int Dk, int Dl, int D_max) : BaseTensor(Di, Dj, Dk, Dl, D_max) {};

        long long int normalization(int c) override;
    };

    class ImpureTensor : public BaseImpureTensor<Tensor> {
    public:
        ImpureTensor() : BaseImpureTensor<Tensor>() {};

        ImpureTensor(int D) : BaseImpureTensor<Tensor>(D) {};

        ImpureTensor(int D, int D_max) : BaseImpureTensor<Tensor>(D, D_max) {};

        ImpureTensor(int Di, int Dj, int Dk, int Dl) : BaseImpureTensor<Tensor>(Di, Dj, Dk, Dl) {};

        ImpureTensor(int Di, int Dj, int Dk, int Dl, int D_max) : BaseImpureTensor<Tensor>(Di, Dj, Dk, Dl, D_max) {};

        ImpureTensor(int d, BaseImpureTensor<Tensor> &T) : BaseImpureTensor<Tensor>(d, T) {};

        ImpureTensor(BaseImpureTensor<Tensor> &rhs) : BaseImpureTensor<Tensor>(rhs) {};
    };

    void initialize_spherical_harmonics(Tensor &T, const double &K, const int &D_cut, const int &l_max);

    void initialize_gauss_quadrature(Tensor &T, const double &K, const int &D_cut, const int &n_node);

    void contractionX(const int &D_cut, BaseTensor &leftT, BaseTensor &rightT, const double *U, const std::string mergeT);

    void contractionY(const int &D_cut, BaseTensor &bottomT, BaseTensor &topT, const double *U, const std::string mergeT);

    void SVD_X(const int &D_cut, BaseTensor &T, double *&U);

    void SVD_Y(const int &D_cut, BaseTensor &T, double *&U);

    namespace renormalization {
        double partition_alt(Tensor &T, long long int *orders, const int &n, const int &normalize_factor);
    }
}

#endif //O3_SIGMA_MODEL_HOTRG_HPP
