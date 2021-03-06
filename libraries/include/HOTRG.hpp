#ifndef O3_SIGMA_MODEL_HOTRG_HPP
#define O3_SIGMA_MODEL_HOTRG_HPP

#include <string>
#include "tensor.hpp"
#include "impure_tensor.hpp"

namespace HOTRG {
    class Tensor : public BaseTensor {
    public:
        Tensor() : BaseTensor() {};

        explicit Tensor(int D_cut) : BaseTensor(D_cut) {};

        Tensor(int D, int D_max) : BaseTensor(D, D_max) {};

        Tensor(int Di, int Dj, int Dk, int Dl) : BaseTensor(Di, Dj, Dk, Dl) {};

        Tensor(int Di, int Dj, int Dk, int Dl, int D_max) : BaseTensor(Di, Dj, Dk, Dl, D_max) {};
    };

    class ImpureTensor : public BaseImpureTensor<Tensor> {
    public:
        ImpureTensor() : BaseImpureTensor<Tensor>() {};

        explicit ImpureTensor(int D) : BaseImpureTensor<Tensor>(D) {};

        ImpureTensor(int D, int D_max) : BaseImpureTensor<Tensor>(D, D_max) {};

        ImpureTensor(int Di, int Dj, int Dk, int Dl) : BaseImpureTensor<Tensor>(Di, Dj, Dk, Dl) {};

        ImpureTensor(int Di, int Dj, int Dk, int Dl, int D_max) : BaseImpureTensor<Tensor>(Di, Dj, Dk, Dl, D_max) {};

        ImpureTensor(int d, BaseImpureTensor<Tensor> &T) : BaseImpureTensor<Tensor>(d, T) {};

        explicit ImpureTensor(BaseImpureTensor<Tensor> &rhs) : BaseImpureTensor<Tensor>(rhs) {};
    };

    void initialize_spherical_harmonics(Tensor &T, const double &beta, const int &D_cut, const int &l_max);

    void initialize_gauss_quadrature(Tensor &T, const double &beta, const int &D_cut, const int &n_node);

    void initialize_spherical_harmonics_with_impure(Tensor &T, ImpureTensor &IMT, const double &beta, const int &D_cut, const int &l_max);

    void initialize_gauss_quadrature_with_impure(Tensor &T, ImpureTensor &IMT, const double &beta, const int &D_cut, const int &n_node);

    void contractionX(const int &D_cut, BaseTensor &leftT, BaseTensor &rightT, const double *U, const std::string &mergeT);

    void contractionY(const int &D_cut, BaseTensor &bottomT, BaseTensor &topT, const double *U, const std::string &mergeT);

    void SVD_X(const int &D_cut, BaseTensor &T, double *&U);

    void SVD_Y(const int &D_cut, BaseTensor &T, double *&U);

    namespace renormalization {
        double partition_alt(Tensor &T, long long int *orders, const int &n, const int &normalize_factor);

        void one_point_alt(Tensor &T, ImpureTensor &IMT, long long *orders, const int &n, const int &normalize_factor, double *res);

        void two_point_alt(Tensor &T, ImpureTensor &IMT, long long *orders, const int &n, const int &merge_point, const int &normalize_factor, double *res);

        void
        two_point_manual(Tensor &T, ImpureTensor &originIMT, ImpureTensor &IMT, long long *orders, const int &n, std::pair<int, int> &p, const int &normalize_factor, double *res);

        void
        mass(Tensor &T, ImpureTensor &IMT, long long *orders, const int &N, const int &n, const int &merge_point, const int &normalize_factor, double *res);

        /// deprecated
        void mass_alt(Tensor &T, ImpureTensor &IMT, long long *orders, const int &n, const int &merge_point, const int &normalize_factor, double *res);

        void
        mass_manual(Tensor &T, ImpureTensor &originIMT, ImpureTensor &IMT, long long *orders, const int &N, const int &n, int &distance, const int &normalize_factor, double *res);

        /**
         * compress alternatively until just before the impure tensors merge with each other
        **/
        void
        mass_v1(Tensor &T, ImpureTensor &IMT, long long *orders, const int &N, const int &n, const int &merge_point, int &x_count, int &y_count, const int &normalize_factor, double *res);
    }
}

#endif //O3_SIGMA_MODEL_HOTRG_HPP
