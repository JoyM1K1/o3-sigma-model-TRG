#ifndef O3_SIGMA_MODEL_TRG_HPP
#define O3_SIGMA_MODEL_TRG_HPP

#include "tensor.hpp"
#include "impure_tensor.hpp"

#define MAX_IMT_NUM 6

namespace TRG {
    class Unitary_S {
    public:
        double *array{nullptr};
        int D_cut{0};

        Unitary_S() = default;

        explicit Unitary_S(int D_cut);

        ~Unitary_S();
    };

    class Tensor : public BaseTensor {
    public:
        std::pair<Unitary_S *, Unitary_S *> S{std::make_pair(nullptr, nullptr)};

        Tensor() : BaseTensor() {};

        explicit Tensor(int D_cut) : BaseTensor(D_cut) {};

        Tensor(int D, int D_max) : BaseTensor(D, D_max) {};

        Tensor(int Di, int Dj, int Dk, int Dl) : BaseTensor(Di, Dj, Dk, Dl) {};

        Tensor(int Di, int Dj, int Dk, int Dl, int D_max) : BaseTensor(Di, Dj, Dk, Dl, D_max) {};

        ~Tensor();

        Tensor &operator=(const Tensor &rhs);
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

    void SVD(Tensor &T, bool is_default, bool is_odd_times);

    void contraction(Tensor &T, Tensor &T1, Tensor &T2, Tensor &T3, Tensor &T4);

    void initialize_spherical_harmonics(Tensor &T1, Tensor &T2, const int &D, const int &D_cut, const double &beta, const int &l_max);

    void initialize_gauss_quadrature(Tensor &T1, Tensor &T2, const int &D, const int &D_cut, const double &beta, const int &n_node);

    void
    initialize_spherical_harmonics_with_impure(Tensor &T1, Tensor &T2, ImpureTensor (&IMTs)[MAX_IMT_NUM], const int &D, const int &D_cut, const double &beta, const int &l_max, const int &merge_point);

    void
    initialize_gauss_quadrature_with_impure(Tensor &T1, Tensor &T2, ImpureTensor (&IMTs)[MAX_IMT_NUM], const int &D, const int &D_cut, const double &beta, const int &n_node, const int &merge_point);

    void allocate_tensor(Tensor &T, const int &D, const int &D_cut);

    namespace renormalization {
        double partition(Tensor &T1, Tensor &T2, long long int *orders, const int &n, const int &normalize_factor);

        void
        two_point(Tensor &T1, Tensor &T2, ImpureTensor (&IMTs)[MAX_IMT_NUM], long long *orders, const int &N, const int &n, const int &merge_point, const int &normalize_factor);

        void trace(Tensor &T, ImpureTensor &IMT, const long long *orders, const int &normalize_factor, double *res);
    }
}

#endif //O3_SIGMA_MODEL_TRG_HPP
