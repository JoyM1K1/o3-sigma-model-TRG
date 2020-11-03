#ifndef O3_SIGMA_MODEL_TRG_HPP
#define O3_SIGMA_MODEL_TRG_HPP

#include <vector>
#include <queue>
#include "tensor.hpp"
#include "impure_tensor.hpp"

namespace TRG {
    class Unitary_S {
    public:
        double *tensor;
        unsigned long long int order{0};
        std::vector<int> orders;
        int D_cut{0};

        Unitary_S();

        Unitary_S(int D_cut);

        ~Unitary_S();

        void normalization(int c = 10);
    };

    class Tensor : public BaseTensor {
    public:
        std::pair<Unitary_S *, Unitary_S *> S{std::make_pair(nullptr, nullptr)};

        Tensor();

        Tensor(int D_cut);

        Tensor(int D, int D_max) : BaseTensor(D, D_max) {};

        Tensor(int Di, int Dj, int Dk, int Dl) : BaseTensor(Di, Dj, Dk, Dl) {};

        Tensor(int Di, int Dj, int Dk, int Dl, int D_max) : BaseTensor(Di, Dj, Dk, Dl, D_max) {};

        Tensor &operator=(const Tensor &rhs);

        void normalization(int c) override;
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

    void SVD(const int &D, const int &D_new, Tensor &T, bool isRightUp);

    void contraction(const int &D, const int &D_new, Tensor &T, Unitary_S *S1, Unitary_S *S2, Unitary_S *S3, Unitary_S *S4);
}

#endif //O3_SIGMA_MODEL_TRG_HPP
