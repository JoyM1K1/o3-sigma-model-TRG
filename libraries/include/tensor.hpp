#ifndef O3_SIGMA_MODEL_TENSOR_HPP
#define O3_SIGMA_MODEL_TENSOR_HPP

#include <functional>

class BaseTensor {
private:
    int Di, Dj, Dk, Dl, D_max;
    double *M;
public:
    long long int order{0};

    BaseTensor();

    explicit BaseTensor(int D);

    BaseTensor(int D, int D_max);

    BaseTensor(int Di, int Dj, int Dk, int Dl);

    BaseTensor(int Di, int Dj, int Dk, int Dl, int D_max);

    BaseTensor(BaseTensor &rhs);

    ~BaseTensor();

    int GetDx() const;

    int GetDy() const;

    int GetDi() const;

    int GetDj() const;

    int GetDk() const;

    int GetDl() const;

    int GetD_max() const;

    double *GetMatrix() const;

    void UpdateDx(int Dx);

    void UpdateDy(int Dy);

    void SetDi(int Di);

    void SetDj(int Dj);

    void SetDk(int Dk);

    void SetDl(int Dl);

    BaseTensor &operator=(const BaseTensor &rhs);

    const double &operator()(int i, int j, int k, int l) const;

    double &operator()(int i, int j, int k, int l);

    void forEach(const std::function<void(double *)> &f);

    void forEach(const std::function<void(int, int, int, int, double *)> &f);

    long long int normalization(int c);

    double trace();
};

#endif //O3_SIGMA_MODEL_TENSOR_HPP
