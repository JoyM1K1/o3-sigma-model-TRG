#ifndef O3_SIGMA_MODEL_TENSOR_HPP
#define O3_SIGMA_MODEL_TENSOR_HPP

#include <functional>

class Tensor {
private:
    int Di, Dj, Dk, Dl, D_max;
    int N;
    double *M;
    int *order;
public:
    Tensor();

    Tensor(int D, int N);

    Tensor(int D, int D_max, int N);

    Tensor(int Di, int Dj, int Dk, int Dl, int N);

    Tensor(int Di, int Dj, int Dk, int Dl, int D_max, int N);

    Tensor(Tensor &rhs);

    ~Tensor();

    int GetDx() const;

    int GetDy() const;

    int GetDi() const;

    int GetDj() const;

    int GetDk() const;

    int GetDl() const;

    double *GetMatrix() const;

    int *GetOrder() const;

    void UpdateDx(int Dx);

    void UpdateDy(int Dy);

    Tensor &operator=(const Tensor &rhs);

    const double &operator()(int i, int j, int k, int l) const;

    double &operator()(int i, int j, int k, int l);

    void forEach(const std::function<void(double)> &f);

    void forEach(const std::function<void(int, int, int, int, double)> &f);

    void normalization(int n);
};

#endif //O3_SIGMA_MODEL_TENSOR_HPP
