#ifndef O3_SIGMA_MODEL_TENSOR_HPP
#define O3_SIGMA_MODEL_TENSOR_HPP

#include <functional>
#include <vector>

class BaseTensor {
private:
    int Di, Dj, Dk, Dl, D_max;
    double *M;
public:
    unsigned long long int order{0};
    std::vector<int> orders;

    BaseTensor();

    BaseTensor(int D);

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

    double *GetMatrix() const;

//    int *GetOrder() const;

    void UpdateDx(int Dx);

    void UpdateDy(int Dy);

    BaseTensor &operator=(const BaseTensor &rhs);

    const double &operator()(int i, int j, int k, int l) const;

    double &operator()(int i, int j, int k, int l);

    void forEach(const std::function<void(double *)> &f);

    void forEach(const std::function<void(int, int, int, int, double *)> &f);

    virtual void normalization(int c);
};

#endif //O3_SIGMA_MODEL_TENSOR_HPP
