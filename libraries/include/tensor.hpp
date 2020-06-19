//
// Created by Joy on 2020/06/16.
//

#ifndef O3_SIGMA_MODEL_TENSOR_HPP
#define O3_SIGMA_MODEL_TENSOR_HPP

class Tensor {
private:
    int Dx, Dy, Dx_max, Dy_max;
    double *M;
public:
    Tensor();
    Tensor(int Dx, int Dy);
    Tensor(int Dx, int Dy, int Dx_max, int Dy_max);
    Tensor(Tensor &rhs);
    ~Tensor();

    int GetDx() const;
    int GetDy() const;
    double * GetMatrix() const;

    void UpdateDx(int Dx);
    void UpdateDy(int Dy);

    Tensor & operator=(const Tensor & rhs);
    const double & operator()(int i, int j, int k, int l) const;
    double & operator()(int i, int j, int k, int l);
};

#endif //O3_SIGMA_MODEL_TENSOR_HPP
