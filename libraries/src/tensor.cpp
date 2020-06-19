//
// Created by Joy on 2020/06/16.
//

#include "../include/tensor.hpp"
#include <cassert>

Tensor::Tensor() {
    this->Dx = 0;
    this->Dy = 0;
    this->Dx_max = 0;
    this->Dy_max = 0;
    M = new double[1];
}

Tensor::Tensor(int Dx, int Dy) {
    this->Dx = Dx;
    this->Dy = Dy;
    this->Dx_max = Dx;
    this->Dy_max = Dy;
    M = new double[Dx * Dx * Dy * Dy];
    for (int i = 0; i < Dx_max * Dx_max * Dy_max * Dy_max; ++i) {
        M[i] = 0;
    }
}

Tensor::Tensor(int Dx, int Dy, int Dx_max, int Dy_max) {
    this->Dx = Dx;
    this->Dy = Dy;
    this->Dx_max = Dx_max;
    this->Dy_max = Dy_max;
    M = new double[Dx_max * Dx_max * Dy_max * Dy_max];
    for (int i = 0; i < Dx_max * Dx_max * Dy_max * Dy_max; ++i) M[i] = 0;
}

Tensor::Tensor(Tensor &rhs) {
    this->Dx = rhs.Dx;
    this->Dy = rhs.Dy;
    this->Dx_max = rhs.Dx_max;
    this->Dy_max = rhs.Dy_max;
    M = new double[Dx_max * Dx_max * Dy_max * Dy_max];
    double *M_ = rhs.GetMatrix();
    for (int i = 0; i < Dx * Dx * Dy * Dy; ++i) M[i] = M_[i];
}

Tensor::~Tensor() {
    Dx = 0;
    Dy = 0;
    Dx_max = 0;
    Dy_max = 0;
    delete [] M;
    M = nullptr;
}

int Tensor::GetDx() const {
    return Dx;
}

int Tensor::GetDy() const {
    return Dy;
}

double * Tensor::GetMatrix() const {
    return M;
}

void Tensor::UpdateDx(int Dx_) {
    assert(Dx_ <= Dx_max);
    this->Dx = Dx_;
}

void Tensor::UpdateDy(int Dy_) {
    assert(Dy_ <= Dy_max);
    this->Dy = Dy_;
}

Tensor & Tensor::operator=(const Tensor &rhs) {
    Dx = rhs.Dx;
    Dy = rhs.Dy;
    Dx_max = rhs.Dx_max;
    Dy_max = rhs.Dy_max;
    delete [] M;
    M = new double[Dx_max * Dx_max * Dy_max * Dy_max];
    double *M_ = rhs.GetMatrix();
    for (int i = 0; i < Dx * Dx * Dy * Dy; ++i) M[i] = M_[i];
    return *this;
}

const double & Tensor::operator()(int i, int j, int k, int l) const {
    assert(0 <= i && i <= Dx);
    assert(0 <= j && j <= Dy);
    assert(0 <= k && k <= Dx);
    assert(0 <= l && l <= Dy);
    return M[Dy * Dx * Dy * i + Dx * Dy * j + Dy * k + l];
}

double & Tensor::operator()(int i, int j, int k, int l) {
    assert(0 <= i && i <= Dx);
    assert(0 <= j && j <= Dy);
    assert(0 <= k && k <= Dx);
    assert(0 <= l && l <= Dy);
    return M[Dy * Dx * Dy * i + Dx * Dy * j + Dy * k + l];
}