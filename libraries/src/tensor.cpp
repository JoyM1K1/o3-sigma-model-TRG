//
// Created by Joy on 2020/06/16.
//

#include "../include/tensor.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define LINF 1e300

Tensor::Tensor() {
    this->Dx = 0;
    this->Dy = 0;
    this->Dx_max = 0;
    this->Dy_max = 0;
    M = new double[1];
}

Tensor::Tensor(int D) {
    this->Dx = D;
    this->Dy = D;
    this->Dx_max = D;
    this->Dy_max = D;
    M = new double[D * D * D * D];
    for (int i = 0; i < D * D * D * D; ++i) M[i] = 0;
}

Tensor::Tensor(int Dx, int Dy) {
    this->Dx = Dx;
    this->Dy = Dy;
    this->Dx_max = Dx;
    this->Dy_max = Dy;
    M = new double[Dx * Dx * Dy * Dy];
    for (int i = 0; i < Dx_max * Dx_max * Dy_max * Dy_max; ++i) M[i] = 0;
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

int Tensor::normalization(Tensor &T) {
    const int Dx = T.GetDx();
    const int Dy = T.GetDy();
    double _min = LINF;
    double _max = 0;
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    const double t = T(i, j, k, l);
                    if (std::abs(t) > 0) {
                        _min = std::min(_min, std::abs(t));
                        _max = std::max(_max, std::abs(t));
                    }
                }
    auto o = static_cast<int>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    return o;
}