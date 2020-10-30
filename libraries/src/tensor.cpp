#include "../include/tensor.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

//#define LINF 1e300

BaseTensor::BaseTensor() {
    Di = 0;
    Dj = 0;
    Dk = 0;
    Dl = 0;
    D_max = 0;
    M = new double[1];
}

BaseTensor::BaseTensor(int D) {
    this->Di = D;
    this->Dj = D;
    this->Dk = D;
    this->Dl = D;
    this->D_max = D;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
}

BaseTensor::BaseTensor(int D, int D_max) {
    this->Di = D;
    this->Dj = D;
    this->Dk = D;
    this->Dl = D;
    this->D_max = D_max;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
}

BaseTensor::BaseTensor(int Di, int Dj, int Dk, int Dl) {
    this->Di = Di;
    this->Dj = Dj;
    this->Dk = Dk;
    this->Dl = Dl;
    this->D_max = std::max(Di, std::max(Dj, std::max(Dk, Dl)));
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
}

BaseTensor::BaseTensor(int Di, int Dj, int Dk, int Dl, int D_max) {
    this->Di = Di;
    this->Dj = Dj;
    this->Dk = Dk;
    this->Dl = Dl;
    this->D_max = D_max;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
}

BaseTensor::BaseTensor(BaseTensor &rhs) {
    this->Di = rhs.Di;
    this->Dj = rhs.Dj;
    this->Dk = rhs.Dk;
    this->Dl = rhs.Dl;
    this->D_max = rhs.D_max;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = rhs.M[i];
}

BaseTensor::~BaseTensor() {
    Di = 0;
    Dj = 0;
    Dk = 0;
    Dl = 0;
    D_max = 0;
    delete[] M;
    M = nullptr;
}

int BaseTensor::GetDx() const {
    return Di; // same as Dk
}

int BaseTensor::GetDy() const {
    return Dj; // same as Dl
}

int BaseTensor::GetDi() const {
    return Di;
}

int BaseTensor::GetDj() const {
    return Dj;
}

int BaseTensor::GetDk() const {
    return Dk;
}

int BaseTensor::GetDl() const {
    return Dl;
}

double *BaseTensor::GetMatrix() const {
    return M;
}

//int *BaseTensor::GetOrder() const {
//    return orders;
//}

void BaseTensor::UpdateDx(int Dx) {
    assert(Dx <= D_max);
    this->Di = Dx;
    this->Dk = Dx;
}

void BaseTensor::UpdateDy(int Dy) {
    assert(Dy <= D_max);
    this->Dj = Dy;
    this->Dl = Dy;
}

BaseTensor &BaseTensor::operator=(const BaseTensor &rhs) {
    this->Di = rhs.Di;
    this->Dj = rhs.Dj;
    this->Dk = rhs.Dk;
    this->Dl = rhs.Dl;
    this->D_max = rhs.D_max;
    delete[] M;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = rhs.M[i];
//    orders.clear();
//    for (auto o : rhs.orders) {
//        orders.push_back(o);
//    }
    order = rhs.order;
    return *this;
}

const double &BaseTensor::operator()(int i, int j, int k, int l) const {
    assert(0 <= i && i <= Di);
    assert(0 <= j && j <= Dj);
    assert(0 <= k && k <= Dk);
    assert(0 <= l && l <= Dl);
    return M[Dj * Dk * Dl * i + Dk * Dl * j + Dl * k + l];
}

double &BaseTensor::operator()(int i, int j, int k, int l) {
    assert(0 <= i && i <= Di);
    assert(0 <= j && j <= Dj);
    assert(0 <= k && k <= Dk);
    assert(0 <= l && l <= Dl);
    return M[Dj * Dk * Dl * i + Dk * Dl * j + Dl * k + l];
}

void BaseTensor::forEach(const std::function<void(double *)> &f) {
    REP(i, Di)REP(j, Dj)REP(k, Dk)REP(l, Dl) {
                    f(&(*this)(i, j, k, l));
                }
}

void BaseTensor::forEach(const std::function<void(int, int, int, int, double *)> &f) {
    REP(i, Di)REP(j, Dj)REP(k, Dk)REP(l, Dl) {
                    f(i, j, k, l, &(*this)(i, j, k, l));
                }
}

void BaseTensor::normalization(int c) {
//    double _min = LINF;
    double _max = 0;
    REP(i, Di)REP(j, Dj)REP(k, Dk)REP(l, Dl) {
                    const double t = std::abs((*this)(i, j, k, l));
                    if (std::isnan(t)) {
                        std::cerr << "T(" << i << ',' << j << ',' << k << ',' << l << ") is nan";
                        exit(1);
                    }
                    if (t > 0) {
//                        _min = std::min(_min, t);
                        _max = std::max(_max, t);
                    }
                }
//    auto o = static_cast<int>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    auto o = static_cast<int>(std::floor(std::log10(_max) / std::log10(c)));
    REP(i, Di)REP(j, Dj)REP(k, Dk)REP(l, Dl) {
                    if (o > 0) {
                        REP(t, std::abs(o)) (*this)(i, j, k, l) /= c;
                    } else {
                        REP(t, std::abs(o)) (*this)(i, j, k, l) *= c;
                    }
                }
    order += o;
}