#include "../include/tensor.hpp"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)

BaseTensor::BaseTensor(int D) {
    this->Di = D;
    this->Dj = D;
    this->Dk = D;
    this->Dl = D;
    this->D_max = D;
    array = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) array[i] = 0;
}

BaseTensor::BaseTensor(int D, int D_max) {
    this->Di = D;
    this->Dj = D;
    this->Dk = D;
    this->Dl = D;
    this->D_max = D_max;
    array = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) array[i] = 0;
}

BaseTensor::BaseTensor(int Di, int Dj, int Dk, int Dl) {
    this->Di = Di;
    this->Dj = Dj;
    this->Dk = Dk;
    this->Dl = Dl;
    this->D_max = std::max(Di, std::max(Dj, std::max(Dk, Dl)));
    array = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) array[i] = 0;
}

BaseTensor::BaseTensor(int Di, int Dj, int Dk, int Dl, int D_max) {
    this->Di = Di;
    this->Dj = Dj;
    this->Dk = Dk;
    this->Dl = Dl;
    this->D_max = D_max;
    array = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) array[i] = 0;
}

BaseTensor::BaseTensor(BaseTensor &rhs) {
    this->Di = rhs.Di;
    this->Dj = rhs.Dj;
    this->Dk = rhs.Dk;
    this->Dl = rhs.Dl;
    this->D_max = rhs.D_max;
    array = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) array[i] = rhs.array[i];
}

BaseTensor::~BaseTensor() {
    Di = 0;
    Dj = 0;
    Dk = 0;
    Dl = 0;
    D_max = 0;
    delete[] array;
    array = nullptr;
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

int BaseTensor::GetD_max() const {
    return D_max;
}

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

void BaseTensor::SetDi(int Di_) {
    assert(Di_ <= D_max);
    this->Di = Di_;
}

void BaseTensor::SetDj(int Dj_) {
    assert(Dj_ <= D_max);
    this->Dj = Dj_;
}

void BaseTensor::SetDk(int Dk_) {
    assert(Dk_ <= D_max);
    this->Dk = Dk_;
}

void BaseTensor::SetDl(int Dl_) {
    assert(Dl_ <= D_max);
    this->Dl = Dl_;
}

BaseTensor &BaseTensor::operator=(const BaseTensor &rhs) {
    this->Di = rhs.Di;
    this->Dj = rhs.Dj;
    this->Dk = rhs.Dk;
    this->Dl = rhs.Dl;
    this->D_max = rhs.D_max;
    delete[] array;
    array = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) array[i] = rhs.array[i];
    order = rhs.order;
    return *this;
}

const double &BaseTensor::operator()(int i, int j, int k, int l) const {
    assert(0 <= i && i <= Di);
    assert(0 <= j && j <= Dj);
    assert(0 <= k && k <= Dk);
    assert(0 <= l && l <= Dl);
    return array[Dj * Dk * Dl * i + Dk * Dl * j + Dl * k + l];
}

double &BaseTensor::operator()(int i, int j, int k, int l) {
    assert(0 <= i && i <= Di);
    assert(0 <= j && j <= Dj);
    assert(0 <= k && k <= Dk);
    assert(0 <= l && l <= Dl);
    return array[Dj * Dk * Dl * i + Dk * Dl * j + Dl * k + l];
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

long long int BaseTensor::normalization(int c) {
    double _max = 0;
    this->forEach([&](int i, int j, int k, int l, const double *t) {
        const double absT = std::abs(*t);
        if (std::isnan(absT)) {
            std::cerr << "T(" << i << ',' << j << ',' << k << ',' << l << ") is nan";
            exit(1);
        }
        _max = std::max(_max, absT);
    });
    auto o = static_cast<int>(std::floor(std::log10(_max) / std::log10(c)));
    auto absO = std::abs(o);
    if (o > 0) {
        this->forEach([&](double *t) {
            REP(a, absO) *t /= c;
        });
    } else if (o < 0) {
        this->forEach([&](double *t) {
            REP(a, absO) *t *= c;
        });
    }
    return order = o;
}

double BaseTensor::trace() {
    double res = 0;
    REP(i, Di)REP(j, Dj) res += (*this)(i, j, i, j);
    return res;
}