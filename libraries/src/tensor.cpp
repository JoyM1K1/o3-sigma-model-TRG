#include "../include/tensor.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

//#define LINF 1e300

Tensor::Tensor() {
    Di = 0;
    Dj = 0;
    Dk = 0;
    Dl = 0;
    D_max = 0;
    N = 0;
    M = new double[1];
    order = new int[1];
}

Tensor::Tensor(int D, int N) {
    assert(N > 0);
    this->Di = D;
    this->Dj = D;
    this->Dk = D;
    this->Dl = D;
    this->D_max = D;
    this->N = N;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
    order = new int[N];
    for (int i = 0; i < N; ++i) order[i] = 0;
}

Tensor::Tensor(int D, int D_max, int N) {
    assert(N > 0);
    this->Di = D;
    this->Dj = D;
    this->Dk = D;
    this->Dl = D;
    this->D_max = D_max;
    this->N = N;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
    order = new int[N];
    for (int i = 0; i < N; ++i) order[i] = 0;
}

Tensor::Tensor(int Di, int Dj, int Dk, int Dl, int N) {
    assert(N > 0);
    this->Di = Di;
    this->Dj = Dj;
    this->Dk = Dk;
    this->Dl = Dl;
    this->D_max = std::max(Di, std::max(Dj, std::max(Dk, Dl)));
    this->N = N;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
    order = new int[N];
    for (int i = 0; i < N; ++i) order[i] = 0;
}

Tensor::Tensor(int Di, int Dj, int Dk, int Dl, int D_max, int N) {
    assert(N > 0);
    this->Di = Di;
    this->Dj = Dj;
    this->Dk = Dk;
    this->Dl = Dl;
    this->D_max = D_max;
    this->N = N;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = 0;
    order = new int[N];
    for (int i = 0; i < N; ++i) order[i] = 0;
}

Tensor::Tensor(Tensor &rhs) {
    this->Di = rhs.Di;
    this->Dj = rhs.Dj;
    this->Dk = rhs.Dk;
    this->Dl = rhs.Dl;
    this->D_max = rhs.D_max;
    this->N = rhs.N;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = rhs.M[i];
    order = new int[rhs.N];
    for (int i = 0; i < this->N; ++i) order[i] = rhs.order[i];
}

Tensor::~Tensor() {
    Di = 0;
    Dj = 0;
    Dk = 0;
    Dl = 0;
    D_max = 0;
    N = 0;
    delete[] M;
    M = nullptr;
    delete[] order;
    order = nullptr;
}

int Tensor::GetDx() const {
    return Di; // same as Dk
}

int Tensor::GetDy() const {
    return Dj; // same as Dl
}

int Tensor::GetDi() const {
    return Di;
}

int Tensor::GetDj() const {
    return Dj;
}

int Tensor::GetDk() const {
    return Dk;
}

int Tensor::GetDl() const {
    return Dl;
}

double *Tensor::GetMatrix() const {
    return M;
}

int *Tensor::GetOrder() const {
    return order;
}

void Tensor::UpdateDx(int Dx) {
    assert(Dx <= D_max);
    this->Di = Dx;
    this->Dk = Dx;
}

void Tensor::UpdateDy(int Dy) {
    assert(Dy <= D_max);
    this->Dj = Dy;
    this->Dl = Dy;
}

Tensor &Tensor::operator=(const Tensor &rhs) {
    this->Di = rhs.Di;
    this->Dj = rhs.Dj;
    this->Dk = rhs.Dk;
    this->Dl = rhs.Dl;
    this->D_max = rhs.D_max;
    this->N = rhs.N;
    delete[] M;
    M = new double[D_max * D_max * D_max * D_max];
    for (int i = 0; i < D_max * D_max * D_max * D_max; ++i) M[i] = rhs.M[i];
    delete[] order;
    order = new int[rhs.N];
    for (int i = 0; i < rhs.N; ++i) order[i] = rhs.order[i];
    return *this;
}

const double &Tensor::operator()(int i, int j, int k, int l) const {
    assert(0 <= i && i <= Di);
    assert(0 <= j && j <= Dj);
    assert(0 <= k && k <= Dk);
    assert(0 <= l && l <= Dl);
    return M[Dj * Dk * Dl * i + Dk * Dl * j + Dl * k + l];
}

double &Tensor::operator()(int i, int j, int k, int l) {
    assert(0 <= i && i <= Di);
    assert(0 <= j && j <= Dj);
    assert(0 <= k && k <= Dk);
    assert(0 <= l && l <= Dl);
    return M[Dj * Dk * Dl * i + Dk * Dl * j + Dl * k + l];
}

void Tensor::forEach(const std::function<void(double*)> &f) {
    REP(i, Di)REP(j, Dj)REP(k, Dk)REP(l, Dl) {
                    f(&(*this)(i, j, k, l));
                }
}

void Tensor::forEach(const std::function<void(int, int, int, int, double*)> &f) {
    REP(i, Di)REP(j, Dj)REP(k, Dk)REP(l, Dl) {
                    f(i, j, k, l, &(*this)(i, j, k, l));
                }
}

void Tensor::normalization(int n) {
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
    auto o = static_cast<int>(std::floor(std::log10(_max)));
    REP(i, Di)REP(j, Dj)REP(k, Dk)REP(l, Dl) {
                    if (o > 0) {
                        REP(t, std::abs(o)) (*this)(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) (*this)(i, j, k, l) *= 10;
                    }
                }
    order[n] = o;
}