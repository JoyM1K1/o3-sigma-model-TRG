//
// Created by Joy on 2020/06/16.
//

#include "../include/tensor.hpp"
#include <cassert>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define CGFileName "clebsch_gordan.txt"

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

void Tensor::initSphericalHarmonics(const double &K, const int &l_max, Tensor &T, std::map<CG, frac> &map, std::ofstream &CGFile) {
    auto A = new double[l_max];
    REP(i, l_max) {
        A[i] = std::cyl_bessel_i(i + 0.5, K) * (i * 2 + 1);
    }

    REP4(i, j, k, l, l_max) {
        for (int im = 0; im <= 2 * i; ++im)
            for (int jm = 0; jm <= 2 * j; ++jm)
                for (int km = 0; km <= 2 * k; ++km)
                    for (int lm = 0; lm <= 2 * l; ++lm) {
                        double sum = 0;
                        for (int L = std::abs(i - j); L <= i + j; ++L)
                            for (int M = -L; M <= L; ++M) {
                                if (L < std::abs(k - l) || k + l < L || im - i + jm - j != M || km - k + lm - l != M)
                                    continue; // CG係数としてありえないものは0なので飛ばす
                                frac c(1);
                                c *= CG::getCoeff(frac(i), frac(j), frac(im - i), frac(jm - j), frac(L),
                                                  frac(M), map, CGFile);
                                c *= CG::getCoeff(frac(i), frac(j), frac(0), frac(0), frac(L), frac(0), map, CGFile);
                                c *= CG::getCoeff(frac(k), frac(l), frac(km - k), frac(lm - l), frac(L),
                                                  frac(M), map, CGFile);
                                c *= CG::getCoeff(frac(k), frac(l), frac(0), frac(0), frac(L), frac(0), map, CGFile);
                                c /= frac(2 * L + 1).sign() * (2 * L + 1) * (2 * L + 1);
                                sum += c.sign().toDouble() * std::sqrt(frac::abs(c).toDouble());
                            }
                        T(i * i + im, j * j + jm, k * k + km, l * l + lm) = std::sqrt(A[i] * A[j] * A[k] * A[l]) * sum;
                    }
    }
    delete [] A;
}