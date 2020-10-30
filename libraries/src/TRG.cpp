#include "../include/TRG.hpp"
#include <mkl.h>
#include <iostream>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void TRG::SVD(const int &D, const int &D_new, Tensor &T, bool isRightUp) {
    auto sigma = new double[D * D];
    auto U_ = new double[D * D * D * D], VT_ = new double[D * D * D * D];
    auto superb = new double[D * D - 1];
    if (isRightUp) { // (ij)(kl)
        MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, T.GetMatrix(), D * D, sigma, U_, D * D, VT_, D * D, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        REP(k, D_new) {
            double s = std::sqrt(sigma[k]);
            REP(i, D)REP(j, D) {
                    T.S.first->tensor[D * D * k + D * j + i] = U_[D * D * D * i + D * D * j + k] * s; // S1
                    T.S.second->tensor[D * D * k + D * j + i] = VT_[D * D * k + D * i + j] * s; // S3
                }
        }
    } else { // (jk)(li)
        BaseTensor M(D);
        T.forEach([&](int i, int j, int k, int l, const double *t) {
            M(j, k, l, i) = *t; // M(jk)(li)
        });
        MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, M.GetMatrix(), D * D, sigma, U_, D * D, VT_, D * D, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        REP(k, D_new) {
            double s = std::sqrt(sigma[k]);
            REP(i, D)REP(j, D) {
                    T.S.first->tensor[D_new * D * j + D_new * i + k] = U_[D * D * D * i + D * D * j + k] * s;
                    T.S.second->tensor[D_new * D * j + D_new * i + k] = VT_[D * D * k + D * i + j] * s;
                }
        }
    }
    delete[] U_;
    delete[] VT_;
    delete[] sigma;
    delete[] superb;
    /* order */
//    T.S.first->orders.clear();
//    for (auto order : T.orders) {
//        T.S.first->orders.push_back(order);
//    }
//    T.orders.clear();
    T.S.first->order = T.order;
}

void TRG::contraction(const int &D, const int &D_new, Tensor &T, Unitary_S *S1, Unitary_S *S2, Unitary_S *S3, Unitary_S *S4) {
    auto top = new double[D_new * D_new * D * D], bottom = new double[D_new * D_new * D * D];
    auto X = new double[D_new * D_new * D * D];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D_new * D, D_new * D, D, 1, S1->tensor,
            D, S2->tensor, D_new * D, 0, X, D_new * D);
    REP(a, D)REP(b, D)REP(i, D_new)REP(j, D_new) {
                    bottom[D_new * D_new * D * a + D_new * D_new * b + D_new * i + j] = X[D_new * D * D * i + D_new * D * b + D_new * a + j];
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D_new * D, D_new * D, D, 1, S3->tensor,
            D, S4->tensor, D_new * D, 0, X, D_new * D);
    REP(a, D)REP(b, D)REP(i, D_new)REP(j, D_new) {
                    top[D * D * D_new * i + D * D * j + D * a + b] = X[D_new * D * D * i + D_new * D * a + D_new * b + j];
                }
    delete[] X;
    T.UpdateDx(D_new);
    T.UpdateDy(D_new);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            D_new * D_new, D_new * D_new, D * D, 1, top, D * D, bottom, D_new * D_new, 0, T.GetMatrix(), D_new * D_new);
    delete[] top;
    delete[] bottom;
    /* order */
//    T.orders.clear();
//    for (auto order : S1->orders) {
//        T.orders.push_back(order);
//    }
//    for (auto order : S2->orders) {
//        T.orders.push_back(order);
//    }
//    for (auto order : S3->orders) {
//        T.orders.push_back(order);
//    }
//    for (auto order : S4->orders) {
//        T.orders.push_back(order);
//    }
    T.order = S1->order + S2->order + S3->order + S4->order;
}

TRG::Unitary_S::Unitary_S() {
    tensor = new double[1];
}

TRG::Unitary_S::Unitary_S(int D_cut) {
    tensor = new double[D_cut * D_cut * D_cut];
    this->D_cut = D_cut;
    REP(i, D_cut * D_cut * D_cut) tensor[i] = 0;
}

TRG::Unitary_S::~Unitary_S() {
    delete[] tensor;
}

void TRG::Unitary_S::normalization(int c) {
    double _max = 0;
    REP(i, D_cut * D_cut * D_cut) {
        const double t = std::abs(tensor[i]);
        if (t > 0) {
            _max = std::max(_max, t);
        }
    }
    auto o = static_cast<int>(std::floor(std::log10(_max) / std::log10(c)));
    REP(i, D_cut * D_cut * D_cut) {
        if (o > 0) {
            REP(t, std::abs(o)) tensor[i] /= c;
        } else {
            REP(t, std::abs(o)) tensor[i] *= c;
        }
    }
//    orders.push(o);
}

TRG::Tensor::Tensor() : BaseTensor() {
//    S = std::make_pair(new Unitary_S(), new Unitary_S());
}

TRG::Tensor::Tensor(int D_cut) : BaseTensor(D_cut) {
//    S = std::make_pair(new Unitary_S(D_cut), new Unitary_S(D_cut));
}

TRG::Tensor &TRG::Tensor::operator=(const Tensor &rhs) {
    BaseTensor::operator=(rhs);
    return *this;
}