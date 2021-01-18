#include "../include/TRG.hpp"
#include "../include/time_counter.hpp"
#include "../include/spherical_harmonics.hpp"
#include "../include/gauss_quadrature.hpp"
#include <mkl.h>
#include <iostream>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)

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
//    T.S.first->order = T.order;
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
//    T.order = S1->order + S2->order + S3->order + S4->order;
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

long long int TRG::Tensor::normalization(int c) {
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

void TRG::allocate_tensor(Tensor &T, const int &D, const int &D_cut) {
    T = TRG::Tensor(D, D_cut);
    T.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
}

void TRG::initialize_spherical_harmonics(Tensor &T1, Tensor &T2, const int &D, const int &D_cut, const double &beta, const int &l_max) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    allocate_tensor(T1, D, D_cut); /* (ij)(kl) -> S1 S3 */
    allocate_tensor(T2, D, D_cut); /* (jk)(li) -> S2 S4 */
    SphericalHarmonics::init_tensor(beta, l_max, T1);
    time.end();
    cout << "in " << time.duration_cast_to_string() << "\n" << std::flush;
}

void TRG::initialize_gauss_quadrature(Tensor &T1, Tensor &T2, const int &D, const int &D_cut, const double &beta, const int &n_node) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    allocate_tensor(T1, D, D_cut); /* (ij)(kl) -> S1 S3 */
    allocate_tensor(T2, D, D_cut); /* (jk)(li) -> S2 S4 */
    GaussQuadrature::init_tensor(beta, n_node, D_cut, T1);
    time.end();
    cout << "in " << time.duration_cast_to_string() << "\n" << std::flush;
}

void TRG::initialize_spherical_harmonics_with_impure(Tensor &T1, Tensor &T2, ImpureTensor *IMTs, const int &D, const int &D_cut, const double &beta, const int &l_max, const int &merge_point) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    allocate_tensor(T1, D, D_cut); /* (ij)(kl) -> S1 S3 */
    allocate_tensor(T2, D, D_cut); /* (jk)(li) -> S2 S4 */
    REP(i, MAX_IMT_NUM) IMTs[i] = ImpureTensor(D, D_cut);
    SphericalHarmonics::init_tensor_with_impure(beta, l_max, T1, IMTs[0]);
    IMTs[0].isImpure = true;
    if (merge_point == 1) {
        IMTs[1] = IMTs[0];
        IMTs[1].isImpure = true;
    }
    time.end();
    cout << "in " << time.duration_cast_to_string() << "\n" << std::flush;
}

void TRG::initialize_gauss_quadrature_with_impure(Tensor &T1, Tensor &T2, ImpureTensor *IMTs, const int &D, const int &D_cut, const double &beta, const int &n_node, const int &merge_point) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    allocate_tensor(T1, D, D_cut); /* (ij)(kl) -> S1 S3 */
    allocate_tensor(T2, D, D_cut); /* (jk)(li) -> S2 S4 */
    REP(i, MAX_IMT_NUM) IMTs[i] = ImpureTensor(D, D_cut);
    GaussQuadrature::init_tensor_with_impure(beta, n_node, D_cut, D, T1, IMTs[0]);
    IMTs[0].isImpure = true;
    if (merge_point == 1) {
        IMTs[1] = IMTs[0];
        IMTs[1].isImpure = true;
    }
    REP(i, MAX_IMT_NUM) {
        for (auto & tensor : IMTs[i].tensors) {
            if (i % 2) {
                tensor.S = T1.S;
            } else {
                tensor.S = T2.S;
            }
        }
    }
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;
}

double TRG::renormalization::partition(Tensor &T1, Tensor &T2, long long int *orders, const int &n, const int &normalize_factor) {
    /* normalization */
    orders[n - 1] = T1.normalization(normalize_factor);

    /* SVD */
    T2 = T1;
    const int D = T1.GetDx();
    const int D_cut = T1.GetD_max();
    const int D_new = std::min(D * D, D_cut);
    TRG::SVD(D, D_new, T1, true);
    TRG::SVD(D, D_new, T2, false);

    /* contraction */
    TRG::contraction(D_cut, D_cut, T1, T1.S.first, T2.S.first, T1.S.second, T2.S.second);

    /* trace */
    double Tr = T1.trace();
    Tr = std::log(Tr);
    REP(i, n) Tr /= 2; // 体積で割る
    REP(i, n) {
        double tmp = orders[i] * std::log(normalize_factor);
        REP(j, i) tmp /= 2;
        Tr += tmp;
    }
    return Tr;
}

void TRG::index_rotation(Tensor &T, Tensor &tmp) {
    T.UpdateDx(tmp.GetDx());
    T.UpdateDy(tmp.GetDy());
    tmp.forEach([&](int i, int j, int k, int l, const double *t) {
        T(l, i, j, k) = *t;
    });
    T.order = tmp.order;
}

void TRG::renormalization::two_point(Tensor &T1, Tensor &T2, ImpureTensor *IMTs, long long *orders, const int &N, const int &n, const int &merge_point, const int &normalize_factor) {
    const int D = T1.GetDx();
    const int D_cut = T1.GetD_max();
    const int D_new = std::min(D * D, D_cut);
    const int count = (n + 1) / 2;

    /* SVD pure tensor T */
    T2 = T1;
    TRG::SVD(D, D_new, T1, true);
    TRG::SVD(D, D_new, T2, false);

    /* SVD impure tensor IMTs */
    REP(i, MAX_IMT_NUM) {
        if (IMTs[i].isImpure) {
//                string allocate;
//                string tmp;
            for (auto & tensor : IMTs[i].tensors) {
                if (tensor.S.first == T1.S.first || tensor.S.first == T2.S.first/* same as tensor.S.second == T1.S.second || tensor.S.second == T2.S.second */) {
//                        allocate = " allocate";
                    tensor.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
                }
                if (merge_point == 1 || count <= merge_point - 1) {
//                        tmp = " " + std::to_string(i) + (i % 2 == 1 ? "rightUp " : "leftUp ");
                    TRG::SVD(D, D_new, tensor, i % 2 == 1);
                } else if (n <= N - 2) {
                    if (n % 2) {
//                            tmp = " " + std::to_string(i) + (i % 2 == 1 ? "rightUp " : "leftUp ");
                        TRG::SVD(D, D_new, tensor, i % 2 == 1);
                    } else {
//                            tmp = " " + std::to_string(i) + (i % 4 == 0 ? "rightUp " : "leftUp ");
                        TRG::SVD(D, D_new, tensor, i % 4 == 0);
                    }
                } else {
//                        tmp = " " + std::to_string(i) + (i % 2 == 1 ? "rightUp " : "leftUp ");
                    TRG::SVD(D, D_new, tensor, i % 2 == 1);
                }
            }
//                cout << allocate << tmp;
        } else {
            for (auto & tensor : IMTs[i].tensors) {
                if (merge_point == 1 || count < merge_point - 1) {
                    if (i % 2 == 1) {
                        tensor.S = T1.S;
                    } else {
                        tensor.S = T2.S;
                    }
                } else if (n <= N - 2) {
                    if (n % 2) {
                        if (i % 2 == 1) {
                            tensor.S = T1.S;
                        } else {
                            tensor.S = T2.S;
                        }
                    } else {
                        if (i % 4 == 0) {
                            tensor.S = T1.S;
                        } else {
                            tensor.S = T2.S;
                        }
                    }
                } else {
                    if (i % 2 == 1) {
                        tensor.S = T1.S;
                    } else {
                        tensor.S = T2.S;
                    }
                }
            }
        }
    }

    /* contraction */
    if (n % 2) {
        TRG::contraction(D, D_new, T1, T1.S.first, T2.S.first, T1.S.second, T2.S.second);
    } else {
        TRG::Tensor tmp(D);
        TRG::contraction(D, D_new, tmp, T1.S.first, T2.S.first, T1.S.second, T2.S.second);
        index_rotation(T1, tmp);
    }
    if ((merge_point == 1 && n <= N - 2) || count < merge_point - 1) {
        if (n % 2) {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::contraction(D, D_new, IMTs[0].tensors[i], T1.S.first, T2.S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
            }
            if (IMTs[1].isImpure) {
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[1].tensors[i], IMTs[1].tensors[i].S.first, T2.S.first, T1.S.second, IMTs[2].tensors[i].S.second);
                }
                if (IMTs[3].isImpure) {
                    /* 2 */
                    REP(i, DIMENSION) {
                        TRG::contraction(D, D_new, IMTs[2].tensors[i], IMTs[3].tensors[i].S.first, IMTs[2].tensors[i].S.first, T1.S.second, T2.S.second);
                    }
                    IMTs[2].isImpure = true;
                }
            }
            /* 3 */
            REP(i, DIMENSION) {
                TRG::contraction(D, D_new, IMTs[3].tensors[i], T1.S.first, IMTs[0].tensors[i].S.first, IMTs[3].tensors[i].S.second, T2.S.second);
            }
            IMTs[3].isImpure = true;
        } else {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[0].tensors[i].S.first, IMTs[3].tensors[i].S.second, T2.S.second);
                index_rotation(IMTs[0].tensors[i], tmp);
            }
            /* 1 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, T1.S.first, T2.S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
                index_rotation(IMTs[1].tensors[i], tmp);
            }
            if (IMTs[1].isImpure) {
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, IMTs[1].tensors[i].S.first, T2.S.first, T1.S.second, IMTs[2].tensors[i].S.second);
                    index_rotation(IMTs[2].tensors[i], tmp);
                }
                IMTs[2].isImpure = true;
            }
            /* 3 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, IMTs[3].tensors[i].S.first, IMTs[2].tensors[i].S.first, T1.S.second, T2.S.second);
                index_rotation(IMTs[3].tensors[i], tmp);
            }
            IMTs[1].isImpure = true;
        }
    } else if (count == merge_point - 1) {
        if (merge_point == 2) {
            if (n % 2) {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[0].tensors[i], T1.S.first, T2.S.first, T1.S.second, IMTs[0].tensors[i].S.second);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[1].tensors[i], T1.S.first, IMTs[0].tensors[i].S.first, T1.S.second, T2.S.second);
                }
                IMTs[1].isImpure = true;
            } else {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[0].tensors[i].S.first, IMTs[1].tensors[i].S.second, T2.S.second);
                    index_rotation(IMTs[0].tensors[i], tmp);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[0].tensors[i].S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
                    index_rotation(IMTs[1].tensors[i], tmp);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, T2.S.first, T1.S.second, IMTs[0].tensors[i].S.second);
                    index_rotation(IMTs[2].tensors[i], tmp);
                }
                /* 4 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, IMTs[1].tensors[i].S.first, T2.S.first, T1.S.second, T2.S.second);
                    index_rotation(IMTs[4].tensors[i], tmp);
                }
                /* 5 */
                REP(a, DIMENSION) {
                    IMTs[5].tensors[a].UpdateDx(IMTs[4].tensors[a].GetDx());
                    IMTs[5].tensors[a].UpdateDy(IMTs[4].tensors[a].GetDy());
                    IMTs[4].tensors[a].forEach([&](int i, int j, int k, int l, const double *t) {
                        IMTs[5].tensors[a](i, j, k, l) = *t;
                    });
                    IMTs[5].tensors[a].order = IMTs[4].tensors[a].order;
                }
                IMTs[2].isImpure = true;
                IMTs[4].isImpure = true;
                IMTs[5].isImpure = true;
            }
        } else if (merge_point == N/2) {
            if (n == N - 3) {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[0].tensors[i], T1.S.first, T2.S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[1].tensors[i], IMTs[1].tensors[i].S.first, IMTs[0].tensors[i].S.first, IMTs[3].tensors[i].S.second, IMTs[2].tensors[i].S.second);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[2].tensors[i], IMTs[3].tensors[i].S.first, IMTs[2].tensors[i].S.first, T1.S.second, T2.S.second);
                }
                for (auto & tensor : IMTs[3].tensors) {
                    delete tensor.S.first;
                    delete tensor.S.second;
                    tensor.S.first = T1.S.first;
                    tensor.S.second = T1.S.second;
                }
                IMTs[3].isImpure = false;
            } else { // n == N - 2
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[0].tensors[i].S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
                    index_rotation(IMTs[0].tensors[i], tmp);
                }
                /* 1 */
                REP(a, DIMENSION) {
                    IMTs[1].tensors[a].UpdateDx(IMTs[0].tensors[a].GetDx());
                    IMTs[1].tensors[a].UpdateDy(IMTs[0].tensors[a].GetDy());
                    IMTs[0].tensors[a].forEach([&](int i, int j, int k, int l, const double *t) {
                        IMTs[1].tensors[a](i, j, k, l) = *t;
                    });
                    IMTs[1].tensors[a].order = IMTs[0].tensors[a].order;
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, IMTs[1].tensors[i].S.first, IMTs[2].tensors[i].S.first, T1.S.second, IMTs[2].tensors[i].S.second);
                    index_rotation(IMTs[2].tensors[i], tmp);
                }
                /* 3 */
                REP(a, DIMENSION) {
                    IMTs[3].tensors[a].UpdateDx(IMTs[0].tensors[a].GetDx());
                    IMTs[3].tensors[a].UpdateDy(IMTs[0].tensors[a].GetDy());
                    IMTs[2].tensors[a].forEach([&](int i, int j, int k, int l, const double *t) {
                        IMTs[3].tensors[a](i, j, k, l) = *t;
                    });
                    IMTs[3].tensors[a].order = IMTs[2].tensors[a].order;
                }
                IMTs[3].isImpure = true;
            }
        } else { // merge_point > 2
            if (n % 2) {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[0].tensors[i], T1.S.first, T2.S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[1].tensors[i], IMTs[1].tensors[i].S.first, T2.S.first, T1.S.second, IMTs[2].tensors[i].S.second);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[2].tensors[i], IMTs[3].tensors[i].S.first, IMTs[2].tensors[i].S.first, T1.S.second, T2.S.second);
                }
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[3].tensors[i], IMTs[1].tensors[i].S.first, IMTs[0].tensors[i].S.first, IMTs[3].tensors[i].S.second, IMTs[2].tensors[i].S.second);
                }
                /* 5 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[5].tensors[i], T1.S.first, IMTs[0].tensors[i].S.first, IMTs[3].tensors[i].S.second, T2.S.second);
                }
                IMTs[2].isImpure = true;
                IMTs[5].isImpure = true;
            } else {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[0].tensors[i].S.first, IMTs[5].tensors[i].S.second, T2.S.second);
                    index_rotation(IMTs[0].tensors[i], tmp);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[0].tensors[i].S.first, IMTs[3].tensors[i].S.second, IMTs[0].tensors[i].S.second);
                    index_rotation(IMTs[1].tensors[i], tmp);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, T2.S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
                    index_rotation(IMTs[2].tensors[i], tmp);
                }
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, IMTs[1].tensors[i].S.first, T2.S.first, T1.S.second, IMTs[2].tensors[i].S.second);
                    index_rotation(IMTs[3].tensors[i], tmp);
                }
                /* 4 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, IMTs[3].tensors[i].S.first, IMTs[2].tensors[i].S.first, T1.S.second, IMTs[2].tensors[i].S.second);
                    index_rotation(IMTs[4].tensors[i], tmp);
                }
                /* 5 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, IMTs[5].tensors[i].S.first, IMTs[2].tensors[i].S.first, T1.S.second, T2.S.second);
                    index_rotation(IMTs[5].tensors[i], tmp);
                }
                IMTs[4].isImpure = true;
            }
        }
    } else if (n <= N - 3) {
        if (n % 2) {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::contraction(D, D_new, IMTs[0].tensors[i], T1.S.first, IMTs[0].tensors[i].S.first, IMTs[5].tensors[i].S.second, T2.S.second);
            }
            /* 1 */
            REP(i, DIMENSION) {
                TRG::contraction(D, D_new, IMTs[1].tensors[i], T1.S.first, T2.S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
            }
            /* 2 */
            REP(i, DIMENSION) {
                TRG::contraction(D, D_new, IMTs[2].tensors[i], T1.S.first, T2.S.first, T1.S.second, IMTs[2].tensors[i].S.second);
            }
            if (IMTs[3].isImpure) {
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::contraction(D, D_new, IMTs[3].tensors[i], IMTs[3].tensors[i].S.first, T2.S.first, T1.S.second, T2.S.second);
                }
            }
            /* 4 */
            REP(i, DIMENSION) {
                TRG::contraction(D, D_new, IMTs[4].tensors[i], IMTs[1].tensors[i].S.first, IMTs[2].tensors[i].S.first, IMTs[3].tensors[i].S.second, IMTs[4].tensors[i].S.second);
            }
            /* 5 */
            REP(i, DIMENSION) {
                TRG::contraction(D, D_new, IMTs[5].tensors[i], IMTs[5].tensors[i].S.first, IMTs[4].tensors[i].S.first, T1.S.second, T2.S.second);
            }
        } else {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[1].tensors[i].S.first, IMTs[0].tensors[i].S.second, T2.S.second);
                index_rotation(IMTs[0].tensors[i], tmp);
            }
            /* 1 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[2].tensors[i].S.first, IMTs[4].tensors[i].S.second, IMTs[1].tensors[i].S.second);
                index_rotation(IMTs[1].tensors[i], tmp);
            }
            /* 2 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, T1.S.first, T2.S.first, T1.S.second, IMTs[2].tensors[i].S.second);
                index_rotation(IMTs[2].tensors[i], tmp);
            }
            if (IMTs[3].isImpure) {
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::Tensor tmp(D);
                    TRG::contraction(D, D_new, tmp, T1.S.first, T2.S.first, T1.S.second, IMTs[3].tensors[i].S.second);
                    index_rotation(IMTs[3].tensors[i], tmp);
                }
            }
            /* 4 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, IMTs[4].tensors[i].S.first, IMTs[3].tensors[i].S.first, T1.S.second, IMTs[5].tensors[i].S.second);
                index_rotation(IMTs[4].tensors[i], tmp);
            }
            /* 5 */
            REP(i, DIMENSION) {
                TRG::Tensor tmp(D);
                TRG::contraction(D, D_new, tmp, IMTs[0].tensors[i].S.first, IMTs[5].tensors[i].S.first, T1.S.second, T2.S.second);
                index_rotation(IMTs[5].tensors[i], tmp);
            }
        }
    } else if (n == N - 2) {
        /* 0 */
        REP(i, DIMENSION) {
            TRG::Tensor tmp(D);
            TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[1].tensors[i].S.first, IMTs[0].tensors[i].S.second, IMTs[2].tensors[i].S.second);
            index_rotation(IMTs[0].tensors[i], tmp);
        }
        /* 1 */
        REP(i, DIMENSION) {
            TRG::Tensor tmp(D);
            TRG::contraction(D, D_new, tmp, T1.S.first, IMTs[2].tensors[i].S.first, IMTs[4].tensors[i].S.second, IMTs[1].tensors[i].S.second);
            index_rotation(IMTs[1].tensors[i], tmp);
        }
        /* 2 */
        REP(i, DIMENSION) {
            TRG::Tensor tmp(D);
            TRG::contraction(D, D_new, tmp, IMTs[4].tensors[i].S.first, IMTs[3].tensors[i].S.first, T1.S.second, IMTs[5].tensors[i].S.second);
            index_rotation(IMTs[2].tensors[i], tmp);
        }
        /* 3 */
        REP(i, DIMENSION) {
            TRG::Tensor tmp(D);
            TRG::contraction(D, D_new, tmp, IMTs[0].tensors[i].S.first, IMTs[5].tensors[i].S.first, T1.S.second, IMTs[3].tensors[i].S.second);
            index_rotation(IMTs[3].tensors[i], tmp);
        }
        for (auto & tensor : IMTs[4].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
            }
        }
        for (auto & tensor : IMTs[5].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
            }
        }
        IMTs[3].isImpure = true;
        IMTs[4].isImpure = false;
        IMTs[5].isImpure = false;
    } else if (n == N - 1) {
        /* 0 */
        REP(i, DIMENSION) {
            TRG::contraction(D, D_new, IMTs[0].tensors[i], IMTs[3].tensors[i].S.first, IMTs[2].tensors[i].S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
        }
        /* 1 */
        REP(i, DIMENSION) {
            TRG::contraction(D, D_new, IMTs[1].tensors[i], IMTs[1].tensors[i].S.first, IMTs[0].tensors[i].S.first, IMTs[3].tensors[i].S.second, IMTs[2].tensors[i].S.second);
        }
        for (auto & tensor : IMTs[2].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
            }
        }
        for (auto & tensor : IMTs[3].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
            }
        }
        IMTs[2].isImpure = false;
        IMTs[3].isImpure = false;
    } else {
        REP(i, DIMENSION) {
            TRG::contraction(D, D_new, IMTs[0].tensors[i], IMTs[1].tensors[i].S.first, IMTs[0].tensors[i].S.first, IMTs[1].tensors[i].S.second, IMTs[0].tensors[i].S.second);
        }
        for (auto & tensor : IMTs[1].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
            }
        }
        for (auto & tensor : IMTs[0].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
            }
        }
        IMTs[1].isImpure = false;
    }

    /* normalization */
    T1.normalization(normalize_factor);
    REP(i, MAX_IMT_NUM) {
        auto IMT = &IMTs[i];
        if ((*IMT).isImpure) {
            REP(a, DIMENSION) {
                auto tensor = &(*IMT).tensors[a];
                (*tensor).normalization(normalize_factor);
                long long int diff = (*tensor).order - T1.order;
                if (merge_point == 2) {
                    if (n == 1) {
                        diff *= 2;
                    }
                } else if (merge_point == 3) {
                    if (n <= 2) {
                        diff *= 2;
                    } else if (n == 3) {
                        if (i == 0 || i == 2) {
                            diff *= 2;
                        }
                    }
                } else if (merge_point == N/2) {
                    if (n <= N - 3) {
                        diff *= 2;
                    }
                } else {
                    if (count < merge_point - 1) {
                        diff *= 2;
                    } else if (n % 2 == 1 && count == merge_point - 1) {
                        if (i == 0 || i == 2) {
                            diff *= 2;
                        }
                    }
                }
                orders[a] += diff;
            }
        }
    }
}

void TRG::renormalization::trace(Tensor &T, ImpureTensor &IMT, const long long *orders, const int &normalize_factor, double *res) {
    double Tr = T.trace();
    REP(k, DIMENSION) {
        long long int order = orders[k];
        double impureTr = IMT.tensors[k].trace();
        long long int times = std::abs(order);
        if (order > 0) {
            REP(i, times) impureTr *= normalize_factor;
        } else {
            REP(i, times) impureTr /= normalize_factor;
        }
        res[k] = impureTr/Tr;
    }
}
