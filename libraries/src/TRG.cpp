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

void TRG::SVD(Tensor &T, bool is_default, bool is_odd_times) {
    const int D = T.GetDx(), D_cut = T.GetD_max(), D_new = std::min(D * D, D_cut);
    auto sigma = new double[D * D];
    auto U_ = new double[D * D * D * D], VT_ = new double[D * D * D * D];
    auto superb = new double[D * D - 1];
    MKL_INT info;
    if (is_default) { // (ij)(kl)
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, T.array, D * D, sigma, U_, D * D, VT_, D * D, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        if (is_odd_times) { // S1 S3
            REP(k, D_new) {
                double s = std::sqrt(sigma[k]);
                REP(i, D)REP(j, D) {
                        T.S.first->array[D * D * k + D * j + i] = U_[D * D * D * i + D * D * j + k] * s;
                        T.S.second->array[D * D * k + D * j + i] = VT_[D * D * k + D * i + j] * s;
                    }
            }
        } else { // S2 S4
            REP(k, D_new) {
                double s = std::sqrt(sigma[k]);
                REP(i, D)REP(j, D) {
                        T.S.first->array[D_new * D * j + D_new * i + k] = U_[D * D * D * i + D * D * j + k] * s;
                        T.S.second->array[D_new * D * j + D_new * i + k] = VT_[D * D * k + D * i + j] * s;
                    }
            }
        }
    } else {
        BaseTensor M(D);
        if (is_odd_times) {
            T.forEach([&](int i, int j, int k, int l, const double *t) {
                M(j, k, l, i) = *t; // M(jk)(li)
            });
        } else {
            T.forEach([&](int i, int j, int k, int l, const double *t) {
                M(l, i, j, k) = *t; // M(li)(jk)
            });
        }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, M.array, D * D, sigma, U_, D * D, VT_, D * D, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        if (is_odd_times) { // S2 S4
            REP(k, D_new) {
                double s = std::sqrt(sigma[k]);
                REP(i, D)REP(j, D) {
                        T.S.first->array[D_new * D * j + D_new * i + k] = U_[D * D * D * i + D * D * j + k] * s;
                        T.S.second->array[D_new * D * j + D_new * i + k] = VT_[D * D * k + D * i + j] * s;
                    }
            }
        } else { // S1 S3
            REP(k, D_new) {
                double s = std::sqrt(sigma[k]);
                REP(i, D)REP(j, D) {
                        T.S.first->array[D * D * k + D * j + i] = U_[D * D * D * i + D * D * j + k] * s;
                        T.S.second->array[D * D * k + D * j + i] = VT_[D * D * k + D * i + j] * s;
                    }
            }
        }
    }
    delete[] U_;
    delete[] VT_;
    delete[] sigma;
    delete[] superb;
}

void TRG::contraction(Tensor &T, Tensor &T1, Tensor &T2, Tensor &T3, Tensor &T4) {
    const int D = T.GetDx(), D_cut = T.GetD_max(), D_new = std::min(D * D, D_cut);
    BaseTensor top_(D_new, D_new, D, D), bottom_(D, D, D_new, D_new);
    BaseTensor X_(D_new, D, D, D_new);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D_new * D, D_new * D, D, 1, T1.S.first->array,
            D, T2.S.first->array, D_new * D, 0, X_.array, D_new * D);
    X_.forEach([&](int i, int b, int a, int j, const double *t) {
        bottom_(a, b, i, j) = *t;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D_new * D, D_new * D, D, 1, T3.S.second->array,
            D, T4.S.second->array, D_new * D, 0, X_.array, D_new * D);
    X_.forEach([&](int i, int a, int b, int j, const double *t) {
        top_(i, j, a, b) = *t;
    });
    T.UpdateDx(D_new);
    T.UpdateDy(D_new);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            D_new * D_new, D_new * D_new, D * D, 1, top_.array, D * D, bottom_.array, D_new * D_new, 0, T.array, D_new * D_new);
}

TRG::Unitary_S::Unitary_S(int D_cut) {
    array = new double[D_cut * D_cut * D_cut];
    this->D_cut = D_cut;
    REP(i, D_cut * D_cut * D_cut) array[i] = 0;
}

TRG::Unitary_S::~Unitary_S() {
    delete[] array;
    array = nullptr;
}

TRG::Tensor::~Tensor() {
    delete S.first;
    delete S.second;
    S.first = nullptr;
    S.second = nullptr;
}

TRG::Tensor &TRG::Tensor::operator=(const Tensor &rhs) {
    BaseTensor::operator=(rhs);
    return *this;
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

void
TRG::initialize_spherical_harmonics_with_impure(Tensor &T1, Tensor &T2, ImpureTensor (&IMTs)[MAX_IMT_NUM], const int &D, const int &D_cut, const double &beta, const int &l_max, const int &merge_point) {
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

void
TRG::initialize_gauss_quadrature_with_impure(Tensor &T1, Tensor &T2, ImpureTensor (&IMTs)[MAX_IMT_NUM], const int &D, const int &D_cut, const double &beta, const int &n_node, const int &merge_point) {
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
        for (auto &tensor : IMTs[i].tensors) {
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
    const bool is_odd_times = n % 2;
    /* normalization */
    orders[n - 1] = T1.normalization(normalize_factor);

    /* SVD */
    T2 = T1;
    if (is_odd_times) {
        TRG::SVD(T1, true, true);
        TRG::SVD(T2, false, true);
    } else {
        TRG::SVD(T1, false, false);
        TRG::SVD(T2, true, false);
    }

    /* contraction */
    TRG::contraction(T1, T1, T2, T1, T2);

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

void
TRG::renormalization::two_point(Tensor &T1, Tensor &T2, ImpureTensor (&IMTs)[MAX_IMT_NUM], long long *orders, const int &N, const int &n, const int &merge_point, const int &normalize_factor) {
    const bool is_odd_times = n % 2;
    const int D_cut = T1.GetD_max();
    const int count = (n + 1) / 2;

    /* SVD pure tensor T */
    T2 = T1;
    if (is_odd_times) {
        TRG::SVD(T1, true, true);
        TRG::SVD(T2, false, true);
    } else {
        TRG::SVD(T1, false, false);
        TRG::SVD(T2, true, false);
    }

    /* SVD impure tensor IMTs */
    REP(i, MAX_IMT_NUM) {
        auto IMT = &IMTs[i];
        if (IMT->isImpure) {
            for (auto &tensor : IMT->tensors) {
                /* same as tensor.S.second == T1.S.second || tensor.S.second == T2.S.second */
                if (tensor.S.first == T1.S.first || tensor.S.first == T2.S.first) {
                    tensor.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
                }
                if (merge_point == 1 || count <= merge_point - 1) {
                    TRG::SVD(tensor, i % 2 == 1, is_odd_times);
                } else if (n <= N - 2) {
                    if (is_odd_times) {
                        TRG::SVD(tensor, i % 2 == 1, true);
                    } else {
                        TRG::SVD(tensor, i % 4 == 0, false);
                    }
                } else {
                    TRG::SVD(tensor, i % 2 == 1, is_odd_times);
                }
            }
        } else {
            for (auto &tensor : IMTs[i].tensors) {
                if (merge_point == 1 || count < merge_point - 1) {
                    if ((i % 2 == 1) == is_odd_times) {
                        tensor.S = T1.S;
                    } else {
                        tensor.S = T2.S;
                    }
                } else if (n <= N - 2) {
                    if (is_odd_times) {
                        if (i % 2 == 1) {
                            tensor.S = T1.S;
                        } else {
                            tensor.S = T2.S;
                        }
                    } else {
                        if (i % 4 != 0) {
                            tensor.S = T1.S;
                        } else {
                            tensor.S = T2.S;
                        }
                    }
                } else {
                    if ((i % 2 == 1) == is_odd_times) {
                        tensor.S = T1.S;
                    } else {
                        tensor.S = T2.S;
                    }
                }
            }
        }
    }

    /* contraction pure tensor T */
    TRG::contraction(T1, T1, T2, T1, T2);

    /* contraction impure tensor IMTs */
    if ((merge_point == 1 && n <= N - 2) || count < merge_point - 1) {
        if (n % 2) {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[0].tensors[i], T1, T2, IMTs[1].tensors[i], IMTs[0].tensors[i]);
            }
            if (IMTs[1].isImpure) {
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[1].tensors[i], IMTs[1].tensors[i], T2, T1, IMTs[2].tensors[i]);
                }
                if (IMTs[3].isImpure) {
                    /* 2 */
                    REP(i, DIMENSION) {
                        TRG::contraction(IMTs[2].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i], T1, T2);
                    }
                    IMTs[2].isImpure = true;
                }
            }
            /* 3 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[3].tensors[i], T1, IMTs[0].tensors[i], IMTs[3].tensors[i], T2);
            }
            IMTs[3].isImpure = true;
        } else {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[0].tensors[i], T1, T2, IMTs[0].tensors[i], IMTs[3].tensors[i]);
            }
            /* 1 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[1].tensors[i], IMTs[0].tensors[i], T2, T1, IMTs[1].tensors[i]);
            }
            if (IMTs[1].isImpure) {
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[2].tensors[i], IMTs[2].tensors[i], IMTs[1].tensors[i], T1, T2);
                }
                IMTs[2].isImpure = true;
            }
            /* 3 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[3].tensors[i], T1, IMTs[3].tensors[i], IMTs[2].tensors[i], T2);
            }
            IMTs[1].isImpure = true;
        }
    } else if (count == merge_point - 1) {
        if (merge_point == 2) {
            if (n == 1) {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[0].tensors[i], T1, T2, T1, IMTs[0].tensors[i]);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[1].tensors[i], T1, IMTs[0].tensors[i], T1, T2);
                }
                IMTs[1].isImpure = true;
            } else { // n == 2
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[0].tensors[i], T1, T2, IMTs[0].tensors[i], IMTs[1].tensors[i]);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[1].tensors[i], IMTs[0].tensors[i], T2, IMTs[0].tensors[i], IMTs[1].tensors[i]);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[2].tensors[i], IMTs[0].tensors[i], T2, T1, T2);
                }
                /* 4 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[4].tensors[i], T1, IMTs[1].tensors[i], T1, T2);
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
        } else if (merge_point == N / 2) {
            if (n == N - 3) {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[0].tensors[i], T1, T2, IMTs[1].tensors[i], IMTs[0].tensors[i]);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[1].tensors[i], IMTs[1].tensors[i], IMTs[0].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i]);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[2].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i], T1, T2);
                }
                for (auto &tensor : IMTs[3].tensors) {
                    delete tensor.S.first;
                    delete tensor.S.second;
                    tensor.S = std::make_pair(nullptr, nullptr);
                }
                IMTs[3].isImpure = false;
            } else { // n == N - 2
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[0].tensors[i], IMTs[0].tensors[i], T2, IMTs[0].tensors[i], IMTs[1].tensors[i]);
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
                    TRG::contraction(IMTs[2].tensors[i], IMTs[2].tensors[i], IMTs[1].tensors[i], IMTs[2].tensors[i], T2);
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
                    TRG::contraction(IMTs[0].tensors[i], T1, T2, IMTs[1].tensors[i], IMTs[0].tensors[i]);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[1].tensors[i], IMTs[1].tensors[i], T2, T1, IMTs[2].tensors[i]);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[2].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i], T1, T2);
                }
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[3].tensors[i], IMTs[1].tensors[i], IMTs[0].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i]);
                }
                /* 5 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[5].tensors[i], T1, IMTs[0].tensors[i], IMTs[3].tensors[i], T2);
                }
                IMTs[2].isImpure = true;
                IMTs[5].isImpure = true;
            } else {
                /* 0 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[0].tensors[i], T1, T2, IMTs[0].tensors[i], IMTs[5].tensors[i]);
                }
                /* 1 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[1].tensors[i], IMTs[0].tensors[i], T2, IMTs[0].tensors[i], IMTs[3].tensors[i]);
                }
                /* 2 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[2].tensors[i], IMTs[0].tensors[i], T2, T1, IMTs[1].tensors[i]);
                }
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[3].tensors[i], IMTs[2].tensors[i], IMTs[1].tensors[i], T1, T2);
                }
                /* 4 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[4].tensors[i], IMTs[2].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i], T2);
                }
                /* 5 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[5].tensors[i], T1, IMTs[5].tensors[i], IMTs[2].tensors[i], T2);
                }
                IMTs[4].isImpure = true;
            }
        }
    } else if (n <= N - 3) {
        if (n % 2) {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[0].tensors[i], T1, IMTs[0].tensors[i], IMTs[5].tensors[i], T2);
            }
            /* 1 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[1].tensors[i], T1, T2, IMTs[1].tensors[i], IMTs[0].tensors[i]);
            }
            /* 2 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[2].tensors[i], T1, T2, T1, IMTs[2].tensors[i]);
            }
            if (IMTs[3].isImpure) {
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[3].tensors[i], IMTs[3].tensors[i], T2, T1, T2);
                }
            }
            /* 4 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[4].tensors[i], IMTs[1].tensors[i], IMTs[2].tensors[i], IMTs[3].tensors[i], IMTs[4].tensors[i]);
            }
            /* 5 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[5].tensors[i], IMTs[5].tensors[i], IMTs[4].tensors[i], T1, T2);
            }
        } else {
            /* 0 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[0].tensors[i], T1, T2, IMTs[1].tensors[i], IMTs[0].tensors[i]);
            }
            /* 1 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[1].tensors[i], IMTs[1].tensors[i], T2, IMTs[2].tensors[i], IMTs[4].tensors[i]);
            }
            /* 2 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[2].tensors[i], IMTs[2].tensors[i], T2, T1, T2);
            }
            if (IMTs[3].isImpure) {
                /* 3 */
                REP(i, DIMENSION) {
                    TRG::contraction(IMTs[3].tensors[i], IMTs[3].tensors[i], T2, T1, T2);
                }
            }
            /* 4 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[4].tensors[i], IMTs[5].tensors[i], IMTs[4].tensors[i], IMTs[3].tensors[i], T2);
            }
            /* 5 */
            REP(i, DIMENSION) {
                TRG::contraction(IMTs[5].tensors[i], T1, IMTs[0].tensors[i], IMTs[5].tensors[i], T2);
            }
        }
    } else if (n == N - 2) {
        /* 0 */
        REP(i, DIMENSION) {
            TRG::contraction(IMTs[0].tensors[i], IMTs[2].tensors[i], T2, IMTs[1].tensors[i], IMTs[0].tensors[i]);
        }
        /* 1 */
        REP(i, DIMENSION) {
            TRG::contraction(IMTs[1].tensors[i], IMTs[1].tensors[i], T2, IMTs[2].tensors[i], IMTs[4].tensors[i]);
        }
        /* 2 */
        REP(i, DIMENSION) {
            TRG::contraction(IMTs[2].tensors[i], IMTs[5].tensors[i], IMTs[4].tensors[i], IMTs[3].tensors[i], T2);
        }
        /* 3 */
        REP(i, DIMENSION) {
            TRG::contraction(IMTs[3].tensors[i], IMTs[3].tensors[i], IMTs[0].tensors[i], IMTs[5].tensors[i], T2);
        }
        for (auto &tensor : IMTs[4].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
                tensor.S = std::make_pair(nullptr, nullptr);
            }
        }
        for (auto &tensor : IMTs[5].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
                tensor.S = std::make_pair(nullptr, nullptr);
            }
        }
        IMTs[3].isImpure = true;
        IMTs[4].isImpure = false;
        IMTs[5].isImpure = false;
    } else if (n == N - 1) {
        /* 0 */
        REP(i, DIMENSION) {
            TRG::contraction(IMTs[0].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i], IMTs[1].tensors[i], IMTs[0].tensors[i]);
        }
        /* 1 */
        REP(i, DIMENSION) {
            TRG::contraction(IMTs[1].tensors[i], IMTs[1].tensors[i], IMTs[0].tensors[i], IMTs[3].tensors[i], IMTs[2].tensors[i]);
        }
        for (auto &tensor : IMTs[2].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
                tensor.S = std::make_pair(nullptr, nullptr);
            }
        }
        for (auto &tensor : IMTs[3].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
                tensor.S = std::make_pair(nullptr, nullptr);
            }
        }
        IMTs[2].isImpure = false;
        IMTs[3].isImpure = false;
    } else {
        REP(i, DIMENSION) {
            TRG::contraction(IMTs[0].tensors[i], IMTs[0].tensors[i], IMTs[1].tensors[i], IMTs[0].tensors[i], IMTs[1].tensors[i]);
        }
        for (auto &tensor : IMTs[1].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
                tensor.S = std::make_pair(nullptr, nullptr);
            }
        }
        for (auto &tensor : IMTs[0].tensors) {
            if (tensor.S.first != T1.S.first && tensor.S.first != T2.S.first) {
                delete tensor.S.first;
                delete tensor.S.second;
                tensor.S = std::make_pair(nullptr, nullptr);
            }
        }
        IMTs[1].isImpure = false;
    }

    /* normalization */
    T1.normalization(normalize_factor);
    REP(i, MAX_IMT_NUM) {
        auto IMT = &IMTs[i];
        if (IMT->isImpure) {
            REP(a, DIMENSION) {
                auto tensor = &IMT->tensors[a];
                tensor->normalization(normalize_factor);
                long long int diff = tensor->order - T1.order;
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
                } else if (merge_point == N / 2) {
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

    for (auto &IMT : IMTs) {
        if (!IMT.isImpure) {
            for (auto &tensor : IMT.tensors) tensor.S = std::make_pair(nullptr, nullptr);
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
        res[k] = impureTr / Tr;
    }
}
