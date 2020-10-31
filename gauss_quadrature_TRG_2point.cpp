#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <impure_tensor.hpp>
#include <TRG.hpp>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N)REP(j, N)REP(k, N)REP(l, N)

#define MESH 1e-1
#define MAX_IMT_NUM 6
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void index_rotation(TRG::Tensor &T, TRG::Tensor &tmp) {
    T.UpdateDx(tmp.GetDx());
    T.UpdateDy(tmp.GetDy());
    tmp.forEach([&](int i, int j, int k, int l, const double *t) {
        T(l, i, j, k) = *t;
    });
    T.order = tmp.order;
}

void Trace(const int merge_point, double const K, const int D_cut, const int n_node, const int N, std::ofstream &file) {
    time_counter time;
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    TRG::Tensor T1(D, D_cut); /* (ij)(kl) -> S1 S3 */
    TRG::Tensor T2(D, D_cut); /* (jk)(li) -> S2 S4 */
    T1.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
    T2.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
    TRG::ImpureTensor IMTs[MAX_IMT_NUM];
    for (auto &IMT : IMTs) IMT = TRG::ImpureTensor(D, D_cut);
    GaussQuadrature::initTensorWithImpure(K, n_node, D_cut, D, T1, IMTs[0]);
    IMTs[0].isImpure = true;
    if (merge_point == 1) {
        IMTs[1] = IMTs[0];
        IMTs[1].isImpure = true;
    }
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    REP(i, MAX_IMT_NUM) {
        for (auto & tensor : IMTs[i].tensors) {
            if (i % 2) {
                tensor.S = T1.S;
//                cout << " T1.S " << &T1.S << " tensor.S " << &tensor.S << " " << std::flush;
            } else {
                tensor.S = T2.S;
            }
        }
    }

    for (int n = 1; n <= N; ++n) {
        const int count = (n + 1) / 2;
        time.start();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        const int D_new = std::min(D * D, D_cut);

        /* normalization */
//        T1.normalization(NORMALIZE_FACTOR);
//        for (auto &IMT : IMTs) {
//            if (IMT.isImpure) {
//                for (auto &tensor : IMT.tensors) {
//                    tensor.normalization(NORMALIZE_FACTOR);
//                }
//            }
//        }

        /* SVD pure tensor T */
        T2 = T1;
        TRG::SVD(D, D_new, T1, true);
        TRG::SVD(D, D_new, T2, false);

        /* SVD impure tensor IMTs */
        REP(i, MAX_IMT_NUM) {
            if (IMTs[i].isImpure) {
                string allocate;
                string tmp;
                for (auto & tensor : IMTs[i].tensors) {
                    if (tensor.S.first == T1.S.first || tensor.S.first == T2.S.first/* same as tensor.S.second == T1.S.second || tensor.S.second == T2.S.second */) {
                        allocate = " allocate";
                        tensor.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
                    }
                    if (merge_point == 1 || count <= merge_point - 1) {
                        tmp = " " + std::to_string(i) + (i % 2 == 1 ? "rightUp " : "leftUp ");
                        TRG::SVD(D, D_new, tensor, i % 2 == 1);
                    } else if (n <= N - 2) {
                        if (n % 2) {
                            tmp = " " + std::to_string(i) + (i % 2 == 1 ? "rightUp " : "leftUp ");
                            TRG::SVD(D, D_new, tensor, i % 2 == 1);
                        } else {
                            tmp = " " + std::to_string(i) + (i % 4 == 0 ? "rightUp " : "leftUp ");
                            TRG::SVD(D, D_new, tensor, i % 4 == 0);
                        }
                    } else {
                        tmp = " " + std::to_string(i) + (i % 2 == 1 ? "rightUp " : "leftUp ");
                        TRG::SVD(D, D_new, tensor, i % 2 == 1);
                    }
                }
                cout << allocate << tmp;
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

        D = T1.GetDx(); // same as T.GetDy();

        /* normalization */
        T1.normalization(NORMALIZE_FACTOR);
        for (auto &IMT : IMTs) {
            if (IMT.isImpure) {
                for (auto &tensor : IMT.tensors) {
                    tensor.normalization(NORMALIZE_FACTOR);
                }
            }
        }

        cout << " pure order " << T1.order << " : impure order ";
        REP(i, MAX_IMT_NUM) cout << IMTs[i].tensors[0].order << " ";
        cout << ":" << std::flush;

        if (n < N) {
            time.end();
            cout << " in " << time.duration_cast_to_string() << '\n';
            continue;
        }

        double Tr = 0;
        REP(i, D)REP(j, D) Tr += T1(i, j, i, j);

        double impure_Tr[DIMENSION];
        REP(k, DIMENSION) {
            impure_Tr[k] = 0;
            long long int order = IMTs[0].tensors[k].order - T1.order;
            REP(i, D)REP(j, D) {
                    impure_Tr[k] += IMTs[0].tensors[k](i, j, i, j);
                }
//            REP(i, T1.orders.size()) {
//                order += IMTs[0].tensors[k].orders[i] - T1.orders[i];
//            }
            const long long int times = std::abs(order);
            if (order > 0) {
                REP(i, times) impure_Tr[k] *= NORMALIZE_FACTOR;
            } else {
                REP(i, times) impure_Tr[k] /= NORMALIZE_FACTOR;
            }
        }
        double res = (impure_Tr[0] + impure_Tr[1] + impure_Tr[2]) / Tr;
        IMTs[0].corrs.push_back(res);
        cout << '\t' << std::scientific << std::setprecision(16) << res << std::flush;

        time.end();
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
    for (auto &IMT : IMTs) {
        file << IMT.distance;
        for (double corr : IMT.corrs) {
            file << '\t' << std::scientific << std::setprecision(16) << corr << std::flush;
        }
        file << '\n';
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    MKL_INT N = 20;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 16; // bond dimension
    double K = 1.8; // inverse temperature
    int merge_point = 7; // d = 2^(merge_point - 1)

    N = std::stoi(argv[1]);
    n_node = std::stoi(argv[2]);
    D_cut = std::stoi(argv[3]);
    K = std::stod(argv[4]);
    merge_point = std::stoi(argv[5]);

    const string dir = "gauss_quadrature_TRG_2point";
    time_counter time;
    string fileName;
    std::ofstream dataFile;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << K << ", merge_point = " << merge_point << '\n';
    fileName = dir + "_N" + std::to_string(N) + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_beta" + ss.str() + "_" + std::to_string(merge_point) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(merge_point, K, D_cut, n_node, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 56; D_cut <= 64; D_cut += 8) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(merge_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    /* vs n_node */
//    for (n_node = 48; n_node <= 64; n_node += 16) {
//        time.start();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(merge_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
