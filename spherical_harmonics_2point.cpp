//
// Created by Joy on 2020/06/24.
//

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <functional>
#include <HOTRG.hpp>
#include <CG.hpp>
#include <frac.hpp>
#include <tensor.hpp>
#include <impure_tensor.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define DATA_POINTS 12
#define MESH 1e-1
#define LINF 1e300
#define CGFileName "clebsch_gordan.txt"

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void initSphericalHarmonics(const double &K, const int &l_max, Tensor &T, std::map<CG, frac> &map, std::ofstream &CGFile) {
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
    delete[] A;
}

void initTensor(const double &K, const int &l_max, const int &D_cut, Tensor &T, ImpureTensor &IMT, std::map<CG, frac> &map,
                std::ofstream &CGFile) {
    initSphericalHarmonics(K, l_max, T, map, CGFile);

    auto A = new double[l_max];
    REP(i, l_max) {
        A[i] = std::cyl_bessel_i(i + 0.5, K) * (i * 2 + 1);
    }

    REP4(i, j, k, l, l_max) {
                    for (int im = 0; im <= 2 * i; ++im)
                        for (int jm = 0; jm <= 2 * j; ++jm)
                            for (int km = 0; km <= 2 * k; ++km)
                                for (int lm = 0; lm <= 2 * l; ++lm) {
                                    auto sum = new double[3];
                                    sum[0] = 0, sum[1] = 0, sum[2] = 0;
                                    for (int L = std::abs(i - j); L <= i + j; ++L)
                                        for (int L_ = std::abs(k - l); L_ <= k + l; ++L_)
                                            for (int M = -L; M <= L; ++M)
                                                for (int M_ = -L_; M_ <= L_; ++M_) {
                                                    for (int m = 0; m < 3; ++m) {
                                                        if (im - i + jm - j != M || km - k + lm - l != M_ || M_ + m - 1 != M ||
                                                            L < std::abs(L_ - 1) || L_ + 1 < L)
                                                            continue; // CG係数としてありえないものは0なので飛ばす
                                                        frac c(1);
                                                        c *= CG::getCoeff(frac(i), frac(j), frac(im - i), frac(jm - j), frac(L),
                                                                          frac(M),
                                                                          map, CGFile);
                                                        c *= CG::getCoeff(frac(i), frac(j), frac(0), frac(0), frac(L), frac(0), map,
                                                                          CGFile);
                                                        c *= CG::getCoeff(frac(k), frac(l), frac(km - k), frac(lm - l), frac(L_),
                                                                          frac(M_),
                                                                          map, CGFile);
                                                        c *= CG::getCoeff(frac(k), frac(l), frac(0), frac(0), frac(L_), frac(0), map,
                                                                          CGFile);
                                                        c *= CG::getCoeff(frac(L_), frac(1), frac(M_), frac(m - 1), frac(L), frac(M),
                                                                          map,
                                                                          CGFile);
                                                        c *= CG::getCoeff(frac(L_), frac(1), frac(0), frac(0), frac(L), frac(0), map,
                                                                          CGFile);
                                                        c /= frac(2 * L + 1).sign() * (2 * L + 1) * (2 * L + 1);
                                                        sum[m] += c.sign().toDouble() * std::sqrt(frac::abs(c).toDouble());
                                                    }
                                                }
                                    double a = std::sqrt(A[i] * A[j] * A[k] * A[l]);
                                    IMT.tensors[0](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * (sum[0] - sum[2]) / std::sqrt(2);
                                    IMT.tensors[1](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * (sum[0] + sum[2]) / std::sqrt(2);
                                    IMT.tensors[2](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * sum[1];
                                }
                }
    delete[] A;
}

int normalization(Tensor &T, ImpureTensor &originIMT, std::vector<ImpureTensor> &IMTs) {
    double _min = LINF;
    double _max = 0;
    int Dx = T.GetDx();
    int Dy = T.GetDy();
    bool isAllMerged = true;
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    double t = std::abs(T(i, j, k, l));
                    if (t > 0) {
                        _min = std::min(_min, t);
                        _max = std::max(_max, t);
                    }
                }
    for (ImpureTensor &IMT : IMTs) {
        if (!IMT.isMerged) isAllMerged = false;
        for (Tensor &tensor : IMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            double t = std::abs(tensor(i, j, k, l));
                            if (t > 0) {
                                _min = std::min(_min, t);
                                _max = std::max(_max, t);
                            }
                        }
        }
    }
    if (!isAllMerged) {
        for (Tensor &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            double t = std::abs(tensor(i, j, k, l));
                            if (t > 0) {
                                _min = std::min(_min, t);
                                _max = std::max(_max, t);
                            }
                        }
        }
    }
    auto o = static_cast<MKL_INT>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    if (!isAllMerged) {
        for (Tensor &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            if (o > 0) {
                                REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                            } else {
                                REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                            }
                        }
        }
    }
    for (ImpureTensor &IMT : IMTs) {
        for (Tensor &tensor : IMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            if (o > 0) {
                                REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                            } else {
                                REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                            }
                        }
        }
    }
    return o;
}

void Trace(const int n_data_point, double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N, std::map<CG, frac> &map,
           std::ofstream &file) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    // initialize tensor network : max index size is D_cut
    Tensor T(D_cut);
    ImpureTensor originIMT(D_cut);

    std::ofstream CGFile;
    CGFile.open(CGFileName, std::ios::app);
    initTensor(K, l_max, D_cut, T, originIMT, map, CGFile);
    CGFile.close();

    std::vector<ImpureTensor> IMTs(n_data_point);

    auto order = new int[N];
    MKL_INT Dx = D_cut, Dy = D_cut;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        cout << "N = " << n << " :" << std::flush;

        order[n - 1] = normalization(T, originIMT, IMTs);

        if (n <= N / 2) { // compress along x-axis
            cout << " compress along x-axis :" << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            if (n <= n_data_point) {
                int d = 1;
                REP(i, n - 1) d *= 2;
                IMTs[n - 1] = ImpureTensor(d, originIMT);
                IMTs[n - 1].isMerged = true;
                for (int i = 0; i < 3; ++i) {
                    HOTRG::contractionX(D_cut, IMTs[n - 1].tensors[i], originIMT.tensors[i], U, "left");
                }
                for (int i = 0; i < n - 1; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                for (Tensor &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            } else {
                isMerged = true;
                for (int i = 0; i < n_data_point; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along y-axis
            cout << " compress along y-axis :" << std::flush;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (int i = 0; i < n_data_point; ++i) {
                for (auto &tensor : IMTs[i].tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            }
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        if (!isMerged) {
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            cout << " 計算時間 " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << '\n';
            continue;
        }

        double Tr = 0;
        REP(i, Dx) {
            REP(j, Dy) {
                Tr += T(i, j, i, j);
            }
        }

        for (ImpureTensor &IMT : IMTs) {
            double Tr1 = 0, Tr2 = 0, Tr3 = 0;
            REP(i, Dx)
                REP(j, Dy) {
                    Tr1 += IMT.tensors[0](i, j, i, j);
                    Tr2 += IMT.tensors[1](i, j, i, j);
                    Tr3 += IMT.tensors[2](i, j, i, j);
                }
            double res = (Tr1 - Tr2 + Tr3) / Tr;
            IMT.corrs.push_back(res);
            cout << '\t' << std::fixed << std::setprecision(10) << res << std::flush;
        }
        cout << '\n';
    }
    for (ImpureTensor &IMT : IMTs) {
        file << IMT.distance;
        for (double corr : IMT.corrs) {
            file << '\t' << std::fixed << std::setprecision(10) << corr << std::flush;
        }
        file << '\n';
    }
    delete[] order;
}

int main() {
    /* inputs */
    MKL_INT N = 15;     // volume : 2^N
    MKL_INT l_max;  // l_max
    MKL_INT D_cut; // bond dimension
    double K = 1.9; // inverse temperature
    int n_data_point = 7; // number of d. d = 1, 2, 4, 8, 16, ...

    /* Clebsch-Gordan coefficient */
    std::map<CG, frac> map;
    std::ifstream CGFile;
    CGFile.open(CGFileName, std::ios::in);
    int l1, l2, m1, m2, L, M, num, den;
    while (CGFile >> l1 >> l2 >> m1 >> m2 >> L >> M >> num >> den) {
        if (map.find(CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))) != map.end()) {
            cerr << "clebsch_gordan.txt is broken." << '\n';
            return 1;
        }
        map[CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))] = frac(num, den);
    }
    CGFile.close();

    /* calculation */
    for (l_max = 1; l_max <= 4; ++l_max) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        cout << "---------- " << l_max << " ----------\n";
        const string fileName = "new_2point_spherical_harmonics_l" + std::to_string(l_max) + "_N" + std::to_string(N) + ".txt";
        std::ofstream dataFile;
        dataFile.open(fileName, std::ios::trunc);
        D_cut = (l_max + 1) * (l_max + 1);
        Trace(n_data_point, K, D_cut, l_max, N, map, dataFile);
        dataFile.close();
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
    }

    return 0;
}
