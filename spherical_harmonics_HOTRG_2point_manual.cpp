#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <spherical_harmonics.hpp>
#include <HOTRG.hpp>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1
#define LINF 1e300
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

int normalization(HOTRG::Tensor &T, HOTRG::ImpureTensor &originIMT, std::vector<HOTRG::ImpureTensor> &IMTs) {
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
    for (auto &IMT : IMTs) {
        if (!IMT.isMerged) isAllMerged = false;
        for (BaseTensor &tensor : IMT.tensors) {
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
        for (auto &tensor : originIMT.tensors) {
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
                    REP(t, std::abs(o)) {
                        if (o > 0) {
                            T(i, j, k, l) /= 10;
                        } else {
                            T(i, j, k, l) *= 10;
                        }
                    }
                }
    for (auto &IMT : IMTs) {
        for (auto &tensor : IMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            REP(t, std::abs(o)) {
                                if (o > 0) {
                                    tensor(i, j, k, l) /= 10;
                                } else {
                                    tensor(i, j, k, l) *= 10;
                                }
                            }
                        }
        }
    }
    if (!isAllMerged) {
        for (auto &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            REP(t, std::abs(o)) {
                                if (o > 0) {
                                    tensor(i, j, k, l) /= 10;
                                } else {
                                    tensor(i, j, k, l) *= 10;
                                }
                            }
                        }
        }
    }
    return o;
}

void Trace(double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N, std::vector<int> &d, std::ofstream &file) {
    time_counter time;

    const int DATA_POINTS = d.size();

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D_cut);
    HOTRG::ImpureTensor originIMT(D_cut);
    SphericalHarmonics::initTensorWithImpure(K, l_max, T, originIMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    std::vector<HOTRG::ImpureTensor> IMTs(DATA_POINTS);
    REP(i, DATA_POINTS) {
        IMTs[i] = HOTRG::ImpureTensor(d[i], originIMT);
    }

    auto order = new int[N];
    MKL_INT Dx = D_cut, Dy = D_cut;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        order[n - 1] = normalization(T, originIMT, IMTs);

        if (n <= N / 2) { // compression along x-axis
            cout << " compress along x-axis " << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            bool isAllMerged = true;
            for (auto &IMT : IMTs) {
                if (IMT.isMerged) {
                    for (BaseTensor &tensor : IMT.tensors) {
                        HOTRG::contractionX(D_cut, tensor, T, U, "left");
                    }
                } else {
                    if (IMT.distance >> n) {
                        if (IMT.distance & (1 << (n - 1))) {
                            for (auto &tensor : IMT.tensors) {
                                HOTRG::contractionX(D_cut, T, tensor, U, "right");
                            }
                        } else {
                            for (auto &tensor : IMT.tensors) {
                                HOTRG::contractionX(D_cut, tensor, T, U, "left");
                            }
                        }
                        isAllMerged = false;
                    } else {
                        for (int i = 0; i < 3; ++i) {
                            HOTRG::contractionX(D_cut, originIMT.tensors[i], IMT.tensors[i], U, "right");
                        }
                        IMT.isMerged = true;
                    }
                }
            }
            if (!isMerged) {
                if (isAllMerged) {
                    isMerged = true;
                } else {
                    for (auto &tensor : originIMT.tensors) {
                        HOTRG::contractionX(D_cut, tensor, T, U, "left");
                    }
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compression along y-axis
            cout << " compress along y-axis " << std::flush;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (auto &IMT : IMTs) {
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
                }
            }
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        if (!isMerged) {
            time.end();
            cout << "in " << time.duration_cast_to_string() << '\n';
            continue;
        }

        double Tr = 0;
        REP(i, Dx) {
            REP(j, Dy) {
                Tr += T(i, j, i, j);
            }
        }

        for (auto &IMT : IMTs) {
            double Tr1 = 0, Tr2 = 0, Tr3 = 0;
            REP(i, Dx)
                REP(j, Dy) {
                    Tr1 += IMT.tensors[0](i, j, i, j);
                    Tr2 += IMT.tensors[1](i, j, i, j);
                    Tr3 += IMT.tensors[2](i, j, i, j);
                }
            double res = (Tr1 - Tr2 + Tr3) / Tr;
            IMT.corrs.push_back(res);
            cout << '\t' << std::scientific << std::setprecision(16) << res << std::flush;
        }
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
    delete[] order;
}

int main() {
    /* inputs */
    MKL_INT N = 16;     // volume : 2^N
    MKL_INT l_max;  // l_max
    MKL_INT D_cut; // bond dimension
    double K = 1.9; // inverse temperature
    std::vector<int> d = {8};

    const string dir = "spherical_harmonics_HOTRG_2point_manual";
    time_counter time;
    string fileName;
    std::ofstream dataFile;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;

    /* calculation */
    for (l_max = 1; l_max <= 4; ++l_max) {
        time.start();
        cout << "---------- " << l_max << " ----------\n";
        fileName = dir + "_l" + std::to_string(l_max) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
        dataFile.open(fileName, std::ios::trunc);
        D_cut = (l_max + 1) * (l_max + 1);
        Trace(K, D_cut, l_max, N, d, dataFile);
        dataFile.close();
        time.end();
        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
    }

    return 0;
}
