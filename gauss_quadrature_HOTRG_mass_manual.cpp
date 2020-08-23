#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <tensor.hpp>
#include <impure_tensor.hpp>
#include <HOTRG.hpp>
#include <cmath>
#include <sstream>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1
#define LINF 1e300

using std::cin;
using std::cout;
using std::cerr;
using std::string;

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


void Trace(double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, std::vector<int> d, std::ofstream &file) {
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    const int DATA_POINTS = d.size();

    // initialize tensor network : max index size is D_cut
    Tensor T(D, D_cut, N);
    ImpureTensor originIMT(D, D_cut, N);

    GaussQuadrature::initTensorWithImpure(K, n_node, D_cut, D, T, originIMT);

    std::vector<ImpureTensor> IMTs(DATA_POINTS);
    REP(i, DATA_POINTS) {
        IMTs[i] = ImpureTensor(d[i], originIMT);
    }

    auto order = new int[N];
    MKL_INT Dx = D, Dy = D;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        order[n - 1] = normalization(T, originIMT, IMTs);

        if (n <= N / 2) { // compression along x-axis
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            bool isAllMerged = true;
            for (ImpureTensor &IMT : IMTs) {
                if (IMT.isMerged) {
                    for (Tensor &tensor : IMT.tensors) {
                        HOTRG::contractionX(D_cut, tensor, T, U, "left");
                    }
                } else {
                    if (IMT.distance >> n) {
                        if (IMT.distance & (1 << (n - 1))) {
                            for (Tensor &tensor : IMT.tensors) {
                                HOTRG::contractionX(D_cut, T, tensor, U, "right");
                            }
                        } else {
                            for (Tensor &tensor : IMT.tensors) {
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
                    for (Tensor &tensor : originIMT.tensors) {
                        HOTRG::contractionX(D_cut, tensor, T, U, "left");
                    }
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compression along y-axis
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (ImpureTensor &IMT : IMTs) {
                for (Tensor &tensor : IMT.tensors) {
                    HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
                }
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
            double res = (Tr1 + Tr2 + Tr3) / Tr;
            IMT.corrs.push_back(res);
            cout << '\t' << std::fixed << std::setprecision(16) << res << std::flush;
        }
        cout << '\n';
    }
    for (ImpureTensor &IMT : IMTs) {
        file << IMT.distance;
        for (double corr : IMT.corrs) {
            file << '\t' << std::fixed << std::setprecision(16) << corr << std::flush;
        }
        file << '\n';
    }
    delete[] order;
}

int main() {
    /* inputs */
    MKL_INT N = 16;     // volume : 2^N
    double K = 1.9; // inverse temperature
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 36; // bond dimension
    std::vector<int> d = {64}; // distances

    const string dir = "gauss_quadrature_HOTRG_mass_manual";
    time_counter time;
    string fileName;
    std::ofstream dataFile;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;

    /* calculation */
    time.start();
    fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(K, D_cut, n_node, N, d, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
    for (D_cut = 8; D_cut <= 32; D_cut += 4) {
        time.start();
        cout << "---------- " << D_cut << " ----------\n";
        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
        dataFile.open(fileName, std::ios::trunc);
        Trace(K, D_cut, n_node, N, d, dataFile);
        dataFile.close();
        time.end();
        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
    }

    /* vs n_node */
    for (n_node = 8; n_node <= 32; n_node += 8) {
        time.start();
        cout << "---------- " << n_node << " ----------\n";
        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
        dataFile.open(fileName, std::ios::trunc);
        Trace(K, D_cut, n_node, N, d, dataFile);
        dataFile.close();
        time.end();
        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
    }

    return 0;
}
