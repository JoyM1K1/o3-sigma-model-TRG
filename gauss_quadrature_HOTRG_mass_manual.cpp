#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <HOTRG.hpp>
#include <cmath>
#include <sstream>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)

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
        for (auto &tensor : IMT.tensors) {
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
    auto o = static_cast<int>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    if (!isAllMerged) {
        for (auto &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            if (o > 0) {
                                REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                            } else {
                                REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                            }
                        }
        }
    }
    for (auto &IMT : IMTs) {
        for (auto &tensor : IMT.tensors) {
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


void Trace(double const K, int const D_cut, int const n_node, int const N, std::vector<int> d, std::ofstream &file) {
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    const int DATA_POINTS = d.size();

    // initialize tensor network : max index size is D_cut
    HOTRG::Tensor T(D, D_cut);
    HOTRG::ImpureTensor originIMT(D, D_cut);

    GaussQuadrature::init_tensor_with_impure(K, n_node, D_cut, D, T, originIMT);

    std::vector<HOTRG::ImpureTensor> IMTs(DATA_POINTS);
    REP(i, DATA_POINTS) {
        IMTs[i] = HOTRG::ImpureTensor(d[i], originIMT);
    }

    auto order = new int[N];
    int Dx = D, Dy = D;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        order[n - 1] = normalization(T, originIMT, IMTs);

        if (n <= N / 2) { // compression along x-axis
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            bool isAllMerged = true;
            for (auto &IMT : IMTs) {
                if (IMT.isMerged) {
                    for (auto &tensor : IMT.tensors) {
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

        for (auto &IMT : IMTs) {
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
    for (auto &IMT : IMTs) {
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
    int N = 16;     // volume : 2^N
    double K = 1.9; // inverse temperature
    int n_node = 32;  // n_node
    int D_cut = 36; // bond dimension
    std::vector<int> d = {64}; // distances
    // TODO HOTRG_2point_manualみたいにする

    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;
    const string dir = "../data/gauss_quadrature/HOTRG_mass_manual/beta" + ss.str() + "/N" + std::to_string(N) + "_node" + std::to_string(n_node) + "/D" + std::to_string(D_cut) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(K, D_cut, n_node, N, d, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 8; D_cut <= 32; D_cut += 4) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(K, D_cut, n_node, N, d, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    /* vs n_node */
//    for (n_node = 8; n_node <= 32; n_node += 8) {
//        time.start();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(K, D_cut, n_node, N, d, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
