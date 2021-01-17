#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <HOTRG.hpp>
#include <cmath>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, int const D_cut, int const n_node, int const N, std::pair<int, int> p, std::ofstream &file) {
    time_counter time;
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    const int x = p.first;
    const int y = p.second;

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D, D_cut);
    HOTRG::ImpureTensor originIMT(D, D_cut);
    GaussQuadrature::init_tensor_with_impure(K, n_node, D_cut, D, T, originIMT);
    auto IMT = originIMT;
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    long long int orders[DIMENSION];
    for (auto &order : orders) order = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        if (n % 2) { // compression along x-axis
            cout << " compress along x-axis " << std::flush;
            cout << std::setw(7);
            const int Dy = T.GetDy();
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            if (p.first) {
                if (p.first == 1) {
                    if (p.second) {
                        cout << "right" << std::flush;
                        for (auto &tensor : IMT.tensors) {
                            HOTRG::contractionX(D_cut, T, tensor, U, "right");
                        }
                    } else {
                        cout << "merged" << std::flush;
                        REP(i, DIMENSION) {
                            HOTRG::contractionX(D_cut, originIMT.tensors[i], IMT.tensors[i], U, "right");
                        }
                        IMT.isMerged = true;
                    }
                } else if (p.first & 1) {
                    cout << "right" << std::flush;
                    for (auto &tensor : IMT.tensors) {
                        HOTRG::contractionX(D_cut, T, tensor, U, "right");
                    }
                } else {
                    cout << "left" << std::flush;
                    for (auto &tensor : IMT.tensors) {
                        HOTRG::contractionX(D_cut, tensor, T, U, "left");
                    }
                }
                p.first >>= 1;
            } else {
                cout << "left" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            if (!IMT.isMerged) {
                for (auto &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compression along y-axis
            cout << " compress along y-axis " << std::flush;
            cout << std::setw(7);
            const int Dx = T.GetDx();
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            if (p.second) {
                if (p.second == 1) {
                    if (p.first) {
                        cout << "top" << std::flush;
                        for (auto &tensor : IMT.tensors) {
                            HOTRG::contractionY(D_cut, T, tensor, U, "top");
                        }
                    } else {
                        cout << "merged" << std::flush;
                        REP(i, DIMENSION) {
                            HOTRG::contractionY(D_cut, originIMT.tensors[i], IMT.tensors[i], U, "top");
                        }
                        IMT.isMerged = true;
                    }
                } else if (p.second & 1) {
                    cout << "top" << std::flush;
                    for (auto &tensor : IMT.tensors) {
                        HOTRG::contractionY(D_cut, T, tensor, U, "top");
                    }
                } else {
                    cout << "bottom" << std::flush;
                    for (auto &tensor : IMT.tensors) {
                        HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
                    }
                }
                p.second >>= 1;
            } else {
                cout << "bottom" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
                }
            }
            if (!IMT.isMerged) {
                for (auto &tensor : originIMT.tensors) {
                    HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
                }
            }
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        /* normalization */
        T.normalization(NORMALIZE_FACTOR);
        if (p.first || p.second) {
            REP(i, DIMENSION) {
                originIMT.tensors[i].normalization(NORMALIZE_FACTOR);
                orders[i] += originIMT.tensors[i].order - T.order;
            }
        }
        REP(i, DIMENSION) {
            IMT.tensors[i].normalization(NORMALIZE_FACTOR);
            orders[i] += IMT.tensors[i].order - T.order;
        }

        double Tr = T.trace();

        double impure_Tr[DIMENSION];
        REP(k, DIMENSION) {
            long long int order = orders[k];
            double tmp_Tr = IMT.tensors[k].trace();
            unsigned long long int absOrder = std::abs(order);
            if (order > 0) {
                REP(i, absOrder) tmp_Tr *= NORMALIZE_FACTOR;
            } else {
                REP(i, absOrder) tmp_Tr /= NORMALIZE_FACTOR;
            }
            impure_Tr[k] = tmp_Tr;
        }
        double res = (impure_Tr[0] + impure_Tr[1] + impure_Tr[2]) / Tr;
        IMT.corrs.push_back(res);
        cout << '\t' << std::scientific << std::setprecision(16) << res << std::flush;
        time.end();
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
    file << x << '\t' << y;
    for (double corr : IMT.corrs) {
        file << '\t' << std::scientific << std::setprecision(16) << corr << std::flush;
    }
    file << '\n';
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 14;     // volume : 2^N
    double K = 1.90; // inverse temperature
    int n_node = 16;  // n_node
    int D_cut = 16; // bond dimension
    std::pair<int, int> p(2, 0); // impure tensorの座標

    if (argc == 7) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        K = std::stod(argv[4]);
        p.first = std::stoi(argv[5]);
        p.second = std::stoi(argv[6]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << K;
    const string dir = "../data/gauss_quadrature/HOTRG_2point_manual/beta" + ss.str() + "/N" + std::to_string(N) + "/node" + std::to_string(n_node) + "/D" + std::to_string(D_cut) + "/data/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << ss.str() << ", impure tensor coordinate = (" << p.first << "," << p.second << ")" << '\n';
    fileName = dir + std::to_string(p.first) + "-" + std::to_string(p.second) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(K, D_cut, n_node, N, p, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 8; D_cut <= 32; D_cut += 4) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "N" + std::to_string(N) + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_beta" + ss.str() + "_" + std::to_string(p.first) + "-" + std::to_string(p.second) + ".txt";
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
//        fileName = dir + "N" + std::to_string(N) + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_beta" + ss.str() + "_" + std::to_string(p.first) + "-" + std::to_string(p.second) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(K, D_cut, n_node, N, d, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
