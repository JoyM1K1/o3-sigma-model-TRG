#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <HOTRG.hpp>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define MESH 1e-1
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int n_data_point_start, const int n_data_point_end, double const K, int const D_cut, int const n_node, int const N, std::ofstream &file) {
    time_counter time;
    const int n_data_point = n_data_point_end - n_data_point_start + 1;
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D, D_cut);
    HOTRG::ImpureTensor originIMT(D, D_cut);
    GaussQuadrature::initTensorWithImpure(K, n_node, D_cut, D, T, originIMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    std::vector<HOTRG::ImpureTensor> IMTs(n_data_point);

    int Dx = D, Dy = D;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        T.normalization(NORMALIZE_FACTOR);
        for (auto &IMT : IMTs) {
            if (IMT.isMerged) {
                for (auto &tensor : IMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
            }
        }
        if (!isMerged) {
            for (auto &tensor : originIMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
        }

        if (n > N / 2) { // compress along x-axis
            cout << " compress along x-axis " << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            const int times = n - N / 2;
            if (times < n_data_point_start) {
                for (auto &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            } else if (times <= n_data_point_end) {
                int d = 1;
                REP(i, times - 1) d *= 2;
                IMTs[times - n_data_point_start] = HOTRG::ImpureTensor(d, originIMT);
                IMTs[times - n_data_point_start].isMerged = true;
                IMTs[times - n_data_point_start].mergeIndex = n;
                for (int i = 0; i < 3; ++i) {
                    HOTRG::contractionX(D_cut, IMTs[times - n_data_point_start].tensors[i], originIMT.tensors[i], U, "left");
                }
                for (int i = 0; i < times - n_data_point_start; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                for (auto &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                if (times == n_data_point_end) isMerged = true;
            } else {
                for (auto &IMT : IMTs) {
                    for (auto &tensor : IMT.tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along y-axis
            cout << " compress along y-axis " << std::flush;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (auto &IMT : IMTs) {
                if (!IMT.isMerged) continue;
                auto imt1 = IMT;
                auto imt2 = IMT;
                for (int a = 0; a < 3; ++a) {
                    HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
                    HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
                    IMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
                    IMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                        *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
                    });
                }
            }
            if (!isMerged) {
                auto imt1 = originIMT;
                auto imt2 = originIMT;
                for (int a = 0; a < 3; ++a) {
                    HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
                    HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
                    originIMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
                    originIMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                        *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
                    });
                }
            }
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        if (n < N) {
            time.end();
            cout << " in " << time.duration_cast_to_string() << '\n';
            continue;
        }

        double Tr = 0;
        REP(i, Dx)REP(j, Dy) Tr += T(i, j, i, j);

        for (auto &IMT : IMTs) {
            double impure_Tr[3];
            REP(k, 3) {
                impure_Tr[k] = 0;
                int order = 0;
                REP(i, Dx)REP(j, Dy) {
                        impure_Tr[k] += IMT.tensors[k](i, j, i, j);
                    }
                REP(i, T.orders.size()) {
                    int m = IMT.tensors[k].orders[i] - T.orders[i];
                    if (i < IMT.mergeIndex) m *= 2;
                    order += m;
                }
                const int times = std::abs(order);
                if (order > 0) {
                    REP(i, times) impure_Tr[k] *= NORMALIZE_FACTOR;
                } else {
                    REP(i, times) impure_Tr[k] /= NORMALIZE_FACTOR;
                }
            }
            double res = (impure_Tr[0] + impure_Tr[1] + impure_Tr[2]) / Tr;
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
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 14;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 16; // bond dimension
    double K = 1.8; // inverse temperature
    int n_data_point_start = 1; // d = 2^(n_data_point_start - 1), ..., 2^(n_data_point_end - 1)
    int n_data_point_end = 7;

    N = std::stoi(argv[1]);
    n_node = std::stoi(argv[2]);
    D_cut = std::stoi(argv[3]);
    K = std::stod(argv[4]);
    n_data_point_start = std::stoi(argv[5]);
    n_data_point_end = std::stoi(argv[6]);

    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;
    const string dir = "../data/gauss_quadrature/HOTRG_mass/beta" + ss.str() + "/N" + std::to_string(N) + "_node" + std::to_string(n_node) + "/D" + std::to_string(D_cut) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << K << ", " << n_data_point_start << '-' << n_data_point_end << '\n';
    fileName = dir + "D" + std::to_string(D_cut) + "_" + std::to_string(n_data_point_start) + "-" + std::to_string(n_data_point_end) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(n_data_point_start, n_data_point_end, K, D_cut, n_node, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 16; D_cut <= 64; D_cut += 8) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + "_" + std::to_string(n_data_point_start) + "-" + std::to_string(n_data_point_end) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(n_data_point_start, n_data_point_end, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    /* vs n_node */
//    for (n_node = 8; n_node <= 32; n_node += 8) {
//        time.start();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + "_" + std::to_string(n_data_point_start) + "-" + std::to_string(n_data_point_end) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(n_data_point_start, n_data_point_end, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }
    return 0;
}
