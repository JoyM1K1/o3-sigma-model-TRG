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

/* mergeする直前でx方向(縦方向)のcontractionを取り切るversion */

void Trace(const int merge_t_point, double const K, int const D_cut, int const n_node, int const N, std::ofstream &file) {
    time_counter time;
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

    HOTRG::ImpureTensor IMT;

    int Dx = D, Dy = D;

    int mergeTCount = 0;
    int mergeXCount = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        T.normalization(NORMALIZE_FACTOR);
        if (IMT.isMerged) {
            for (auto &tensor : IMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
        } else {
            for (auto &tensor : originIMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
        }

        if ((n % 2 && mergeTCount < merge_t_point - 1) || mergeXCount == N / 2) { // compress along t-axis
            cout << " compress along t-axis " << std::flush;
            mergeTCount++;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            if (mergeTCount < merge_t_point) {
                for (auto &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            } else if (mergeTCount == merge_t_point) {
                int d = 1;
                REP(i, mergeTCount - 1) d *= 2;
                cout << " d = " << d << " " << std::flush;
                IMT = originIMT;
                IMT.distance = d;
                IMT.isMerged = true;
                IMT.mergeIndex = n;
                for (int i = 0; i < 3; ++i) {
                    HOTRG::contractionX(D_cut, IMT.tensors[i], originIMT.tensors[i], U, "left");
                }
            } else {
                for (auto &tensor : IMT.tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along x-axis
            cout << " compress along x-axis " << std::flush;
            mergeXCount++;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            if (IMT.isMerged) {
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
            } else {
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

        double impure_Tr[3];
        REP(k, 3) {
            impure_Tr[k] = 0;
            int order = 0;
            REP(i, Dx)
                REP(j, Dy) {
                    impure_Tr[k] += IMT.tensors[k](i, j, i, j);
                }
            REP(i, T.orders.size()) {
                int m = IMT.tensors[k].orders[i] - T.orders[i];
                if (i < IMT.mergeIndex) m *= 2;
                order += m;
            }
            const int t = std::abs(order);
            if (order > 0) {
                REP(i, t) impure_Tr[k] *= NORMALIZE_FACTOR;
            } else {
                REP(i, t) impure_Tr[k] /= NORMALIZE_FACTOR;
            }
        }
        double res = (impure_Tr[0] + impure_Tr[1] + impure_Tr[2]) / Tr;
        IMT.corrs.push_back(res);
        cout << '\t' << std::scientific << std::setprecision(16) << res << std::flush;
        time.end();
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
    file << IMT.distance;
    for (double corr : IMT.corrs) {
        file << '\t' << std::scientific << std::setprecision(16) << corr << std::flush;
    }
    file << '\n';
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 14;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 16; // bond dimension
    double K = 1.8; // inverse temperature
    int merge_t_point = 3; // d = 2^(merge_t_point - 1)

    N = std::stoi(argv[1]);
    n_node = std::stoi(argv[2]);
    D_cut = std::stoi(argv[3]);
    K = std::stod(argv[4]);
    merge_t_point = std::stoi(argv[5]);

    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;
    const string dir = "../data/gauss_quadrature/HOTRG_mass_v1/beta" + ss.str() + "/N" + std::to_string(N) + "_node" + std::to_string(n_node) + "/D" + std::to_string(D_cut) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << K << ", merge_point = " << merge_t_point << '\n';
    fileName = dir + "D" + std::to_string(D_cut) + "_" + std::to_string(merge_t_point) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(merge_t_point, K, D_cut, n_node, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 16; D_cut <= 64; D_cut += 8) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + "_" + std::to_string(merge_t_point) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(merge_t_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    /* vs n_node */
//    for (n_node = 8; n_node <= 32; n_node += 8) {
//        time.start();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + "_" + std::to_string(merge_t_point) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(merge_t_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }
    return 0;
}
