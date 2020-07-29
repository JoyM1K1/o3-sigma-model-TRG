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

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N)REP(j, N)REP(k, N)REP(l, N)

#define MESH 1e-1
#define LINF 1e300

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int n_data_point, double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N) {
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    Tensor T(D, D, D_cut, D_cut);
    ImpureTensor originIMT(D, D, D_cut, D_cut);

    GaussQuadrature::initTensorWithImpure(K, n_node, D_cut, D, T, originIMT);

    std::vector<ImpureTensor> IMTs(n_data_point);

    auto order = new int[N];
    MKL_INT Dx = D, Dy = D;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
//        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
//        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        order[n - 1] = ImpureTensor::normalization(T, originIMT, IMTs);

        if (n <= N / 2) { // compress along x-axis
//            cout << " compress along x-axis :" << std::flush;
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
//            cout << " compress along y-axis :" << std::flush;
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
//            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
//            cout << " 計算時間 " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << '\n';
            continue;
        }

        double Tr = 0;
        REP(i, Dx)REP(j, Dy) Tr += T(i, j, i, j);

        for (ImpureTensor &IMT : IMTs) {
            double Tr1 = 0, Tr2 = 0, Tr3 = 0;
            REP(i, Dx)REP(j, Dy) {
                    Tr1 += IMT.tensors[0](i, j, i, j);
                    Tr2 += IMT.tensors[1](i, j, i, j);
                    Tr3 += IMT.tensors[2](i, j, i, j);
                }
            double res = (Tr1 + Tr2 + Tr3) / Tr;
            IMT.corrs.push_back(res);
//            cout << '\t' << std::fixed << std::setprecision(10) << res << std::flush;
        }
//        cout << '\n';
    }
    for (ImpureTensor &IMT : IMTs) {
        cout << IMT.distance;
        for (double corr : IMT.corrs) {
            cout << '\t' << std::fixed << std::setprecision(10) << corr << std::flush;
        }
        cout << '\n';
    }
    delete[] order;
}

int main() {
    /* inputs */
    MKL_INT N = 40;     // volume : 2^N
    MKL_INT n_node = 48;  // n_node
    MKL_INT D_cut = 12; // bond dimension
    double K = 1.9; // inverse temperature
    int n_data_point = 7; // number of d. d = 1, 2, 4, 8, 16, 32, 64, ...

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;

    /* calculation */
//    start = std::chrono::system_clock::now();
//    fileName = "new_2point_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
//    dataFile.open(fileName, std::ios::trunc);
//    Trace(n_data_point, K, D_cut, n_node, N, dataFile);
//    dataFile.close();
//    end = std::chrono::system_clock::now();
//    cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    /* vs D_cut */
    for (D_cut = 44; D_cut <= 64; D_cut += 4) {
//        start = std::chrono::system_clock::now();
        cout << "---------- " << D_cut << " ----------\n";
        Trace(n_data_point, K, D_cut, n_node, N);
    }

    /* vs n_node */
//    for (n_node = 8; n_node <= 32; n_node += 8) {
//        start = std::chrono::system_clock::now();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = "2point_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(n_data_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        end = std::chrono::system_clock::now();
//        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
//    }
    return 0;
}
