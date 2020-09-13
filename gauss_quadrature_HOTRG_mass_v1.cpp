#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <mkl.h>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <tensor.hpp>
#include <impure_tensor.hpp>
#include <HOTRG.hpp>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N)REP(j, N)REP(k, N)REP(l, N)

#define MESH 1e-1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int merge_t_point, double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, std::ofstream &file) {
    time_counter time;
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    Tensor T(D, D_cut, N);
    ImpureTensor originIMT(D, D_cut, N);
    GaussQuadrature::initTensorWithImpure(K, n_node, D_cut, D, T, originIMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    ImpureTensor IMT;

    MKL_INT Dx = D, Dy = D;

    int mergeTCount = 0;
    int mergeXCount = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        T.normalization(n - 1);
        if (IMT.isMerged) {
            for (Tensor &tensor : IMT.tensors) tensor.normalization(n - 1);
        } else {
            for (Tensor &tensor : originIMT.tensors) tensor.normalization(n - 1);
        }

        if ((n % 2 && mergeTCount < merge_t_point - 1) || mergeXCount == N / 2) { // compress along t-axis
            cout << " compress along t-axis " << std::flush;
            mergeTCount++;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            if (mergeTCount < merge_t_point) {
                for (Tensor &tensor : originIMT.tensors) {
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
                ImpureTensor imt1 = IMT;
                ImpureTensor imt2 = IMT;
                for (int a = 0; a < 3; ++a) {
                    HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
                    HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
                    IMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
                    IMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                        *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
                    });
                }
            } else {
                ImpureTensor imt1 = originIMT;
                ImpureTensor imt2 = originIMT;
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
            REP(i, Dx)REP(j, Dy) {
                    impure_Tr[k] += IMT.tensors[k](i, j, i, j);
                }
            REP(i, n) {
                int m = IMT.tensors[k].GetOrder()[i] - T.GetOrder()[i];
                if (i < IMT.mergeIndex) m *= 2;
                order += m;
            }
            const int t = std::abs(order);
            if (order > 0) {
                REP(i, t) impure_Tr[k] *= 10;
            } else {
                REP(i, t) impure_Tr[k] /= 10;
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
    MKL_INT N = 14;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 16; // bond dimension
    double K = 1.8; // inverse temperature
    int merge_t_point = 7; // d = 2^(merge_t_point - 1)

//    N = std::stoi(argv[1]);
//    n_node = std::stoi(argv[2]);
//    D_cut = std::stoi(argv[3]);
//    K = std::stod(argv[4]);
//    merge_t_point = std::stoi(argv[5]);

    const string dir = "gauss_quadrature_HOTRG_mass_v1";
    time_counter time;
    string fileName;
    std::ofstream dataFile;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << K << ", merge_point = " << merge_t_point << '\n';
    fileName = dir + "_N" + std::to_string(N) + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_beta" + ss.str() + "_" +
               std::to_string(merge_t_point) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(merge_t_point, K, D_cut, n_node, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 16; D_cut <= 64; D_cut += 8) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "_N" + std::to_string(N) + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_beta" + ss.str() + "_" + std::to_string(merge_t_point) + "alpha.txt";
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
//        fileName = dir + "_N" + std::to_string(N) + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_beta" + ss.str() + "_" + std::to_string(merge_t_point) + "alpha.txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(merge_t_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }
    return 0;
}
