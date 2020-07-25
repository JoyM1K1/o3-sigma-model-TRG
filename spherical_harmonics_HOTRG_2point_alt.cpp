#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <HOTRG.hpp>
#include <tensor.hpp>
#include <impure_tensor.hpp>
#include <spherical_harmonics.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int n_data_point, double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N,std::ofstream &file) {
    // initialize tensor network : max index size is D_cut
    Tensor T(D_cut);
    ImpureTensor originIMT(D_cut);

    SphericalHarmonics::initTensorWithImpure(K, l_max, T, originIMT);

    std::vector<ImpureTensor> IMTs(n_data_point);

    auto order = new int[N];
    MKL_INT Dx = D_cut, Dy = D_cut;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        order[n - 1] = ImpureTensor::normalization(T, originIMT, IMTs);

        if (n % 2) { // compress along x-axis
            cout << " compress along x-axis :" << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            const int times = (n + 1) / 2;
            if (times <= n_data_point) {
                int d = 1;
                REP(i, times - 1) d *= 2;
                IMTs[times - 1] = ImpureTensor(d, originIMT);
                IMTs[times - 1].isMerged = true;
                for (int i = 0; i < 3; ++i) {
                    HOTRG::contractionX(D_cut, IMTs[times - 1].tensors[i], originIMT.tensors[i], U, "left");
                }
                for (int i = 0; i < times - 1; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                for (Tensor &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                if (times == n_data_point) isMerged = true;
            } else {
                for (auto &IMT : IMTs) {
                    for (auto &tensor : IMT.tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along y-axis
            cout << " compress along y-axis :" << std::flush;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (auto &IMT : IMTs) {
                if (!IMT.isMerged) continue;
                for (auto &tensor : IMT.tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            }
            if (!isMerged) {
                for (auto &tensor : originIMT.tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            }
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        if (n <= N / 2) {
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
    MKL_INT N = 40;     // volume : 2^N
    MKL_INT l_max;  // l_max
    MKL_INT D_cut; // bond dimension
    double K = 1.9; // inverse temperature
    const int n_data_point = 7; // number of d. d = 1, 2, 4, 8, 16, ...

    /* calculation */
    for (l_max = 1; l_max <= 6; ++l_max) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        cout << "---------- " << l_max << " ----------\n" << std::flush;
        const string fileName = "new_2point_spherical_harmonics_l" + std::to_string(l_max) + "_N" + std::to_string(N) + ".txt";
        std::ofstream dataFile;
        dataFile.open(fileName, std::ios::trunc);
        D_cut = (l_max + 1) * (l_max + 1);
        Trace(n_data_point, K, D_cut, l_max, N, dataFile);
        dataFile.close();
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
    }

    return 0;
}
