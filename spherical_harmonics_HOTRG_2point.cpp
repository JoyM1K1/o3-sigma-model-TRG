#include <iostream>
#include <iomanip>
#include <string>
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
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int n_data_point, double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N, std::ofstream &file) {
    time_counter time;

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D_cut);
    HOTRG::ImpureTensor originIMT(D_cut);
    SphericalHarmonics::initTensorWithImpure(K, l_max, T, originIMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    std::vector<HOTRG::ImpureTensor> IMTs(n_data_point);

    MKL_INT Dx = D_cut, Dy = D_cut;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        T.normalization(NORMALIZE_FACTOR);
        for (auto & IMT : IMTs) {
            if (IMT.isMerged) {
                for (auto & tensor : IMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
            }
        }
        if (!isMerged) {
            for (auto & tensor : originIMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
        }

        if (n <= N / 2) { // compress along x-axis
            cout << " compress along x-axis " << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            if (n <= n_data_point) {
                int d = 1;
                REP(i, n - 1) d *= 2;
                IMTs[n - 1] = HOTRG::ImpureTensor(d, originIMT);
                IMTs[n - 1].isMerged = true;
                for (int i = 0; i < 3; ++i) {
                    HOTRG::contractionX(D_cut, IMTs[n - 1].tensors[i], originIMT.tensors[i], U, "left");
                }
                for (int i = 0; i < n - 1; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                for (auto &tensor : originIMT.tensors) {
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
            cout << " compress along y-axis " << std::flush;
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
            time.end();
            cout << "in " << time.duration_cast_to_string() << '\n';
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
            double res = (impure_Tr[0] - impure_Tr[1] + impure_Tr[2]) / Tr;
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

int main() {
    /* inputs */
    MKL_INT N = 40;     // volume : 2^N
    MKL_INT l_max;  // l_max
    MKL_INT D_cut; // bond dimension
    double K = 1.9; // inverse temperature
    int n_data_point = 7; // number of d. d = 1, 2, 4, 8, 16, ...

    const string dir = "spherical_harmonics_HOTRG_2point";
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
        Trace(n_data_point, K, D_cut, l_max, N, dataFile);
        dataFile.close();
        time.end();
        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
    }

    return 0;
}
