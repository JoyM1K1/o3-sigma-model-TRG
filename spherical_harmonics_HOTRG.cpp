#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <spherical_harmonics.hpp>
#include <HOTRG.hpp>
#include <tensor.hpp>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, int const D_cut, int const l_max, int const N, std::ofstream &file) {
    time_counter time;

    // initialize tensor network : max index size is D_cut
    cout << "initialize tensor " << std::flush;
    time.start();
    Tensor T(D_cut);
    SphericalHarmonics::initTensor(K, l_max, T);
    time.end();
    cout << "in " << time.duration_cast_to_string() << " : " << std::flush;

    auto order = new int[N];
    int Dx = D_cut, Dy = D_cut;
    time.start();

    for (int n = 1; n <= N; ++n) {
        order[n - 1] = Tensor::normalization(T);

        if (n % 2) { // compression along x-axis
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compression along y-axis
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        double Tr = 0;
        REP(i, Dx) {
            REP(j, Dy) {
                Tr += T(i, j, i, j);
            }
        }
        Tr = std::log(Tr);
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, n) {
            double tmp = order[i] * std::log(10);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        Tr += std::log(M_PI / (2 * K));
        file << '\t' << std::fixed << std::setprecision(16) << Tr;
        cout << '\t' << std::fixed << std::setprecision(16) << Tr << std::flush;
    }
    delete[] order;
    file << '\n';
    time.end();
    cout << "  in " << time.duration_cast_to_string() << '\n';
}

int main() {
    /* inputs */
    MKL_INT N = 40;     // volume : 2^N
    MKL_INT l_max;  // l_max
    MKL_INT D_cut; // bond dimension

    double K_start = 0.1;
    double K_end = 4.01;
    double K; // inverse temperature

    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    for (l_max = 4; l_max <= 5; ++l_max) {
        time.start();
        cout << "---------- " << l_max << " ----------\n" << std::flush;
        fileName = "spherical_harmonics_HOTRG_l" + std::to_string(l_max) + "_N" + std::to_string(N) + ".txt";
        dataFile.open(fileName, std::ios::trunc);
        D_cut = (l_max + 1) * (l_max + 1);
        K = K_start;
        while (K <= K_end) {
            cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
            dataFile << std::fixed << std::setprecision(1) << K;
            Trace(K, D_cut, l_max, N, dataFile);
            K += MESH;
        }
        dataFile.close();
        time.end();
        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
    }

    return 0;
}
