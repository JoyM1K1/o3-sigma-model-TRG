#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <spherical_harmonics.hpp>
#include <HOTRG.hpp>
#include <tensor.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

string duration_cast_to_string(std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end) {
    long long hours = std::chrono::duration_cast<std::chrono::hours>(end - start).count();
    long long minutes = std::chrono::duration_cast<std::chrono::minutes>(end - start).count() % 60;
    long long seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count() % 60;
    long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() % 1000;
    string res;
    if (hours > 0) {
        res = std::to_string(hours) + " h " + std::to_string(minutes) + " m " + std::to_string(seconds) + " s " + std::to_string(milli) + " ms";
    } else if (minutes > 0) {
        res = std::to_string(minutes) + " m " + std::to_string(seconds) + " s " + std::to_string(milli) + " ms";
    } else if (seconds > 0) {
        res = std::to_string(seconds) + " s " + std::to_string(milli) + " ms";
    } else {
        res = std::to_string(milli) + " ms";
    }
    return res;
}

void Trace(double const K, int const D_cut, int const l_max, int const N, std::ofstream &file) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    // initialize tensor network : max index size is D_cut
    cout << "initialize tensor " << std::flush;
    std::chrono::system_clock::time_point initStart = std::chrono::system_clock::now();
    Tensor T(D_cut);
    SphericalHarmonics::initTensor(K, l_max, T);
    std::chrono::system_clock::time_point initEnd = std::chrono::system_clock::now();
    cout << duration_cast_to_string(initStart, initEnd) << " : ";

    auto order = new int[N];
    int Dx = D_cut, Dy = D_cut;

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
//        cout << '\t' << std::fixed << std::setprecision(10) << Tr << '\n' << std::flush;
    }
    delete[] order;
    file << '\n';
    cout << '\n';
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    cout << "計算時間 : " << duration_cast_to_string(start, end) << '\n';
}

int main() {
    /* inputs */
    MKL_INT N = 40;     // volume : 2^N
    MKL_INT l_max;  // l_max
    MKL_INT D_cut; // bond dimension
    double K_start = 0.1;
    double K_end = 4.01;
    double K; // inverse temperature

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    for (l_max = 4; l_max <= 5; ++l_max) {
        start = std::chrono::system_clock::now();
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
        end = std::chrono::system_clock::now();
        cout << "合計計算時間 : " << duration_cast_to_string(start, end) << "\n\n";
    }

    return 0;
}
