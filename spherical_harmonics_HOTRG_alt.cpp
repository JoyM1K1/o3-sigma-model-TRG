#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <spherical_harmonics.hpp>
#include <HOTRG.hpp>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define MESH 1e-1
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, int const D_cut, int const l_max, int const N, std::ofstream &file) {
    time_counter time;

    // initialize tensor network : max index size is D_cut
    cout << "initialize tensor " << std::flush;
    time.start();
    HOTRG::Tensor T(D_cut);
    SphericalHarmonics::initTensor(K, l_max, T);
    time.end();
    cout << "in " << time.duration_cast_to_string() << " : " << std::flush;

    int Dx = D_cut, Dy = D_cut;
    time.start();

    for (int n = 1; n <= N; ++n) {
        T.normalization(NORMALIZE_FACTOR);

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

        double Tr = T.trace();
        Tr = std::log(Tr);
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, T.orders.size()) {
            double tmp = T.orders[i] * std::log(NORMALIZE_FACTOR);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        file << '\t' << std::fixed << std::setprecision(16) << Tr;
        cout << '\t' << std::fixed << std::setprecision(16) << Tr << std::flush;
    }
    file << '\n';
    time.end();
    cout << "  in " << time.duration_cast_to_string() << '\n';
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 40; // volume : 2^N
    int l_max;  // l_max
    int D_cut; // bond dimension

    double K_start = 0.1;
    double K_end = 4.01;
    double K; // inverse temperature

    N = std::stoi(argv[1]);
    l_max = std::stoi(argv[2]);

    const string dir = "../data/spherical_harmonics/HOTRG_alt/N" + std::to_string(N) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", l_max = " << l_max <<  '\n';
    fileName = dir + "l" + std::to_string(l_max) + ".txt";
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
    cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n";

    /* vs l_max */
//    for (l_max = 4; l_max <= 5; ++l_max) {
//        time.start();
//        cout << "---------- " << l_max << " ----------\n" << std::flush;
//        fileName = dir + "l" + std::to_string(l_max) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        D_cut = (l_max + 1) * (l_max + 1);
//        K = K_start;
//        while (K <= K_end) {
//            cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
//            dataFile << std::fixed << std::setprecision(1) << K;
//            Trace(K, D_cut, l_max, N, dataFile);
//            K += MESH;
//        }
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
