#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <HOTRG.hpp>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, int const D_cut, int const n_node, int const N, std::ofstream &file) {
    time_counter time;

    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D, D_cut);
    GaussQuadrature::initTensor(K, n_node, D_cut, T);
    time.end();
    cout << "in " << time.duration_cast_to_string() << " : " << std::flush;

    int Dx = D, Dy = D;
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
    int N = 20;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 16; // bond dimension
    double K_start = 0.1;
    double K_end = 4.0;
    double K_interval = 0.1;
    double K; // inverse temperature

//    N = std::stoi(argv[1]);
//    n_node = std::stoi(argv[2]);
//    D_cut = std::stoi(argv[3]);
//    K_start = std::stod(argv[4]);
//    K_end = std::stod(argv[5]);
//    K_interval = std::stod(argv[6]);

    const string dir = "../data/gauss_quadrature/HOTRG_alt/N" + std::to_string(N) + "_node" + std::to_string(n_node) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", K = " << K_start << "-" << K_end << " (" << K_interval << " step)" <<  '\n';
    fileName = dir + "D" + std::to_string(D_cut) + ".txt";
    K_end += K_interval / 2; // 誤差対策
    dataFile.open(fileName, std::ios::trunc);
    K = K_start;
    while (K <= K_end) {
        cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
        dataFile << std::fixed << std::setprecision(1) << K;
        Trace(K, D_cut, n_node, N, dataFile);
        K += K_interval;
    }
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    K_end += K_interval / 2; // 誤差対策
//    for (D_cut = 8; D_cut <= 24; D_cut += 4) {
//        K = K_start;
//        time.start();
//        fileName = dir + "D" + std::to_string(D_cut) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        while (K <= K_end) {
//            cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
//            dataFile << std::fixed << std::setprecision(1) << K;
//            Trace(K, D_cut, n_node, N, dataFile);
//            K += K_interval;
//        }
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
