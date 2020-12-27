#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
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
        file << '\t' << std::scientific << std::setprecision(16) << Tr;
        cout << '\t' << std::scientific << std::setprecision(16) << Tr << std::flush;
    }
    file << '\n';
    time.end();
    cout << "  in " << time.duration_cast_to_string() << '\n';
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 20;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 8; // bond dimension
    double K = 0.10; // inverse temperature

    if (argc == 5) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        K = std::stod(argv[4]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << K;
    const string dir = "../data/gauss_quadrature/HOTRG_alt/N" + std::to_string(N) + "/node" + std::to_string(n_node) + "/D" + std::to_string(D_cut) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << ss.str() << '\n';
    fileName = dir + "beta" + ss.str() + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    dataFile << std::fixed << std::setprecision(2) << K;
    Trace(K, D_cut, n_node, N, dataFile);
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
