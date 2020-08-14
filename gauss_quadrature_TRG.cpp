#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <TRG.hpp>
#include <tensor.hpp>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, std::ofstream &file) {
    time_counter time;
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    Tensor T(D, D, D_cut, D_cut);
    GaussQuadrature::initTensor(K, n_node, D_cut, T);
    time.end();
    cout << "in " << time.duration_cast_to_string() << " : " << std::flush;

    auto order = new int[N];
    time.start();

    for (int n = 1; n <= N; ++n) {
        order[n - 1] = Tensor::normalization(T);

        TRG::solver(D_cut, T);

        double Tr = 0;
        REP(i, D)REP(j, D) {
                Tr += T(i, j, i, j);
            }
        Tr = std::log(Tr);
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, n) {
            double tmp = order[i] * std::log(10);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        Tr += std::log(M_PI / 8);
        file << '\t' << std::fixed << std::setprecision(16) << Tr;
        cout << '\t' << std::fixed << std::setprecision(16) << Tr << std::flush;
    }
    file << '\n';
    time.end();
    cout << "  in " << time.duration_cast_to_string() << '\n';
}

int main() {
    /* inputs */
    MKL_INT N = 10;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 16; // bond dimension

    double K_start = 0.1;
    double K_end = 4.01;
    double K = K_start; // inverse temperature

    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    fileName = "gauss_quadrature_TRG_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    while (K <= K_end) {
        cout << "K = " << std::fixed << std::setprecision(1) << K << std::flush;
        dataFile << std::setprecision(1) << K;
        Trace(K, D_cut, n_node, N, dataFile);
        K += MESH;
    }
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";

    return 0;
}
