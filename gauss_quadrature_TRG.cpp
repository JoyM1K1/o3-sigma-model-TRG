#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <gauss_quadrature.hpp>
#include <TRG.hpp>
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

    /* initialize tensor network : max index size is D_cut */
    TRG::Tensor T1, T2;
    TRG::initialize_gauss_quadrature(T1, T2, D, D_cut, K, n_node);

    auto orders = new long long int[N];

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        double Tr = TRG::renormalization::partition(T1, T2, orders, n, NORMALIZE_FACTOR);

        time.end();
        file << '\t' << std::scientific << std::setprecision(16) << Tr;
        cout << '\t' << std::scientific << std::setprecision(16) << Tr << "  in " << time.duration_cast_to_string() << '\n' << std::flush;
    }
    delete T1.S.first;
    delete T1.S.second;
    delete T2.S.first;
    delete T2.S.second;
    delete[] orders;
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 20;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 8; // bond dimension
    double K = 0.01; // inverse temperature

    if (argc == 5) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        K = std::stod(argv[4]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << K;
    const string dir = "../data/gauss_quadrature/TRG/N" + std::to_string(N) + "/node" + std::to_string(n_node) + "/D" + std::to_string(D_cut) + "/";
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

    return 0;
}
