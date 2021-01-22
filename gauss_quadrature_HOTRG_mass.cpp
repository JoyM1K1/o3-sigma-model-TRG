#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <HOTRG.hpp>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int N, const int n_node, const int D_cut, const double beta, const int merge_point, std::ofstream &file) {
    time_counter time;

    /* distance */
    long long int distance = 1;
    REP(i, merge_point - 1) distance *= 2;
    file << distance;

    // initialize tensor network : max index size is D_cut
    HOTRG::Tensor T;
    HOTRG::ImpureTensor IMT;
    HOTRG::initialize_gauss_quadrature_with_impure(T, IMT, beta, D_cut, n_node);

    /* orders */
    long long int orders[DIMENSION];
    for (auto &order : orders) order = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        double res[DIMENSION];
        HOTRG::renormalization::mass(T, IMT, orders, N, n, merge_point, NORMALIZE_FACTOR, res);
        double sum = res[0] + res[1] + res[2];

        time.end();
        file << '\t' << std::scientific << std::setprecision(16) << sum << std::flush;
        cout << '\t' << std::scientific << std::setprecision(16) << sum << std::flush;
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 14;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 16; // bond dimension
    double beta = 1.90; // inverse temperature
    int merge_point = 4; // d = 2^(merge_point - 1)

    if (argc == 6) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        beta = std::stod(argv[4]);
        merge_point = std::stoi(argv[5]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << beta;
    const string dir = "../data/gauss_quadrature/HOTRG_mass/beta" + ss.str()
                       + "/N" + std::to_string(N)
                       + "/node" + std::to_string(n_node)
                       + "/D" + std::to_string(D_cut) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N
         << ", node = " << n_node
         << ", D_cut = " << D_cut
         << ", beta = " << ss.str()
         << ", merge_point = " << merge_point
         << '\n' << std::flush;
    fileName = dir + std::to_string(merge_point) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(N, n_node, D_cut, beta, merge_point, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    return 0;
}
