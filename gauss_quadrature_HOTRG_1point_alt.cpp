#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
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

void Trace(const int N, const int n_node, const int D_cut, const double beta, std::ofstream &file) {
    time_counter time;

    // initialize tensor network : max index size is D_cut
    HOTRG::Tensor T;
    HOTRG::ImpureTensor IMT;
    HOTRG::initialize_gauss_quadrature_with_impure(T, IMT, beta, D_cut, n_node);

    /* orders */
    long long int orders[DIMENSION];
    for (auto &order : orders) order = 0;

    /* correlations */
    std::vector<double> correlations_list[DIMENSION];

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        double res[DIMENSION];
        HOTRG::renormalization::one_point_alt(T, IMT, orders, n, NORMALIZE_FACTOR, res);
        REP(i, DIMENSION) correlations_list[i].push_back(res[i]);

        time.end();
        cout << std::scientific << std::setprecision(16) << res[0] << '\t' << res[1] << '\t' << res[2]
             << "  in " << time.duration_cast_to_string() << '\n' << std::flush;
    }
    for (const auto &correlations : correlations_list) {
        for (auto correlation : correlations) file << std::scientific << std::setprecision(16) << correlation << '\t';
        file << '\n';
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 40;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 16; // bond dimension
    double beta = 1.80; // inverse temperature

    if (argc == 5) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        beta = std::stod(argv[4]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << beta;
    const string dir = "../data/gauss_quadrature/HOTRG_1point_alt/beta" + ss.str()
                       + "/N" + std::to_string(N)
                       + "/node" + std::to_string(n_node) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N
         << ", node = " << n_node
         << ", D_cut = " << D_cut
         << ", beta = " << ss.str()
         << '\n' << std::flush;
    fileName = dir + "D" + std::to_string(D_cut) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(N, n_node, D_cut, beta, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    return 0;
}
