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

void Trace(double const K, int const D_cut, int const l_max, int const N, std::ofstream &file) {
    time_counter time;

    // initialize tensor network : max index size is D_cut
    HOTRG::Tensor T;
    HOTRG::ImpureTensor IMT;
    HOTRG::initialize_spherical_harmonics_with_impure(T, IMT, K, D_cut, l_max);

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
    int l_max = 2;  // l_max
    int D_cut; // bond dimension
    double K = 1.80; // inverse temperature

    if (argc == 4) {
        N = std::stoi(argv[1]);
        l_max = std::stoi(argv[2]);
        K = std::stod(argv[3]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << K;
    const string dir = "../data/spherical_harmonics/HOTRG_1point_alt/beta" + ss.str()
                       + "/N" + std::to_string(N) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N
         << ", l_max = " << l_max
         << ", beta = " << ss.str()
         << '\n' << std::flush;
    fileName = dir + "l" + std::to_string(l_max) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    D_cut = (l_max + 1) * (l_max + 1);
    Trace(K, D_cut, l_max, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    return 0;
}
