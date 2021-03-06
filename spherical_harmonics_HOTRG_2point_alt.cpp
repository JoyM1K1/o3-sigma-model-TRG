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

void Trace(const int N, const int l_max, const int D_cut, const double beta, const int merge_point, std::ofstream &file) {
    time_counter time;

    /* distance */
    long long int distance = 1;
    REP(i, merge_point - 1) distance *= 2;
    file << distance;

    // initialize tensor network : max index size is D_cut
    HOTRG::Tensor T;
    HOTRG::ImpureTensor IMT;
    HOTRG::initialize_spherical_harmonics_with_impure(T, IMT, beta, D_cut, l_max);

    /* orders */
    long long int orders[DIMENSION];
    for (auto &order : orders) order = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        double res[DIMENSION];
        HOTRG::renormalization::two_point_alt(T, IMT, orders, n, merge_point, NORMALIZE_FACTOR, res);
        double sum = res[0] - res[1] + res[2];

        time.end();
        file << '\t' << std::scientific << std::setprecision(16) << sum << std::flush;
        cout << '\t' << std::scientific << std::setprecision(16) << sum << std::flush;
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 40;     // volume : 2^N
    int l_max = 1;  // l_max
    int D_cut; // bond dimension
    double beta = 1.90; // inverse temperature
    int merge_point = 4; // d = 2^(merge_point - 1)

    if (argc == 5) {
        N = std::stoi(argv[1]);
        l_max = std::stoi(argv[2]);
        beta = std::stod(argv[3]);
        merge_point = std::stoi(argv[4]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << beta;
    const string dir = "../data/spherical_harmonics/HOTRG_2point_alt/beta" + ss.str()
                       + "/N" + std::to_string(N)
                       + "/l" + std::to_string(l_max) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N
         << ", l_max = " << l_max
         << ", beta = " << ss.str()
         << ", merge_point = " << merge_point
         << '\n' << std::flush;
    fileName = dir + std::to_string(merge_point) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    D_cut = (l_max + 1) * (l_max + 1);
    Trace(N, l_max, D_cut, beta, merge_point, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    return 0;
}
