#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
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

void Trace(double const K, int const D_cut, int const l_max, int const N, std::pair<int, int> p, std::ofstream &file) {
    time_counter time;

    const int x = p.first;
    const int y = p.second;

    // initialize tensor network : max index size is D_cut
    HOTRG::Tensor T;
    HOTRG::ImpureTensor originIMT;
    HOTRG::initialize_spherical_harmonics_with_impure(T, originIMT, K, D_cut, l_max);
    auto IMT = originIMT;

    long long int orders[DIMENSION];
    for (auto &order : orders) order = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        double res[DIMENSION];
        HOTRG::renormalization::two_point_manual(T, originIMT, IMT, orders, n, p, NORMALIZE_FACTOR, res);
        double sum = res[0] - res[1] + res[2];

        IMT.corrs.push_back(sum);
        cout << '\t' << std::scientific << std::setprecision(16) << sum << std::flush;
        time.end();
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
    file << x << '\t' << y;
    for (double corr : IMT.corrs) {
        file << '\t' << std::scientific << std::setprecision(16) << corr << std::flush;
    }
    file << '\n';
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 14;     // volume : 2^N
    int l_max = 1;  // l_max
    int D_cut; // bond dimension
    double K = 1.90; // inverse temperature
    std::pair<int, int> p(2, 0); // impure tensorの座標

    if (argc == 6) {
        N = std::stoi(argv[1]);
        l_max = std::stoi(argv[2]);
        K = std::stod(argv[3]);
        p.first = std::stoi(argv[4]);
        p.second = std::stoi(argv[5]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << K;
    const string dir = "../data/spherical_harmonics/HOTRG_2point_manual/beta" + ss.str() + "/N" + std::to_string(N) + "/l" + std::to_string(l_max) + "/data/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", l_max = " << l_max << ", beta = " << ss.str() << ", impure tensor coordinate = (" << p.first << "," << p.second << ")" << '\n' << std::flush;
    fileName = dir + std::to_string(p.first) + "-" + std::to_string(p.second) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    D_cut = (l_max + 1) * (l_max + 1);
    Trace(K, D_cut, l_max, N, p, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs l_max */
//    for (l_max = 1; l_max <= 4; ++l_max) {
//        time.start();
//        cout << "---------- " << l_max << " ----------\n";
//        fileName = dir + "l" + std::to_string(l_max) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        D_cut = (l_max + 1) * (l_max + 1);
//        Trace(K, D_cut, l_max, N, d, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
