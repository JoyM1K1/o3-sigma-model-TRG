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

void Trace(const int N, const int n_node, const int D_cut, const double beta, std::pair<int, int> p, std::ofstream &file) {
    time_counter time;

    const int x = p.first;
    const int y = p.second;

    // initialize tensor network : max index size is D_cut
    HOTRG::Tensor T;
    HOTRG::ImpureTensor originIMT;
    HOTRG::initialize_gauss_quadrature_with_impure(T, originIMT, beta, D_cut, n_node);
    auto IMT = originIMT;

    long long int orders[DIMENSION];
    for (auto &order : orders) order = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        double res[DIMENSION];
        HOTRG::renormalization::two_point_manual(T, originIMT, IMT, orders, n, p, NORMALIZE_FACTOR, res);
        double sum = res[0] + res[1] + res[2];

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
    int n_node = 16;  // n_node
    int D_cut = 16; // bond dimension
    double beta = 1.90; // inverse temperature
    std::pair<int, int> p(2, 0); // impure tensorの座標

    if (argc == 7) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        beta = std::stod(argv[4]);
        p.first = std::stoi(argv[5]);
        p.second = std::stoi(argv[6]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << beta;
    const string dir = "../data/gauss_quadrature/HOTRG_2point_manual/beta" + ss.str()
                       + "/N" + std::to_string(N)
                       + "/node" + std::to_string(n_node)
                       + "/D" + std::to_string(D_cut) + "/data/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N
         << ", node = " << n_node
         << ", D_cut = " << D_cut
         << ", beta = " << ss.str()
         << ", impure tensor coordinate = (" << p.first << "," << p.second << ")"
         << '\n' << std::flush;
    fileName = dir + std::to_string(p.first) + "-" + std::to_string(p.second) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(N, n_node, D_cut, beta, p, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    return 0;
}
