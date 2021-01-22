#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <TRG.hpp>
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
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    /* distance */
    long long int distance = 1;
    REP(i, merge_point - 1) distance *= 2;

    // initialize tensor network : max index size is D_cut
    TRG::Tensor T1, T2;
    TRG::ImpureTensor IMTs[MAX_IMT_NUM];
    TRG::initialize_gauss_quadrature_with_impure(T1, T2, IMTs, D, D_cut, beta, n_node, merge_point);

    /* orders */
    long long int orders[DIMENSION];
    for (auto &order : orders) order = 0;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        TRG::renormalization::two_point(T1, T2, IMTs, orders, N, n, merge_point, NORMALIZE_FACTOR);

        if (n < N) {
            time.end();
            cout << " in " << time.duration_cast_to_string() << '\n';
            continue;
        }

        double res[DIMENSION];
        TRG::renormalization::trace(T1, IMTs[0], orders, NORMALIZE_FACTOR, res);
        double sum = res[0] + res[1] + res[2];

        cout << '\t' << std::scientific << std::setprecision(16) << sum << std::flush;
        file << distance << '\t' << std::scientific << std::setprecision(16) << sum << '\n' << std::flush;

        time.end();
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 14;     // volume : 2^N
    int n_node = 16;  // n_node
    int D_cut = 16; // bond dimension
    double beta = 1.90; // inverse temperature
    int merge_point = 7; // d = 2^(merge_point - 1)

    if (argc == 6) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        beta = std::stod(argv[4]);
        merge_point = std::stoi(argv[5]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << beta;
    const string dir = "../data/gauss_quadrature/TRG_2point/beta" + ss.str()
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
