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

#define MESH 1e-1
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, int const D_cut, int const n_node, int const N, std::ofstream &file) {
    time_counter time;
    std::stringstream ss;
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    TRG::Tensor T1(D, D_cut); /* (ij)(kl) -> S1 S3 */
    TRG::Tensor T2(D, D_cut); /* (jk)(li) -> S2 S4 */
    T1.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
    T2.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
    GaussQuadrature::initTensor(K, n_node, D_cut, T1);
    time.end();
    cout << "in " << time.duration_cast_to_string() << " : " << std::flush;

    time.start();
    auto orders = new long long int[N];

    for (int n = 1; n <= N; ++n) {
        const int D_new = std::min(D * D, D_cut);

        /* normalization */
        orders[n - 1] = T1.normalization(NORMALIZE_FACTOR);

        /* SVD */
        T2 = T1;
        TRG::SVD(D, D_new, T1, true);
        TRG::SVD(D, D_new, T2, false);

        /* contraction */
        TRG::contraction(D, D_new, T1, T1.S.first, T2.S.first, T1.S.second, T2.S.second);

        double Tr = T1.trace();
        if (std::isnan(std::log(Tr))) {
            cout << "\nTrace is " << Tr;
            exit(1);
        }
        Tr = std::log(Tr);
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, n) {
            double tmp = orders[i] * std::log(NORMALIZE_FACTOR);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        file << '\t' << std::fixed << std::setprecision(16) << Tr;
        cout << '\t' << std::fixed << std::setprecision(16) << Tr << std::flush;
    }
    delete T1.S.first;
    delete T1.S.second;
    delete T2.S.first;
    delete T2.S.second;
    delete[] orders;
    file << '\n';
    time.end();
    cout << "  in " << time.duration_cast_to_string() << '\n';
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 20;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 16; // bond dimension
    double K_start = 0.1;
    double K_end = 4.0;
    double K_interval = 0.1;
    double K; // inverse temperature

    if (argc == 7) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        K_start = std::stod(argv[4]);
        K_end = std::stod(argv[5]);
        K_interval = std::stod(argv[6]);
    }

    const string dir = "../data/gauss_quadrature/TRG/N" + std::to_string(N) + "_node" + std::to_string(n_node) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << K_start << "-" << K_end << " (" << K_interval << " step)" <<  '\n';
    K_end += K_interval / 2; // 誤差対策
    fileName = dir + "D" + std::to_string(D_cut) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    K = K_start;
    while (K <= K_end) {
        cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
        dataFile << std::setprecision(1) << K;
        Trace(K, D_cut, n_node, N, dataFile);
        K += K_interval;
    }
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";

    return 0;
}
