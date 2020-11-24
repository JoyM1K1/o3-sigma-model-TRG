#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <spherical_harmonics.hpp>
#include <TRG.hpp>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define MESH 1e-1
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, int const D_cut, int const l_max, int const N, std::ofstream &file) {
    time_counter time;

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    TRG::Tensor T1(D_cut); /* (ij)(kl) -> S1 S3 */
    TRG::Tensor T2(D_cut); /* (jk)(li) -> S2 S4 */
    T1.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
    T2.S = std::make_pair(new TRG::Unitary_S(D_cut), new TRG::Unitary_S(D_cut));
    SphericalHarmonics::initTensor(K, l_max, T1);
    time.end();
    cout << "in " << time.duration_cast_to_string() << " : " << std::flush;

    time.start();
    auto orders = new long long int[N];

    for (int n = 1; n <= N; ++n) {
        /* normalization */
        T1.normalization(NORMALIZE_FACTOR);
        // TODO 応急処置
        orders[n - 1] = T1.order;

        /* SVD */
        T2 = T1;
        TRG::SVD(D_cut, D_cut, T1, true);
        TRG::SVD(D_cut, D_cut, T2, false);

        /* contraction */
        TRG::contraction(D_cut, D_cut, T1, T1.S.first, T2.S.first, T1.S.second, T2.S.second);

        double Tr = 0;
        int D = T1.GetDx(); // same as T.GetDy()
        REP(i, D)REP(j, D) Tr += T1(i, j, i, j);
        Tr = std::log(Tr) + std::log(NORMALIZE_FACTOR) * T1.order;
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, n) {
            double tmp = orders[i] * std::log(NORMALIZE_FACTOR);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        Tr += std::log(M_PI / (2 * K));
        file << '\t' << std::fixed << std::setprecision(16) << Tr;
        cout << '\t' << std::fixed << std::setprecision(16) << Tr << std::flush;
    }
    file << '\n';
    time.end();
    cout << "  in " << time.duration_cast_to_string() << '\n';
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 20;     // volume : 2^N
    int l_max; // max l
    int D_cut; // bond dimension

    double K_start = 0.1;
    double K_end = 4.01;
    double K; // inverse temperature

    N = std::stoi(argv[1]);
    l_max = std::stoi(argv[2]);

    const string dir = "../data/spherical_harmonics/TRG/N" + std::to_string(N) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", l_max = " << l_max <<  '\n';
    fileName = dir + "l" + std::to_string(l_max) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    D_cut = (l_max + 1) * (l_max + 1);
    K = K_start;
    while (K <= K_end) {
        cout << "K = " << std::fixed << std::setprecision(1) << K << std::flush;
        dataFile << std::fixed << std::setprecision(1) << K;
        Trace(K, D_cut, l_max, N, dataFile);
        K += MESH;
    }
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs l_max */
//    for (l_max = 1; l_max <= 3; ++l_max) {
//        time.start();
//        cout << "---------- " << l_max << " ----------\n";
//        fileName = dir + "l" + std::to_string(l_max) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        D_cut = (l_max + 1) * (l_max + 1);
//        K = K_start;
//        while (K <= K_end) {
//            cout << "K = " << std::fixed << std::setprecision(1) << K << std::flush;
//            dataFile << std::fixed << std::setprecision(1) << K;
//            Trace(K, D_cut, l_max, N, dataFile);
//            K += MESH;
//        }
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
