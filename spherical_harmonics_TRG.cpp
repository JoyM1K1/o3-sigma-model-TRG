#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <TRG.hpp>
#include <time_counter.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int N, const int l_max, const int D_cut, const double beta, std::ofstream &file) {
    time_counter time;

    // initialize tensor network : max index size is D_cut
    TRG::Tensor T1, T2;
    TRG::initialize_spherical_harmonics(T1, T2, D_cut, D_cut, beta, l_max);

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
    int l_max = 3; // max l
    int D_cut; // bond dimension
    double beta = 0.01; // inverse temperature

    if (argc == 4) {
        N = std::stoi(argv[1]);
        l_max = std::stoi(argv[2]);
        beta = std::stod(argv[3]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << beta;
    const string dir = "../data/spherical_harmonics/TRG/N" + std::to_string(N)
                       + "/l" + std::to_string(l_max) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N
         << ", l_max = " << l_max
         << ", beta = " << ss.str()
         << '\n' << std::flush;
    fileName = dir + "beta" + ss.str() + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    D_cut = (l_max + 1) * (l_max + 1);
    dataFile << std::fixed << std::setprecision(2) << beta;
    Trace(N, l_max, D_cut, beta, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    return 0;
}
