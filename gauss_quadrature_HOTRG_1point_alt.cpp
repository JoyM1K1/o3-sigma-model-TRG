#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <HOTRG.hpp>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)

#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, int const D_cut, int const n_node, int const N, std::ofstream &file) {
    time_counter time;
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D, D_cut);
    HOTRG::ImpureTensor originIMT(D, D_cut);
    GaussQuadrature::initTensorWithImpure(K, n_node, D_cut, D, T, originIMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    /* orders */
    long long int orders[DIMENSION];
    for (auto & order : orders) order = 0;

    /* correlations */
    std::vector<double> correlations_list[DIMENSION];

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        if (n % 2) { // compress along x-axis
            cout << " compress along x-axis : " << std::flush;
            const int Dy = T.GetDy();
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            for (auto &tensor : originIMT.tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along y-axis
            cout << " compress along y-axis : " << std::flush;
            const int Dx = T.GetDx();
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (auto &tensor : originIMT.tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        /* normalization */
        T.normalization(NORMALIZE_FACTOR);
        for (auto &tensor : originIMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
        REP(i, DIMENSION) orders[i] += originIMT.tensors[i].order - T.order;

        double Tr = T.trace();

        double impureTrs[DIMENSION];
        REP(i, DIMENSION) {
            double impureTr = originIMT.tensors[i].trace();
            const long long int order = orders[i];
            const long long int absOrder = std::abs(order);
            if (order > 0) {
                REP(k, absOrder) impureTr *= NORMALIZE_FACTOR;
            } else {
                REP(k, absOrder) impureTr /= NORMALIZE_FACTOR;
            }
            impureTrs[i] = impureTr;
            correlations_list[i].push_back(impureTr/Tr);
        }
        time.end();
        cout << std::scientific << std::setprecision(16) << impureTrs[0] / Tr << '\t' << impureTrs[1] / Tr << '\t' << impureTrs[2] / Tr
             << "  in " << time.duration_cast_to_string() << '\n' << std::flush;
    }
    for (const auto& correlations : correlations_list) {
        for (auto correlation : correlations) file << std::scientific << std::setprecision(16) << correlation << '\t';
        file << '\n';
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 40;     // volume : 2^N
    int n_node = 32;  // n_node
    int D_cut = 16; // bond dimension
    double K = 1.80; // inverse temperature

    if (argc == 5) {
        N = std::stoi(argv[1]);
        n_node = std::stoi(argv[2]);
        D_cut = std::stoi(argv[3]);
        K = std::stod(argv[4]);
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << K;
    const string dir = "../data/gauss_quadrature/HOTRG_1point_alt/beta" + ss.str() + "/N" + std::to_string(N) + "/node" + std::to_string(n_node) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << ss.str() << '\n' << std::flush;
    fileName = dir + "D" + std::to_string(D_cut) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(K, D_cut, n_node, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 16; D_cut <= 56; D_cut += 8) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    /* vs n_node */
//    for (n_node = 48; n_node <= 64; n_node += 16) {
//        time.start();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}