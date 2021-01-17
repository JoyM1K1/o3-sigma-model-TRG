#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <spherical_harmonics.hpp>
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
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D_cut);
    HOTRG::ImpureTensor originIMT(D_cut);
    SphericalHarmonics::init_tensor_with_impure(K, l_max, T, originIMT);
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
    const string dir = "../data/spherical_harmonics/HOTRG_1point_alt/beta" + ss.str() + "/N" + std::to_string(N) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", l_max = " << l_max << ", beta = " << ss.str() << '\n' << std::flush;
    fileName = dir + "l" + std::to_string(l_max) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    D_cut = (l_max + 1) * (l_max + 1);
    Trace(K, D_cut, l_max, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs l_max */
//    for (l_max = 4; l_max <= 6; ++l_max) {
//        time.start();
//        cout << "---------- " << l_max << " ----------\n" << std::flush;
//        fileName = dir + std::to_string(merge_point) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        D_cut = (l_max + 1) * (l_max + 1);
//        Trace(merge_point, K, D_cut, l_max, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    /* vs merge_point */
//    for (merge_point = 1; merge_point <= 20; ++merge_point) {
//        time.start();
//        cout << "---------- " << merge_point << " ----------\n" << std::flush;
//        fileName = dir + std::to_string(merge_point) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        D_cut = (l_max + 1) * (l_max + 1);
//        Trace(merge_point, K, D_cut, l_max, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
