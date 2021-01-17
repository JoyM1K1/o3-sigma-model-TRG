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

#define MESH 1e-1
#define NORMALIZE_FACTOR 10

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(const int n_data_point, double const K, int const D_cut, int const n_node, int const N, std::ofstream &file) {
    time_counter time;
    // index dimension
    int D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    HOTRG::Tensor T(D, D_cut);
    HOTRG::ImpureTensor originIMT(D, D_cut);
    GaussQuadrature::init_tensor_with_impure(K, n_node, D_cut, D, T, originIMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;

    std::vector<HOTRG::ImpureTensor> IMTs(n_data_point);

    int Dx = D, Dy = D;

    bool isMerged = false;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << std::setw(std::to_string(N).length()) << n << " :" << std::flush;

        T.normalization(NORMALIZE_FACTOR);
        for (auto &IMT : IMTs) {
            if (IMT.isMerged) {
                for (auto &tensor : IMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
            }
        }
        if (!isMerged) {
            for (auto &tensor : originIMT.tensors) tensor.normalization(NORMALIZE_FACTOR);
        }

        if (n <= N / 2) { // compress along x-axis
            cout << " compress along x-axis " << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            if (n <= n_data_point) {
                int d = 1;
                REP(i, n - 1) d *= 2;
                IMTs[n - 1] = HOTRG::ImpureTensor(d, originIMT);
                IMTs[n - 1].isMerged = true;
                for (int i = 0; i < 3; ++i) {
                    HOTRG::contractionX(D_cut, IMTs[n - 1].tensors[i], originIMT.tensors[i], U, "left");
                }
                for (int i = 0; i < n - 1; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
                for (auto &tensor : originIMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            } else {
                isMerged = true;
                for (int i = 0; i < n_data_point; ++i) {
                    for (auto &tensor : IMTs[i].tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along y-axis
            cout << " compress along y-axis " << std::flush;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (int i = 0; i < n_data_point; ++i) {
                for (auto &tensor : IMTs[i].tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            }
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        if (!isMerged) {
            time.end();
            cout << " in " << time.duration_cast_to_string() << '\n';
            continue;
        }

        double Tr = T.trace();

        for (auto &IMT : IMTs) {
            double Tr1 = IMT.tensors[0].trace(), Tr2 = IMT.tensors[1].trace(), Tr3 = IMT.tensors[2].trace();
            double res = (Tr1 + Tr2 + Tr3) / Tr;
            IMT.corrs.push_back(res);
            cout << '\t' << std::scientific << std::setprecision(16) << res << std::flush;
        }
        time.end();
        cout << "  in " << time.duration_cast_to_string() << '\n';
    }
    for (auto &IMT : IMTs) {
        cout << IMT.distance;
        for (double corr : IMT.corrs) {
            file << '\t' << std::scientific << std::setprecision(16) << corr << std::flush;
        }
        file << '\n';
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    int N = 40;     // volume : 2^N
    int n_node = 48;  // n_node
    int D_cut = 12; // bond dimension
    double K = 1.9; // inverse temperature
    int n_data_point = 7; // number of d. d = 1, 2, 4, 8, 16, 32, 64, ...

    N = std::stoi(argv[1]);
    n_node = std::stoi(argv[2]);
    D_cut = std::stoi(argv[3]);
    K = std::stod(argv[4]);
    n_data_point = std::stoi(argv[5]);

    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;
    const string dir = "../data/gauss_quadrature/HOTRG_2point/beta" + ss.str() + "/N" + std::to_string(N) + "_node" + std::to_string(n_node) + "/";
    time_counter time;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    time.start();
    cout << "N = " << N << ", node = " << n_node << ", D_cut = " << D_cut << ", beta = " << K << ", n_data_point = " << n_data_point << '\n';
    fileName = dir + "D" + std::to_string(D_cut) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(n_data_point, K, D_cut, n_node, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 44; D_cut <= 60; D_cut += 8) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(n_data_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    /* vs n_node */
//    for (n_node = 8; n_node <= 32; n_node += 8) {
//        time.start();
//        cout << "---------- " << n_node << " ----------\n";
//        fileName = dir + "D" + std::to_string(D_cut) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(n_data_point, K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
