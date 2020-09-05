#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <tensor.hpp>
#include <impure_tensor.hpp>
#include <HOTRG.hpp>
#include <time_counter.hpp>
#include <sstream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N)REP(j, N)REP(k, N)REP(l, N)

#define MESH 1e-1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, std::ofstream &file) {
    time_counter time;
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    time.start();
    cout << "initialize tensor " << std::flush;
    Tensor T(D, D_cut, N);
    ImpureTensor originIMT(D, D_cut, N);
    GaussQuadrature::initTensorWithImpure(K, n_node, D_cut, D, T, originIMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;


    MKL_INT Dx = D, Dy = D;

    for (int n = 1; n <= N; ++n) {
        time.start();
        cout << "N = " << (n < 10 ? " " : "") << n << " :" << std::flush;

        T.normalization(n - 1);
        for (Tensor &tensor : originIMT.tensors) {
            tensor.normalization(n - 1);
        }

        if (n % 2) { // compress along x-axis
            cout << " compress along x-axis " << std::flush;
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            for (Tensor &tensor : originIMT.tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compress along y-axis
            cout << " compress along y-axis " << std::flush;
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            for (auto &tensor : originIMT.tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        double Tr = 0;
        REP(i, Dx)REP(j, Dy) Tr += T(i, j, i, j);

        double impure_Tr[3];
        REP(k, 3) {
            impure_Tr[k] = 0;
            int order = 0;
            REP(i, Dx)REP(j, Dy) {
                    impure_Tr[k] += originIMT.tensors[k](i, j, i, j);
                }
            REP(i, n) order += originIMT.tensors[k].GetOrder()[i] - T.GetOrder()[i];
            const int times = std::abs(order);
            if (order > 0) {
                REP(i, times) impure_Tr[k] *= 10;
            } else {
                REP(i, times) impure_Tr[k] /= 10;
            }
        }
        time.end();
        file << std::scientific << std::setprecision(16) << impure_Tr[0] / Tr << '\t' << impure_Tr[1] / Tr << '\t' << impure_Tr[2] / Tr << '\n' << std::flush;
        cout << std::scientific << std::setprecision(16) << impure_Tr[0] / Tr << '\t' << impure_Tr[1] / Tr << '\t' << impure_Tr[2] / Tr
             << "  in " << time.duration_cast_to_string() << '\n' << std::flush;
    }
}

int main(int argc, char *argv[]) {
    /* inputs */
    MKL_INT N = 40;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 16; // bond dimension
    double K = 1.8; // inverse temperature

    N = std::stoi(argv[1]);
    n_node = std::stoi(argv[2]);
    D_cut = std::stoi(argv[3]);
    K = std::stod(argv[4]);

    const string dir = "gauss_quadrature_HOTRG_1point";
    time_counter time;
    string fileName;
    std::ofstream dataFile;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << K;

    /* calculation */
    time.start();
    fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    Trace(K, D_cut, n_node, N, dataFile);
    dataFile.close();
    time.end();
    cout << "合計計算時間 : " << time.duration_cast_to_string() << '\n';

    /* vs D_cut */
//    for (D_cut = 16; D_cut <= 56; D_cut += 8) {
//        time.start();
//        cout << "---------- " << D_cut << " ----------\n";
//        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
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
//        fileName = dir + "_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + "_beta" + ss.str() + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        Trace(K, D_cut, n_node, N, dataFile);
//        dataFile.close();
//        time.end();
//        cout << "合計計算時間 : " << time.duration_cast_to_string() << "\n\n";
//    }

    return 0;
}
