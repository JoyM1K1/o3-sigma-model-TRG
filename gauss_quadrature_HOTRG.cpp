#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <gauss_quadrature.hpp>
#include <HOTRG.hpp>
#include <tensor.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, MKL_INT const D_cut, MKL_INT const n_node, MKL_INT const N, std::ofstream &file) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // index dimension
    MKL_INT D = std::min(D_cut, n_node * n_node);

    // initialize tensor network : max index size is D_cut
    Tensor T(D, D, D_cut, D_cut);
    GaussQuadrature::initTensor(K, n_node, D_cut, T);

    auto order = new int[N];
    MKL_INT Dx = D, Dy = D;

    for (int n = 1; n <= N; ++n) {
        order[n - 1] = Tensor::normalization(T);

        if (n <= N / 2) { // compression along x-axis
            auto U = new double[Dy * Dy * Dy * Dy];
            HOTRG::SVD_Y(D_cut, T, U);
            HOTRG::contractionX(D_cut, T, T, U, "left");
            delete[] U;
        } else { // compression along y-axis
            auto U = new double[Dx * Dx * Dx * Dx];
            HOTRG::SVD_X(D_cut, T, U);
            HOTRG::contractionY(D_cut, T, T, U, "bottom");
            delete[] U;
        }

        Dx = T.GetDx();
        Dy = T.GetDy();

        double Tr = 0;
        REP(i, Dx) {
            REP(j, Dy) {
                Tr += T(i, j, i, j);
            }
        }
        Tr = std::log(Tr);
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, n) {
            double tmp = order[i] * std::log(10);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        Tr += std::log(M_PI / 8);
        file << '\t' << std::fixed << std::setprecision(10) << Tr;
        cout << '\t' << std::fixed << std::setprecision(10) << Tr << std::flush;
    }
    delete[] order;
    file << '\n';
    cout << '\n';
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    cout << "計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << '\n';
}

int main() {
    /* inputs */
    MKL_INT N = 20;     // volume : 2^N
    MKL_INT n_node = 32;  // n_node
    MKL_INT D_cut = 16; // bond dimension

    double K_start = 0.1;
    double K_end = 4.01;
    double K = K_start; // inverse temperature

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    string fileName;
    std::ofstream dataFile;

    /* calculation */
    start = std::chrono::system_clock::now();
    fileName = "gauss_quadrature_HOTRG_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
    dataFile.open(fileName, std::ios::trunc);
    while (K <= K_end) {
        cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
        dataFile << std::setprecision(1) << K;
        Trace(K, D_cut, n_node, N, dataFile);
        K += MESH;
    }
    dataFile.close();
    end = std::chrono::system_clock::now();
    cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    /* vs D_cut */
//    for (D_cut = 8; D_cut <= 24; D_cut += 4) {
//        K = K_start;
//        start = std::chrono::system_clock::now();
//        fileName =
//                "gauss_quadrature_HOTRG_node" + std::to_string(n_node) + "_D" + std::to_string(D_cut) + "_N" + std::to_string(N) + ".txt";
//        dataFile.open(fileName, std::ios::trunc);
//        while (K <= K_end) {
//            cout << "K = " << std::fixed << std::setprecision(1) << K << " : " << std::flush;
//            dataFile << std::setprecision(1) << K;
//            Trace(K, D_cut, n_node, N/*, dataFile*/);
//            K += MESH;
//        }
//        dataFile.close();
//        end = std::chrono::system_clock::now();
//        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
//    }

    return 0;
}
