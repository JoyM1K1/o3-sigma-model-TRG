#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <map>
#include <vector>
#include <mkl.h>
#include <fstream>
#include <frac.hpp>
#include <CG.hpp>
#include <TRG.hpp>
#include <tensor.hpp>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1
#define LINF 1e300
#define CGFileName "clebsch_gordan.txt"

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void normalization(const int n, int *order, Tensor &T) {
    const int D = T.GetDx(); // same as T.GetDx()
    double _min = LINF;
    double _max = 0;
    REP4(i, j, k, l, D) {
                    double t = std::abs(T(i, j, k, l));
                    if (t > 0) {
                        _min = std::min(_min, t);
                        _max = std::max(_max, t);
                    }
                }
    auto o = static_cast<int>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    REP4(i, j, k, l, D) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    order[n - 1] = o;
}

void Trace(double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N, std::map<CG, frac> &map, std::ofstream &file) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    // initialize tensor network : max index size is D_cut
    Tensor T(D_cut);
    std::ofstream CGFile;
    CGFile.open(CGFileName, std::ios::app);
    Tensor::initSphericalHarmonics(K, l_max, T, map, CGFile);
    CGFile.close();

    auto order = new int[N];

    for (int n = 1; n <= N; ++n) {
        normalization(n, order, T);

        TRG::solver(D_cut, T);

        double Tr = 0;
        int D = T.GetDx(); // same as T.GetDy()
        REP(i, D)REP(j, D) Tr += T(i, j, i, j);
        Tr = std::log(Tr);
        REP(i, n) Tr /= 2; // 体積で割る
        REP(i, n) {
            double tmp = order[i] * std::log(10);
            REP(j, i) tmp /= 2;
            Tr += tmp;
        }
        Tr += std::log(M_PI / (2 * K));
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
    MKL_INT D_cut; // bond dimension
    MKL_INT N = 20;     // volume : 2^N
    MKL_INT l_max; // max l
    double K_start;
    double K_end;

    /* Clebsch-Gordan coefficient */
    std::map<CG, frac> map;
    std::ifstream CGFile;
    CGFile.open(CGFileName, std::ios::in);
    int l1, l2, m1, m2, L, M, num, den;
    while (CGFile >> l1 >> l2 >> m1 >> m2 >> L >> M >> num >> den) {
        if (map.find(CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))) != map.end()) {
            cerr << "clebsch_gordan.txt is broken." << '\n';
            return 1;
        }
        map[CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))] = frac(num, den);
    }
    CGFile.close();

    /* calculation */
    for (l_max = 1; l_max <= 3; ++l_max) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        cout << "---------- " << l_max << " ----------\n";
        const string fileName = "spherical_harmonics_l" + std::to_string(l_max) + "_N" + std::to_string(N) + ".txt";
        std::ofstream dataFile;
        dataFile.open(fileName, std::ios::trunc);
        D_cut = (l_max + 1) * (l_max + 1);
        K_start = 0.1;
        K_end = 4.01;
        double K; // inverse temperature
        for (K = K_start; K <= K_end; K += MESH) {
            cout << "K = " << std::fixed << std::setprecision(1) << K << std::flush;
            dataFile << std::setprecision(1) << K;
            Trace(K, D_cut, l_max, N, map, dataFile);
        }
        dataFile.close();
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
    }

    return 0;
}
