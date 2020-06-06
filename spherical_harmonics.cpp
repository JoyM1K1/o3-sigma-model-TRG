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

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

#define CGFileName "clebsch_gordan.txt"

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void Trace(double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N,
             std::map<CG, frac> &map, std::ofstream& file) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // index dimension
    MKL_INT D = D_cut;

    double A[l_max];
    REP(i, l_max) {
        A[i] = std::cyl_bessel_i(i + 0.5, K) * (i * 2 + 1);
    }

    // initialize tensor network : max index size is D_cut
    std::vector<std::vector<std::vector<std::vector<double>>>> T(D_cut, std::vector<std::vector<std::vector<double>>>(D_cut, std::vector<std::vector<double>>(D_cut, std::vector<double>(D_cut))));
    REP4(i, j, k, l, l_max) {
                    for (int im = 0; im <= 2 * i; ++im)
                        for (int jm = 0; jm <= 2 * j; ++jm)
                            for (int km = 0; km <= 2 * k; ++km)
                                for (int lm = 0; lm <= 2 * l; ++lm) {
                                    double sum = 0;
                                    for (int L = std::abs(i - j); L <= i + j; ++L)
                                        for (int M = -L; M <= L; ++M) {
                                            if (L < std::abs(k - l) || k + l < L || im - i + jm - j != M || km - k + lm - l != M)
                                                continue; // CG係数としてありえないものは0なので飛ばす
                                            frac c(1);
                                            c *= CG::getCoeff(frac(i), frac(j), frac(im - i), frac(jm - j), frac(L),
                                                          frac(M), map, CGFileName);
                                            c *= CG::getCoeff(frac(i), frac(j), frac(0), frac(0), frac(L), frac(0), map, CGFileName);
                                            c *= CG::getCoeff(frac(k), frac(l), frac(km - k), frac(lm - l), frac(L),
                                                          frac(M), map, CGFileName);
                                            c *= CG::getCoeff(frac(k), frac(l), frac(0), frac(0), frac(L), frac(0), map, CGFileName);
                                            c /= frac(2 * L + 1).sign() * (2 * L + 1) * (2 * L + 1);
                                            sum += c.sign().toDouble() * std::sqrt(frac::abs(c).toDouble());
                                        }
                                    T[i * i + im][j * j + jm][k * k + km][l * l + lm] =
                                            std::sqrt(A[i] * A[j] * A[k] * A[l]) * sum;
                                }
                }

    MKL_INT order[N];

    for (int n = 1; n <= N; ++n) {
        // Tを 1以上 に丸め込む
        double _min = std::abs(T[0][0][0][0]);
        REP4(i, j, k, l, D) {
                        if (std::abs(T[i][j][k][l]) > 0)
                            _min = std::min(_min, std::abs(T[i][j][k][l]));
                    }
        auto o = static_cast<MKL_INT>(std::floor(std::log10(_min)));
        REP4(i, j, k, l, D) {
                        REP(t, std::abs(o)) {
                            if (o > 0) {
                                T[i][j][k][l] /= 10;
                            } else {
                                T[i][j][k][l] *= 10;
                            }
                        }
                    }
        order[n-1] = o;

        TRG::solver(D, D_cut, T);

        double Tr = 0;
        REP(i, D)
            REP(j, D) {
                Tr += T[i][j][i][j];
            }
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
    MKL_INT l1, l2, m1, m2, L, M, num, den;
    while (CGFile >> l1 >> l2 >> m1 >> m2 >> L >> M >> num >> den) {
        if (map.find(CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))) != map.end()) {
            cerr << "clebsch_gordan.txt is broken." << '\n';
            return 1;
        }
        map[CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))] = frac(num, den);
    }
    CGFile.close();

    /* loop */
    for (l_max = 4; l_max <= 6; ++l_max) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        cout << "---------- " << l_max << " ----------\n";
        const string fileName = "spherical_harmonics_l" + std::to_string(l_max) + "_N" + std::to_string(N) + ".txt";
        std::ofstream dataFile;
        dataFile.open(fileName, std::ios::trunc);
        D_cut = (l_max + 1) * (l_max + 1);
        K_start = 0.1;
        K_end = 4.01;
        double K = K_start; // inverse temperature
        while (K <= K_end) {
            cout << "K = " << std::fixed << std::setprecision(1) << K << std::flush;
            dataFile << std::setprecision(1) << K;
            Trace(K, D_cut, l_max, N, map, dataFile);
            K += MESH;
        }
        dataFile.close();
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
    }

    return 0;
}
