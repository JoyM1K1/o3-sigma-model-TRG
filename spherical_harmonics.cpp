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

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

#define CGFileName "clebsch_gordan.txt"

using std::cin;
using std::cout;
using std::cerr;
using std::string;

// m x n 行列を出力
template<typename T>
void print_matrix(T *matrix, MKL_INT m, MKL_INT n, MKL_INT lda, const string &message) {
    cout << '\n' << message << '\n';
    REP(i, m) {
        REP(j, n) {
            cout << std::scientific << std::setprecision(5) << (matrix[i * lda + j] >= 0 ? " " : "") << matrix[i * lda + j]
                 << ' ';
        }
        cout << '\n';
    }
    cout << '\n';
}

double TRG(double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N, MKL_INT *order,
           std::map<CG, frac> &map) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // index dimension
    MKL_INT D = D_cut;

    double A[l_max];
    REP(i, l_max) {
        A[i] = std::cyl_bessel_i(i + 0.5, K) * (i * 2 + 1);
    }

    // initialize tensor network : max index size is D_cut
    double T[D_cut][D_cut][D_cut][D_cut];
    REP4(i, j, k, l, D_cut) {
                    T[i][j][k][l] = 0;
                }
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

    REP(n, N) {
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
        order[n] = o;

        MKL_INT D_new = std::min(D * D, D_cut);
        double Ma[D * D * D * D], Mb[D * D * D * D]; // Ma = M(ij)(kl)  Mb = M(jk)(li)
        REP(i, D * D * D * D) {
            Ma[i] = 0;
            Mb[i] = 0;
        }
        REP4(i, j, k, l, std::min(D, D_cut)) {
                        Ma[l + D * k + D * D * j + D * D * D * i] = T[i][j][k][l];
                        Mb[i + D * l + D * D * k + D * D * D * j] = T[i][j][k][l];
                    }
        double sigma[D * D];
        double U[D * D * D * D], VH[D * D * D * D];
        double S1[D][D][D_new], S2[D][D][D_new], S3[D][D][D_new], S4[D][D][D_new];
        double superb[D * D];
        if (n < 0) {
            print_matrix(Ma, D * D, D * D, D * D, "Ma");
//            print_matrix(Mb, D * D, D * D, D * D, "Mb");
        }
        MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, Ma, D * D, sigma, U, D * D, VH, D * D,
                                      superb); // Ma = U * sigma * VH
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        if (n < 0) {
            print_matrix(sigma, 1, D * D, D * D, "sigma");
            print_matrix(U, D * D, D * D, D * D, "U");
            print_matrix(VH, D * D, D * D, D * D, "VH");
            double US[D * D * D * D];
            REP4(i, j, k, l, D) {
                            US[i + D * j + D * D * k + D * D * D * l] =
                                    U[i + D * j + D * D * k + D * D * D * l] * sigma[i + D * j];
                        }
            double USVH[D * D * D * D];
            REP(i, D * D * D * D) {
                USVH[i] = 0;
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, D * D, 1, US, D * D, VH, D * D, 0,
                        USVH, D * D);
            print_matrix(USVH, D * D, D * D, D * D, "U * sigma * VH");
        }
        REP(i, D) {
            REP(j, D) {
                REP(k, D_new) {
                    double s = sqrt(sigma[k]);
                    S1[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                    S3[i][j][k] = s * VH[j + D * i + D * D * k];
                }
            }
        }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, Mb, D * D, sigma, U, D * D, VH, D * D, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        if (n < 0) {
            print_matrix(sigma, 1, D * D, D * D, "sigma");
            print_matrix(U, D * D, D * D, D * D, "U");
            print_matrix(VH, D * D, D * D, D * D, "VH");
        }
        REP(i, D) {
            REP(j, D) {
                REP(k, D_new) {
                    double s = sqrt(sigma[k]);
                    S2[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                    S4[i][j][k] = s * VH[j + D * i + D * D * k];
                }
            }
        }

        double X12[D_new][D_new][D][D], X34[D_new][D_new][D][D];
        REP(i, D_new) {
            REP(j, D_new) {
                REP(b, D) {
                    REP(d, D) {
                        X12[i][j][b][d] = 0;
                        X34[i][j][b][d] = 0;
                        REP(a, D) {
                            X12[i][j][b][d] += S1[a][d][i] * S2[b][a][j];
                            X34[i][j][b][d] += S3[a][b][i] * S4[d][a][j];
                        }
                    }
                }
            }
        }

        REP4(i, j, k, l, D_new) {
                        T[i][j][k][l] = 0;
                        REP(b, D) {
                            REP(d, D) {
                                T[i][j][k][l] += X12[k][l][b][d] * X34[i][j][b][d];
                            }
                        }
                    }

        // 更新
        D = D_new;
    }

    double Z = 0;
    REP(i, D)
        REP(j, D) {
            Z += T[i][j][i][j];
        }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    cout << "計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << '\n';
    return Z;
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
    for (l_max = 1; l_max <= 1; ++l_max) {
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
            cout << "K = " << K << "   ";
            MKL_INT order[N];
            double Z = log(TRG(K, D_cut, l_max, N, order, map));
            REP(i, N) Z /= 2; // 体積で割る
            REP(i, N) {
                double tmp = order[i] * log(10);
                REP(j, i) tmp /= 2;
                Z += tmp;
            }
            Z += log(M_PI / (2 * K));
            dataFile << K << '\t' << -Z / K << '\n';
            K += MESH;
        }
        dataFile.close();
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        cout << "合計計算時間 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n\n";
    }

    return 0;
}
