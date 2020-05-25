#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <map>
#include <vector>
#include <mkl.h>
#include <fstream>
#include "libraries/include/frac.hpp"
#include "libraries/include/CG.hpp"
#include "libraries/include/legendre_zero_point.hpp"

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define MESH 1e-1

#define CGFileName "clebsch_gordan.txt"

using namespace std;

frac squareSummation(std::vector<frac> &factors) {
    // 絶対値が同じで符号だけが異なるものは相殺するので消す
    bool flag = true;
    while (flag) {
        flag = false;
        for (int i = 0; i < factors.size(); ++i)
            for (int j = i + 1; j < factors.size(); ++j) {
                if (factors[i] + factors[j] == 0) {
                    factors.erase(factors.begin() + j);
                    factors.erase(factors.begin() + i);
                    flag = true;
                }
            }
    }
    std::vector<frac> test; // 係数の2乗
    for (auto &a : factors)
        for (auto &b : factors) {
            test.push_back(a * b);
        }
    frac sum;
    for (auto a : test) {
        if (frac::abs(a).isSquare()) {
            sum += a.sign() * frac::abs(a).cleanSquareRoot();
            continue;
        }
        std::cerr << "Error : summation failed.\n";
        exit(1);
    }
    double precise = 0.0;
    for (auto &factor : factors)
        precise += factor.sign().toDouble() * sqrt(frac::abs(factor).toDouble());
    if (precise < 0.0)
        sum *= -1;
    return sum;
}

void determineAllCGs(frac l1, frac l2, frac L, std::map<CG, frac> &map) {
    chrono::system_clock::time_point start = chrono::system_clock::now();
    cout << "determine : l1=" << l1 << " l2=" << l2 << " L=" << L << "   ";
    if (l1 + l2 < L) {
        cerr << "l1 + l2 < L" << '\n';
        return;
    }
    frac m1_max = min(l1, L + l2);
    frac m2_max = min(l2, L + l1);
    frac m1_min = -m1_max;
    frac m2_min = -m2_max;
    // CG(l1, l2, m1_max, L - m1_max, L, L) ... 一番右の一番上
    // をとりあえず 1 に設定 ここを基準とする
    map[CG(l1, l2, m1_max, L - m1_max, L, L)] = frac(1);
    // step1
    // m1 = m1_max でのCG係数を求める
    // L- : sqrt((L - M) * (L + M + 1)) * CG(l1, m1, l2, m2, L, M) = sqrt((l1 - m1) * (l1 + m1 + 1)) * CG(l1, m1 + 1, l2, m2, L, M + 1) + sqrt((l2 - m2) * (l2 + m2 + 1)) * CG(l1, m1, l2, m2 + 1, L, M + 1)
    for (frac m2 = L - m1_max - 1; m2 >= m2_min; --m2) {
        frac preFactorA;
        frac M = m1_max + m2;
        frac A = (L - M) * (L + M + 1);
        frac B(0);
        frac C = (l2 - m2) * (l2 + m2 + 1);
        if (A == 0) {
            std::cerr << "Error : A is 0" << '\n';
            exit(1);
        }
        frac preFactorC = map[CG(l1, l2, m1_max, m2 + 1, L, M + 1)];
        preFactorA = (C / A) * preFactorC;
        map[CG(l1, l2, m1_max, m2, L, M)] = preFactorA;
    }
    cout << "step1 OK  ";
    // step2
    // m1_min <= m1 <= m1_max | m2_min <= m2 <= L - m1_max でのCG係数を求める
    // L+ : sqrt((L + M) * (L - M + 1)) * CG(l1, m1, l2, m2, L, M) = sqrt((l1 + m1) * (l1 - m1 + 1)) * CG(l1, m1 - 1, l2, m2, L, M - 1) + sqrt((l2 + m2) * (l2 - m2 + 1)) * CG(l1, m1, l2, m2 - 1, L, M - 1)
    for (frac m1 = m1_max; m1 > m1_min; --m1)
        for (frac m2 = L - m1_max; m2 >= m2_min; --m2) {
            frac M = m1 + m2;
            if (frac::abs(M - 1) <= L) {
                std::vector<frac> preFactorsB;
                frac A = (L + M) * (L - M + 1);
                frac B = (l1 + m1) * (l1 - m1 + 1);
                frac C = (l2 + m2) * (l2 - m2 + 1);
                if (B == 0) {
                    std::cerr << "Error : B is 0" << '\n';
                    exit(1);
                }
                frac preFactorA = map[CG(l1, l2, m1, m2, L, M)];
                frac preFactorC;
                if (frac::abs(m2 - 1) <= l2)
                    preFactorC = map[CG(l1, l2, m1, m2 - 1, L, M - 1)];
                preFactorsB.push_back((A / B) * preFactorA);
                preFactorsB.push_back(-(C / B) * preFactorC);
                if (!preFactorsB.empty()) {
                    frac sum = squareSummation(preFactorsB);
                    map[CG(l1, l2, m1 - 1, m2, L, M - 1)] = sum;
                }
            }
        }
    cout << "step2 OK  ";
    // step3
    // m1 + m2 = M となる残りのCG係数を求める
    // L-
    for (frac m2 = L - m1_max; m2 <= m2_max; ++m2)
        for (frac m1 = m1_max - 1; m1 >= m1_min; --m1) {
            frac M = m1 + m2;
            if (frac::abs(M + 1) <= L && frac::abs(m1 + 1) <= l1 && frac::abs(m2 + 1) <= l2) {
                std::vector<frac> preFactorsC;
                frac A = (L - M) * (L + M + 1);
                frac B = (l1 - m1) * (l1 + m1 + 1);
                frac C = (l2 - m2) * (l2 + m2 + 1);
                if (C == 0) {
                    std::cerr << "Error : C is 0" << '\n';
                    exit(0);
                }
                frac preFactorA;
                if (frac::abs(M) <= L)
                    preFactorA = map[CG(l1, l2, m1, m2, L, M)];
                frac preFactorB = map[CG(l1, l2, m1 + 1, m2, L, M + 1)];
                preFactorsC.push_back((A / C) * preFactorA);
                preFactorsC.push_back(-(B / C) * preFactorB);
                if (!preFactorsC.empty()) {
                    frac sum = squareSummation(preFactorsC);
                    map[CG(l1, l2, m1, m2 + 1, L, M + 1)] = sum;
                }
            }
        }
    cout << "step3 OK  ";
    // step4
    // 規格化
    for (frac M = -L; M <= L; ++M) {
        frac sum;
        for (frac m1 = m1_min; m1 <= m1_max; ++m1) {
            frac m2 = M - m1;
            if (m2_min <= m2 && m2 <= m2_max) {
                sum += frac::abs(map[CG(l1, l2, m1, m2, L, M)]);
            }
        }
        if (sum == 0) {
            cerr << "step4 : summation error." << '\n';
            exit(1);
        }
//        cout << "sum:" << sum << ' ';
        std::ofstream CGFile;
        CGFile.open(CGFileName, std::ios::app);
        for (frac m1 = m1_min; m1 <= m1_max; ++m1) {
            frac m2 = M - m1;
            if (m2_min <= m2 && m2 <= m2_max) {
                map[CG(l1, l2, m1, m2, L, M)] /= sum;
                CGFile << l1 << '\t' << l2 << '\t' << m1 << '\t' << m2 << '\t' << L << '\t' << M << '\t'
                       << map[CG(l1, l2, m1, m2, L, M)].num << '\t' << map[CG(l1, l2, m1, m2, L, M)].den << '\n';
            }
        }
        CGFile.close();
    }
    chrono::system_clock::time_point end = chrono::system_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";
}

// sqrtの配列を足し合わせる
frac getCoeff(frac l1, frac l2, frac m1, frac m2, frac L, frac M, std::map<CG, frac> &map) {
    // CG係数としてありえないものは 0
    if (L < frac::abs(l1 - l2) || l1 + l2 < L || m1 + m2 != M) {
        return frac(0);
    }
    bool minus = false;
    if (l1 < l2) {
        frac tmp = l1;
        l1 = l2;
        l2 = tmp;
        tmp = m1;
        m1 = m2;
        m2 = tmp;
        minus = ((L - l1 - l2) % 2 != 0);
    }
    if (map.find(CG(l1, l2, m1, m2, L, M)) == map.end()) { // まだ計算してなかったら計算する
        determineAllCGs(l1, l2, L, map);
    }
    return map[CG(l1, l2, m1, m2, L, M)] * (minus ? 1 : -1);
}

// m x n 行列を出力
template<typename T>
void print_matrix(T *matrix, MKL_INT m, MKL_INT n, MKL_INT lda, const string &message) {
    cout << '\n' << message << '\n';
    REP(i, m) {
        REP(j, n) {
            cout << scientific << setprecision(5) << (matrix[i * lda + j] >= 0 ? " " : "") << matrix[i * lda + j]
                 << ' ';
        }
        cout << '\n';
    }
    cout << '\n';
}

double TRG(double const K, MKL_INT const D_cut, MKL_INT const l_max, MKL_INT const N, MKL_INT *order
        /*std::map<CG, frac> &map*/) {
    chrono::system_clock::time_point start = chrono::system_clock::now();
    // index dimension
    MKL_INT D = D_cut;

    std::vector<double> x = math::solver::legendre_zero_point(l_max);
    std::vector<double> p(l_max);
    REP(i, l_max) {
        p[i] = std::legendre(l_max - 1, x[i]);
    }

    // initialize tensor network : max index size is D_cut
    double T[D_cut][D_cut][D_cut][D_cut];
    REP4(i, j, k, l, D_cut) {
                    T[i][j][k][l] = 0;
                }
    std::vector<double> w(l_max);
    REP(i, l_max) {
        w[i] = 2 * (1 - x[i] * x[i]) / (l_max * l_max * p[i] * p[i]);
    }
    std::function<double(double, double, double, double)> f = [=](double t1, double u1, double t2, double u2) {
        return t1 * t2 + std::sqrt((1 - t1 * t1) * (1 - t2 * t2)) * std::cos(M_PI * (u1 - u2));
    };
    REP(i1, l_max)
        REP(j1, l_max)
            REP(i2, l_max)
                REP(j2, l_max)
                    REP(i3, l_max)
                        REP(j3, l_max)
                            REP(i4, l_max)
                                REP(j4, l_max) {
                                    T[l_max * i1 + j1][l_max * i2 + j2][l_max * i3 + j3][l_max * i4 + j4] =
                                            std::sqrt(w[i1] * w[j1] * w[i2] * w[j2] * w[i3] * w[j3] * w[i4] * w[j4])
                                            * std::exp(K * f(x[i1], x[j1], x[i2], x[j2]))
                                            * std::exp(K * f(x[i2], x[j2], x[i3], x[j3]))
                                            * std::exp(K * f(x[i3], x[j3], x[i4], x[j4]))
                                            * std::exp(K * f(x[i4], x[j4], x[i1], x[j1]));
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

        MKL_INT D_new = min(D * D, D_cut);
        double Ma[D * D * D * D], Mb[D * D * D * D]; // Ma = M(ij)(kl)  Mb = M(jk)(li)
        REP(i, D * D * D * D) {
            Ma[i] = 0;
            Mb[i] = 0;
        }
        REP4(i, j, k, l, min(D, D_cut)) {
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
            return 1;
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
            return 1;
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

        double T_new[D_cut][D_cut][D_cut][D_cut];

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
                        T_new[i][j][k][l] = 0;
                        REP(b, D) {
                            REP(d, D) {
                                T_new[i][j][k][l] += X12[k][l][b][d] * X34[i][j][b][d];
                            }
                        }
                    }

        // 更新
        D = D_new;
        REP4(i, j, k, l, D) {
                        T[i][j][k][l] = T_new[i][j][k][l];
                    }
    }

    double Z = 0;
    REP(i, D)REP(j, D) {
            Z += T[i][j][i][j];
        }

    chrono::system_clock::time_point end = chrono::system_clock::now();
    cout << "計算時間 : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << '\n';
    return Z;
}

int main() {
    /* inputs */
    MKL_INT D_cut; // bond dimension
    MKL_INT N;     // repeat count
    MKL_INT l_max; // max l
    double K_start;
    double K_end;

//    cout << "input max l : ";
//    cin >> l_max;
    cout << "input N : ";
    cin >> N;
    cout << '\n';

    /* Clebsch-Gordan coefficient */
//    std::map<CG, frac> map;
//    std::ifstream CGFile;
//    CGFile.open(CGFileName, std::ios::in);
//    MKL_INT l1, l2, m1, m2, L, M, num, den;
//    while (CGFile >> l1 >> l2 >> m1 >> m2 >> L >> M >> num >> den) {
//        if (map.find(CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))) != map.end()) {
//            cerr << "clebsch_gordan.txt is broken." << '\n';
//            return 1;
//        }
//        map[CG(frac(l1), frac(l2), frac(m1), frac(m2), frac(L), frac(M))] = frac(num, den);
//    }
//    CGFile.close();

    /* loop */
    for (l_max = 5; l_max <= 6; ++l_max) {
        chrono::system_clock::time_point start = chrono::system_clock::now();
        cout << "---------- " << l_max << " ----------\n";
        const string fileName = "gauss_quadrature_" + to_string(l_max) + '-' + to_string(N) + ".txt";
        std::ofstream dataFile;
        dataFile.open(fileName, std::ios::trunc);
        D_cut = l_max * l_max;
        K_start = 0.1;
        K_end = 4.01;
        double K = K_start; // inverse temperature
        while (K <= K_end) {
            cout << "K = " << K << "   ";
            MKL_INT order[2 * N - 1];
            double Z = log(TRG(K, D_cut, l_max, N * 2 - 1, order /*map*/));
            REP(i, N) Z /= 4; // 体積で割る
            double o = 0;
            REP(i, 2 * N - 1) {
                double tmp = order[i] * log(10);
                REP(j, i + 1) tmp /= 2;
                o += tmp;
            }
            Z += -log(4) + o;
//        Z = log(M_PI / (2 * K)) + order * log(THRESHOLD) / V + Z / V;
            dataFile << K << '\t' << -Z / K << '\n';
//        cout << "結果 : " << -Z/K << '\n';
            K += MESH;
        }
        dataFile.close();
        chrono::system_clock::time_point end = chrono::system_clock::now();
        cout << "合計計算時間 : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n\n";
    }

    return 0;
}
