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

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N)REP(k, N) REP(l, N)

#define THRESHOLD 100000
#define MESH 1e-1

using namespace std;

void determineAllCGs(frac l1, frac l2, frac L, std::map<CG, std::vector<frac>> &map) {
    chrono::system_clock::time_point start = chrono::system_clock::now();
    cout << "determine : l1=" << l1 << " l2=" << l2 << " L=" << L << "   ";
    if (l1 + l2 < L) {
        cout << "l1 + l2 < L" << '\n';
        return;
    }
    // std::vector<frac>は平方根の中身の和を表す．
    // 例えば，(1/2, 3/4, -5/6) -> sqrt(1/2) + sqrt(3/4) - sqrt(5/6)
    std::map<CG, std::vector<frac>> coeffs;
    frac m1_max = min(l1, L + l2);
    frac m2_max = min(l2, L + l1);
    frac m1_min = -m1_max;
    frac m2_min = -m2_max;
    // CG(l1, l2, m1_max, L - m1_max, L, L) ... 一番右の一番上
    // をとりあえず 1 に設定 ここを基準とする
    coeffs[CG(l1, l2, m1_max, L - m1_max, L, L)].push_back(frac(1));
    // m1 = m1_max でのCG係数を求める
    // L- : sqrt((L - M) * (L + M + 1)) * CG(l1, m1, l2, m2, L, M) = sqrt((l1 - m1) * (l1 + m1 + 1)) * CG(l1, m1 + 1, l2, m2, L, M + 1) + sqrt((l2 - m2) * (l2 + m2 + 1)) * CG(l1, m1, l2, m2 + 1, L, M + 1)
    for (frac m2 = L - m1_max - 1; m2 >= m2_min; --m2) {
        std::vector<frac> preFactorsA;
        frac M = m1_max + m2;
        frac A = (L - M) * (L + M + 1); // 変更
        frac B(0);                      // (l1 - m1_max) * (l1 + m1_max + 1) = 0
        frac C = (l2 - m2) * (l2 + m2 + 1);
        if (A == 0) {
            std::cerr << "Error : A is 0" << '\n';
            exit(0);
        }
        std::vector<frac> preFactorsC = coeffs[CG(l1, l2, m1_max, m2 + 1, L, M + 1)];
        preFactorsA.push_back((C / A) * preFactorsC[0]);
        if (!preFactorsA.empty())
            coeffs[CG(l1, l2, m1_max, m2, L, M)] = preFactorsA;
    }
    // m1_min <= m1 <= m1_max | m2_min <= m2 <= L - m1_max でのCG係数を求める
    // L+ : sqrt((L + M) * (L - M + 1)) * CG(l1, m1, l2, m2, L, M) = sqrt((l1 + m1) * (l1 - m1 + 1)) * CG(l1, m1 - 1, l2, m2, L, M - 1) + sqrt((l2 + m2) * (l2 - m2 + 1)) * CG(l1, m1, l2, m2 - 1, L, M - 1)
    for (frac m1 = m1_max; m1 > m1_min; --m1)
        for (frac m2 = L - m1_max; m2 >= m2_min; --m2) { //変更
            frac M = m1 + m2;
            if (frac::abs(M - 1) <= L) {
                std::vector<frac> preFactorsB;
                frac A = (L + M) * (L - M + 1);
                frac B = (l1 + m1) * (l1 - m1 + 1);
                frac C = (l2 + m2) * (l2 - m2 + 1);
                if (B == 0) {
                    std::cerr << "Error : B is 0" << '\n';
                    exit(0);
                }
                std::vector<frac> preFactorsA = coeffs[CG(l1, l2, m1, m2, L, M)];
                std::vector<frac> preFactorsC;
                if (frac::abs(m2 - 1) <= l2)
                    preFactorsC = coeffs[CG(l1, l2, m1, m2 - 1, L, M - 1)];
                for (auto &i : preFactorsA)
                    preFactorsB.push_back((A / B) * i);
                for (auto &i : preFactorsC)
                    preFactorsB.push_back(-(C / B) * i);
                if (!preFactorsB.empty())
                    coeffs[CG(l1, l2, m1 - 1, m2, L, M - 1)] = preFactorsB;
            }
        }
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
                std::vector<frac> preFactorsA;
                if (frac::abs(M) <= L)
                    preFactorsA = coeffs[CG(l1, l2, m1, m2, L, M)];
                std::vector<frac> preFactorsB = coeffs[CG(l1, l2, m1 + 1, m2, L, M + 1)];
                for (auto &i : preFactorsA)
                    preFactorsC.push_back((A / C) * i);
                for (auto &i : preFactorsB)
                    preFactorsC.push_back(-(B / C) * i);
                if (!preFactorsC.empty())
                    coeffs[CG(l1, l2, m1, m2 + 1, L, M + 1)] = preFactorsC;
            }
        }
    // 規格化
    for (frac M = -L; M <= L; ++M) {
        std::vector<frac> total;
        for (frac m1 = m1_min; m1 <= m1_max; ++m1)
            for (frac m2 = m2_min; m2 <= m2_max; ++m2)
                if (m1 + m2 == M) {
                    for (int i = 0; i < coeffs[CG(l1, l2, m1, m2, L, M)].size(); ++i)
                        for (int j = 0; j < coeffs[CG(l1, l2, m1, m2, L, M)].size(); ++j)
                            total.push_back(coeffs[CG(l1, l2, m1, m2, L, M)][i] * coeffs[CG(l1, l2, m1, m2, L, M)][j]);
                }
        frac sum;
        for (auto a : total) {
            if (frac::abs(a) == 1) {
                sum = sum + a;
                continue;
            }
            if (frac::abs(a).isSquare()) {
                sum = sum + a.sign() * frac::abs(a).cleanSquareRoot();
                continue;
            }
            std::cerr << "Error : normalization failed." << '\n';
            exit(0);
        }
        for (frac m1 = -l1; m1 <= l1; ++m1)
            for (frac m2 = -l2; m2 <= l2; ++m2)
                if (m1 + m2 == M) {
                    for (int i = 0; i < coeffs[CG(l1, l2, m1, m2, L, M)].size(); ++i)
                        map[CG(l1, l2, m1, m2, L, M)].push_back(coeffs[CG(l1, l2, m1, m2, L, M)][i] / sum); // 規格化
                }
    }
    chrono::system_clock::time_point end = chrono::system_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

}

// sqrtの配列を足し合わせる
frac getCoeff(frac l1, frac l2, frac m1, frac m2, frac L, frac M, std::map<CG, std::vector<frac>> &map) {
    std::vector<frac> factors = map[CG(l1, l2, m1, m2, L, M)];
    // factorsの中身が1つだけなら計算する必要がないのでreturn
    if (factors.size() == 1)
        return factors[0];
    // CG係数としてありえないものは 0
    if (l1 + l2 < L || m1 + m2 != M) {
        return frac(0);
    }
    if (factors.empty()) {
        determineAllCGs(l1, l2, L, map);
        factors = map[CG(l1, l2, m1, m2, L, M)];
        if (factors.empty()) {
            map[CG(l1, l2, m1, m2, L, M)].push_back(frac(0));
            return frac(0);
        }
    }
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
    for (int i = 0; i < factors.size(); ++i)
        for (int j = 0; j < factors.size(); ++j) {
            test.push_back(factors[i] * factors[j]);
        }
    frac sum;
    for (auto a : test) {
        if (frac::abs(a) == 1) {
            sum = sum + a;
            continue;
        }
        if (frac::abs(a).isSquare()) {
            sum = sum + a.sign() * frac::abs(a).cleanSquareRoot();
            continue;
        }
        std::cerr << "Error : getCoeff" << '\n';
        exit(0);
    }
    double precise = 0.0;
    for (auto &factor : factors)
        precise += factor.sign().toDouble() * sqrt(frac::abs(factor).toDouble());
    if (precise < 0.0)
        sum = -sum;
    map[CG(l1, l2, m1, m2, L, M)].clear();
    map[CG(l1, l2, m1, m2, L, M)].push_back(sum);
    return sum;
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

double TRG(double K, MKL_INT D_cut, MKL_INT l_max, MKL_INT N, MKL_INT &order, std::map<CG, std::vector<frac>> &map) {
    chrono::system_clock::time_point start = chrono::system_clock::now();
    // index dimension
    MKL_INT D = D_cut;

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
                                            if (k + l < L || im - i + jm - j != M || km - k + lm - l != M)
                                                continue; // CG係数としてありえないものは0なので飛ばす
                                            frac c(1);
                                            c *= getCoeff(frac(i), frac(j), frac(im - i), frac(jm - j), frac(L),
                                                          frac(M), map);
                                            c *= getCoeff(frac(i), frac(j), frac(0), frac(0), frac(L), frac(0), map);
                                            c *= getCoeff(frac(k), frac(l), frac(km - k), frac(lm - l), frac(L),
                                                          frac(M), map);
                                            c *= getCoeff(frac(k), frac(l), frac(0), frac(0), frac(L), frac(0), map);
                                            c /= frac(2 * L + 1).sign() * (2 * L + 1) * (2 * L + 1);
                                            sum += c.sign().toDouble() * std::sqrt(frac::abs(c).toDouble());
                                        }
                                    T[i * i + im][j * j + jm][k * k + km][l * l + lm] = std::sqrt(
                                            std::cyl_bessel_i(i + 0.5, K) * std::cyl_bessel_i(j + 0.5, K) *
                                            std::cyl_bessel_i(k + 0.5, K) * std::cyl_bessel_i(l + 0.5, K) *
                                            (2 * i + 1) * (2 * j + 1) * (2 * k + 1) * (2 * l + 1)) * sum;
                                }
                }

    REP(n, N) {
        // Tを THRESHOLD 以下に丸め込む
        MKL_INT c = 0;
        REP4(i, j, k, l, D) {
                        if (T[i][j][k][l] <= THRESHOLD)
                            continue;
                        MKL_INT count = 0;
                        double t = T[i][j][k][l];
                        while (t > THRESHOLD) {
                            t /= THRESHOLD;
                            count++;
                        }
                        c = max(c, count);
                    }
        order += c;
        cout << n + 1 << "-th order : " << order << '\n';

        MKL_UINT64 div = pow(THRESHOLD, c);
        REP4(i, j, k, l, D) {
                        T[i][j][k][l] /= div;
                    }

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
            cout << "The algorithm computing SVD failed to converge.\n";
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
            cout << "The algorithm computing SVD failed to converge.\n";
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
        order *= 2;
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
    // inputs
    double K;      // inverse temperature
    MKL_INT D_cut; // bond dimension
    MKL_INT N;     // repeat count
    MKL_INT l_max; // max l
    double K_start;
    double K_end;

//    cout << "input K : ";
//    cin >> K;
    cout << "input max l : ";
    cin >> l_max;
    cout << "input N : ";
    cin >> N;
    cout << '\n';

    chrono::system_clock::time_point start = chrono::system_clock::now();

    const string fileName = to_string(l_max) + '-' + to_string(N) + ".txt";
    std::ofstream dataFile;
    dataFile.open(fileName, std::ios::trunc);

    D_cut = (l_max + 1) * (l_max + 1);

    std::map<CG, std::vector<frac>> map;

    double V = pow(4, N);

    K_start = 0.1;
    K_end = 4.0;
    K = K_start;
    while (K <= K_end) {
        MKL_INT order = 0;
        double Z = log(TRG(K, D_cut, l_max, N * 2, order, map));
        Z = log(M_PI / (2 * K)) + order * log(THRESHOLD) / V + Z / V;
        dataFile << K << '\t' << -Z / K << '\n';
        K += MESH;
    }
    chrono::system_clock::time_point end = chrono::system_clock::now();
    cout << "合計計算時間 : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    return 0;
}
