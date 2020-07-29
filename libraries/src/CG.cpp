//
// Created by Joy on 2020/04/29.
//

#include <chrono>
#include <fstream>
#include "../include/CG.hpp"

CG::CG(frac l1, frac l2, frac m1, frac m2, frac L, frac M) {
    this->l1 = l1;
    this->l2 = l2;
    this->m1 = m1;
    this->m2 = m2;
    this->L = L;
    this->M = M;
}

// Mapのkeyにするために比較演算子を定義
bool CG::operator<(const CG &rhs) const {
    if (l1 != rhs.l1)
        return l1 < rhs.l1;
    if (l2 != rhs.l2)
        return l2 < rhs.l2;
    if (m1 != rhs.m1)
        return m1 < rhs.m1;
    if (m2 != rhs.m2)
        return m2 < rhs.m2;
    if (L != rhs.L)
        return L < rhs.L;
    return M < rhs.M;
}

frac CG::squareSummation(std::vector<frac> &factors) {
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

void CG::determineAllCGs(frac l1, frac l2, frac L, std::map<CG, frac> &map, std::ofstream &CGFile) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::cout << "determine : l1=" << l1 << " l2=" << l2 << " L=" << L << "   ";
    if (l1 + l2 < L) {
        std::cerr << "l1 + l2 < L" << '\n';
        return;
    }
    frac m1_max = frac::min(l1, L + l2);
    frac m2_max = frac::min(l2, L + l1);
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
        frac C = (l2 - m2) * (l2 + m2 + 1);
        if (A == 0) {
            std::cerr << "Error : A is 0" << '\n';
            exit(1);
        }
        frac preFactorC = map[CG(l1, l2, m1_max, m2 + 1, L, M + 1)];
        preFactorA = (C / A) * preFactorC;
        map[CG(l1, l2, m1_max, m2, L, M)] = preFactorA;
    }
    std::cout << "step1 OK  ";
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
    std::cout << "step2 OK  ";
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
                    exit(1);
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
    std::cout << "step3 OK  ";
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
            std::cerr << "step4 : summation error." << '\n';
            exit(1);
        }
//        cout << "sum:" << sum << ' ';
        for (frac m1 = m1_min; m1 <= m1_max; ++m1) {
            frac m2 = M - m1;
            if (m2_min <= m2 && m2 <= m2_max) {
                map[CG(l1, l2, m1, m2, L, M)] /= sum;
                CGFile << l1 << '\t' << l2 << '\t' << m1 << '\t' << m2 << '\t' << L << '\t' << M << '\t'
                       << map[CG(l1, l2, m1, m2, L, M)].num << '\t' << map[CG(l1, l2, m1, m2, L, M)].den << '\n';
            }
        }
    }
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::cout << "time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
}

frac CG::getCoeff(frac l1, frac l2, frac m1, frac m2, frac L, frac M, std::map<CG, frac> &map, std::ofstream &CGFile) {
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
        determineAllCGs(l1, l2, L, map, CGFile);
    }
    return map[CG(l1, l2, m1, m2, L, M)] * (minus ? 1 : -1);
}