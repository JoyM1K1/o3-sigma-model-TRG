//
// Created by Joy on 2020/04/29.
//

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