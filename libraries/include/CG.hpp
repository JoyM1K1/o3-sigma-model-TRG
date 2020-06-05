//
// Created by Joy on 2020/04/29.
//

#ifndef O3_SIGMA_MODEL_CG_HPP
#define O3_SIGMA_MODEL_CG_HPP

#include <vector>
#include <map>
#include "frac.hpp"

class CG {
public:
    frac l1, l2, m1, m2, L, M;

    explicit CG(frac l1, frac l2, frac m1, frac m2, frac L, frac M);

    bool operator<(const CG &rhs) const;

    static frac squareSummation(std::vector<frac> &factors);

    static void determineAllCGs(frac l1, frac l2, frac L, std::map<CG, frac> &map, const std::string& CGFileName);

    static frac getCoeff(frac l1, frac l2, frac m1, frac m2, frac L, frac M, std::map<CG, frac> &map, const std::string& CGFileName);
};

#endif //O3_SIGMA_MODEL_CG_HPP
