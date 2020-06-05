//
// Created by Joy on 2020/05/15.
//

#ifndef O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
#define O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP

#include <vector>

namespace math::solver {
    // P_n(x) の微分
    double d_legendre(size_t n, double x);

    // newton法
    double newton(size_t n, double x);

    // newton法を用いてn次Legendre多項式のゼロ点を計算する
    std::vector<double> legendre_zero_point(size_t n);
}

#endif //O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
