//
// Created by Joy on 2020/05/15.
//

#ifndef O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
#define O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP

#include <vector>

namespace math {
    namespace solver {
        // P_n(x) の微分
        double d_legendre(int n, double x);

        // newton法
        double newton(int n, double x);

        // newton法を用いてn次Legendre多項式のゼロ点を計算する
        std::vector<double> legendre_zero_point(int n);
    }
}

#endif //O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
