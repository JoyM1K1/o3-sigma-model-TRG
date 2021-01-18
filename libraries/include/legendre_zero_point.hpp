#ifndef O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
#define O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP

#include <vector>

namespace math {
    namespace solver {
        /// Derivative of Legendre polynomial. : P'n(x)
        double d_legendre(int n, double x);

        /// Newton's method.
        double newton(int n, double x);

        /// Calculate root of Legendre polynomial.
        std::vector<double> legendre_zero_point(int n);
    }
}

#endif //O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
