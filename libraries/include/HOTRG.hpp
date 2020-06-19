//
// Created by Joy on 2020/06/09.
//

#ifndef O3_SIGMA_MODEL_HOTRG_HPP
#define O3_SIGMA_MODEL_HOTRG_HPP

#include <string>
#include "tensor.hpp"

namespace HOTRG {
    void contractionX(const int &D_cut, Tensor &leftT, Tensor &rightT, const double *U, const std::string mergeT);
    void contractionY(const int &D_cut, Tensor &bottomT, Tensor &topT, const double *U, const std::string mergeT);
    void SVD_X(const int D_cut, Tensor &T, double *U);
    void SVD_Y(const int D_cut, Tensor &T, double *U);
//    void solver(int &D, const int &D_cut, std::vector<std::vector<std::vector<std::vector<double>>>> &T);
}

#endif //O3_SIGMA_MODEL_HOTRG_HPP
