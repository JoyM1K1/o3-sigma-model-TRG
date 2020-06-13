//
// Created by Joy on 2020/06/09.
//

#ifndef O3_SIGMA_MODEL_HOTRG_HPP
#define O3_SIGMA_MODEL_HOTRG_HPP

#include <vector>

namespace HOTRG {
    void contractionX(const int &Dx, int &Dy, const int &D_cut, std::vector<std::vector<std::vector<std::vector<double>>>> &T);
    void contractionY(int &Dx, const int &Dy, const int &D_cut, std::vector<std::vector<std::vector<std::vector<double>>>> &T);
//    void solver(int &D, const int &D_cut, std::vector<std::vector<std::vector<std::vector<double>>>> &T);
}

#endif //O3_SIGMA_MODEL_HOTRG_HPP
