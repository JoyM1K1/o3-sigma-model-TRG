#include "../include/impure_tensor.hpp"
#include "../include/TRG.hpp"
#include "../include/HOTRG.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

//#define LINF 1e300

template<class Tensor>
int BaseImpureTensor<Tensor>::normalization(Tensor &T, BaseImpureTensor<Tensor> &originIMT) {
    double _max = 0;
    int Dx = T.GetDx();
    int Dy = T.GetDy();
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    double t = std::abs(T(i, j, k, l));
                    if (std::isnan(t)) {
                        std::cerr << "T(" << i << ',' << j << ',' << k << ',' << l << ") is nan";
                        exit(1);
                    }
                    if (t > 0) {
                        _max = std::max(_max, t);
                    }
                }
    for (int n = 0; n < 3; ++n) {
        REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                        double t = std::abs(originIMT.tensors[n](i, j, k, l));
                        if (std::isnan(t)) {
                            std::cerr << "originIMT[" << n << "](" << i << ',' << j << ',' << k << ',' << l << ") is nan";
                            exit(1);
                        }
                        if (t > 0) {
                            _max = std::max(_max, t);
                        }
                    }
    }
    auto o = static_cast<int>(std::floor(std::log10(_max)));
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    for (BaseTensor &tensor : originIMT.tensors) {
        REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                        if (o > 0) {
                            REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                        } else {
                            REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                        }
                    }
    }
    return o;
}

template<class Tensor>
int BaseImpureTensor<Tensor>::normalization(Tensor &T, BaseImpureTensor<Tensor> &originIMT, std::vector<BaseImpureTensor<Tensor>> &IMTs) {
//    double _min = LINF;
    double _max = 0;
    int Dx = T.GetDx();
    int Dy = T.GetDy();
    bool isAllMerged = true;
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    double t = std::abs(T(i, j, k, l));
                    if (std::isnan(t)) {
                        std::cerr << "T(" << i << ',' << j << ',' << k << ',' << l << ") is nan";
                        exit(1);
                    }
                    if (t > 0) {
//            _min = std::min(_min, t);
                        _max = std::max(_max, t);
                    }
                }
    for (BaseImpureTensor &IMT : IMTs) {
        if (!IMT.isMerged) {
            isAllMerged = false;
            continue;
        }
        for (int n = 0; n < 3; ++n) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            double t = std::abs(IMT.tensors[n](i, j, k, l));
                            if (std::isnan(t)) {
                                std::cerr << "IMT[" << IMT.distance << "][" << n << "](" << i << ',' << j << ',' << k << ',' << l << ") is nan";
                                exit(1);
                            }
                            if (t > 0) {
//                    _min = std::min(_min, t);
                                _max = std::max(_max, t);
                            }
                        }
        }
    }
    if (!isAllMerged) {
        for (int n = 0; n < 3; ++n) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            double t = std::abs(originIMT.tensors[n](i, j, k, l));
                            if (std::isnan(t)) {
                                std::cerr << "originIMT[" << n << "](" << i << ',' << j << ',' << k << ',' << l << ") is nan";
                                exit(1);
                            }
                            if (t > 0) {
//                    _min = std::min(_min, t);
                                _max = std::max(_max, t);
                            }
                        }
        }
    }
//    auto o = static_cast<int>(std::floor((std::log10(_min) + std::log10(_max)) / 2));
    auto o = static_cast<int>(std::floor(std::log10(_max)));
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                    if (o > 0) {
                        REP(t, std::abs(o)) T(i, j, k, l) /= 10;
                    } else {
                        REP(t, std::abs(o)) T(i, j, k, l) *= 10;
                    }
                }
    if (!isAllMerged) {
        for (BaseTensor &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            if (o > 0) {
                                REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                            } else {
                                REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                            }
                        }
        }
    }
    for (BaseImpureTensor &IMT : IMTs) {
        if (!IMT.isMerged) continue;
        for (BaseTensor &tensor : IMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                            if (o > 0) {
                                REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                            } else {
                                REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                            }
                        }
        }
    }
    return o;
}

/* 明示的インスタンス生成 */
template class BaseImpureTensor<TRG::Tensor>;
template class BaseImpureTensor<HOTRG::Tensor>;