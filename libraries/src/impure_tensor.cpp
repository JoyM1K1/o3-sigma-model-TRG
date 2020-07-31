#include "../include/impure_tensor.hpp"
#include <cmath>
#include <algorithm>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

#define LINF 1e300

ImpureTensor::ImpureTensor() {
    tensors[0] = Tensor();
    tensors[1] = Tensor();
    tensors[2] = Tensor();
}

ImpureTensor::ImpureTensor(int D) {
    tensors[0] = Tensor(D);
    tensors[1] = Tensor(D);
    tensors[2] = Tensor(D);
}

ImpureTensor::ImpureTensor(int Dx, int Dy, int Dx_max, int Dy_max) {
    tensors[0] = Tensor(Dx, Dy, Dx_max, Dy_max);
    tensors[1] = Tensor(Dx, Dy, Dx_max, Dy_max);
    tensors[2] = Tensor(Dx, Dy, Dx_max, Dy_max);
}

ImpureTensor::ImpureTensor(int d, ImpureTensor &T) {
    this->distance = d;
    tensors[0] = Tensor(T.tensors[0]);
    tensors[1] = Tensor(T.tensors[1]);
    tensors[2] = Tensor(T.tensors[2]);
}

ImpureTensor::ImpureTensor(ImpureTensor &rhs) {
    distance = rhs.distance;
    corrs.clear();
    tensors[0] = rhs.tensors[0];
    tensors[1] = rhs.tensors[1];
    tensors[2] = rhs.tensors[2];
}

ImpureTensor::~ImpureTensor() {
    corrs.clear();
}

ImpureTensor &ImpureTensor::operator=(const ImpureTensor &rhs) {
    distance = rhs.distance;
    tensors[0] = rhs.tensors[0];
    tensors[1] = rhs.tensors[1];
    tensors[2] = rhs.tensors[2];
    return *this;
}

int ImpureTensor::normalization(Tensor &T, ImpureTensor &originIMT, std::vector<ImpureTensor> &IMTs) {
//    double _min = LINF;
    double _max = 0;
    int Dx = T.GetDx();
    int Dy = T.GetDy();
    bool isAllMerged = true;
    REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
        double t = std::abs(T(i, j, k, l));
        if (t > 0) {
//            _min = std::min(_min, t);
            _max = std::max(_max, t);
        }
    }
    for (ImpureTensor &IMT : IMTs) {
        if (!IMT.isMerged) {
            isAllMerged = false;
            continue;
        }
        for (Tensor &tensor : IMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                double t = std::abs(tensor(i, j, k, l));
                if (t > 0) {
//                    _min = std::min(_min, t);
                    _max = std::max(_max, t);
                }
            }
        }
    }
    if (!isAllMerged) {
        for (Tensor &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                double t = std::abs(tensor(i, j, k, l));
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
        for (Tensor &tensor : originIMT.tensors) {
            REP(i, Dx)REP(j, Dy)REP(k, Dx)REP(l, Dy) {
                if (o > 0) {
                    REP(t, std::abs(o)) tensor(i, j, k, l) /= 10;
                } else {
                    REP(t, std::abs(o)) tensor(i, j, k, l) *= 10;
                }
            }
        }
    }
    for (ImpureTensor &IMT : IMTs) {
        if (!IMT.isMerged) continue;
        for (Tensor &tensor : IMT.tensors) {
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