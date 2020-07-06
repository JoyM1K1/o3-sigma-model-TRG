//
// Created by Joy on 2020/06/28.
//

#include "../include/impure_tensor.hpp"

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