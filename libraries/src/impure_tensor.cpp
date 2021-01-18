#include "../include/impure_tensor.hpp"
#include "../include/TRG.hpp"
#include "../include/HOTRG.hpp"

/* explicit instantiation */
template
class BaseImpureTensor<TRG::Tensor>;

template
class BaseImpureTensor<HOTRG::Tensor>;