#include "param.h"

#include <gflags/gflags.h>

DEFINE_string(dataset, "karate", "Dataset to ML");
DEFINE_double(train_ratio, 0.5, "Train data ratio");
DEFINE_uint32(m, 10, "dimension of embedding");
DEFINE_uint32(T_in, 5, "T_in");
DEFINE_uint32(T_out, 10, "T_out");
DEFINE_double(tau, 0.01, "tau");
DEFINE_double(lambda, 1.0, "lambda");
DEFINE_double(mu, 0.01, "mu");
DEFINE_double(rho, 0.01, "rho");