#pragma once
// HIP shim for <cub/util_type.cuh>. See cub/cub.cuh in this directory.
// Provides cub::KeyValuePair / cub::Max / cub::ArgMax via hipCUB.
#include <hipcub/util_type.hpp>

namespace cub = hipcub;
