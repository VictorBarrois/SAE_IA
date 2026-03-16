/**
  ******************************************************************************
  * @file    weights/conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS     1
#define CONV_FILTERS       8
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const float conv2d_5_bias[CONV_FILTERS] = {-0x1.11d4880000000p-3, -0x1.4512440000000p-2, -0x1.b8afbe0000000p-4, -0x1.423b4a0000000p-7, -0x1.81d8bc0000000p-3, -0x1.3628840000000p-2, -0x1.37b0580000000p-8, -0x1.fff1cc0000000p-5}
;


const float conv2d_5_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.e81e260000000p-3}
, {0x1.339baa0000000p-2}
, {0x1.54739a0000000p-2}
}
, {{0x1.1aed520000000p-1}
, {0x1.deb3100000000p-3}
, {-0x1.c4c9fe0000000p-3}
}
, {{-0x1.0b67d00000000p-8}
, {-0x1.cfd36e0000000p-2}
, {-0x1.a2054a0000000p-1}
}
}
, {{{0x1.e4b7760000000p-3}
, {0x1.7ed7fe0000000p-3}
, {0x1.54dafa0000000p-4}
}
, {{0x1.9fa3620000000p-3}
, {0x1.94a75c0000000p-2}
, {0x1.729a260000000p-5}
}
, {{0x1.840b640000000p-4}
, {0x1.3e23900000000p-3}
, {0x1.b4c1900000000p-4}
}
}
, {{{-0x1.ada7c40000000p-2}
, {-0x1.218c960000000p-1}
, {-0x1.ddd36a0000000p-3}
}
, {{-0x1.e477be0000000p-6}
, {0x1.2cf0cc0000000p-4}
, {0x1.1316b80000000p-1}
}
, {{0x1.04c8de0000000p-2}
, {0x1.b964c60000000p-2}
, {0x1.1466ee0000000p-4}
}
}
, {{{-0x1.c237b40000000p-2}
, {-0x1.48e4fa0000000p-2}
, {-0x1.8082620000000p-2}
}
, {{-0x1.d8aa360000000p-6}
, {0x1.e6276e0000000p-5}
, {0x1.2a0e040000000p-3}
}
, {{0x1.dab60e0000000p-2}
, {0x1.47cca80000000p-3}
, {0x1.17f7100000000p-3}
}
}
, {{{-0x1.7ee47c0000000p-4}
, {0x1.1e79160000000p-4}
, {-0x1.cf02360000000p-4}
}
, {{0x1.fa08760000000p-3}
, {0x1.4af0ca0000000p-4}
, {-0x1.7c10940000000p-7}
}
, {{0x1.97ae620000000p-2}
, {0x1.5dda020000000p-2}
, {0x1.531f720000000p-4}
}
}
, {{{0x1.94792a0000000p-3}
, {0x1.1983f60000000p-2}
, {0x1.40c55a0000000p-4}
}
, {{-0x1.4aee240000000p-4}
, {0x1.3699de0000000p-2}
, {0x1.6b390a0000000p-2}
}
, {{-0x1.63b6ca0000000p-1}
, {-0x1.ec8cf20000000p-3}
, {0x1.b1d3da0000000p-2}
}
}
, {{{0x1.3d9e6e0000000p-4}
, {0x1.aa1b140000000p-2}
, {0x1.8e1da00000000p-2}
}
, {{0x1.dcc6b60000000p-3}
, {-0x1.11c2760000000p-5}
, {0x1.65115e0000000p-3}
}
, {{-0x1.62cd280000000p-1}
, {-0x1.07cc5c0000000p-1}
, {-0x1.11c14c0000000p-1}
}
}
, {{{0x1.5d87800000000p-2}
, {0x1.7447f00000000p-10}
, {-0x1.1c7e020000000p-1}
}
, {{0x1.6e05000000000p-3}
, {0x1.e7fd080000000p-5}
, {-0x1.2045fe0000000p-1}
}
, {{0x1.2533ee0000000p-1}
, {0x1.5998460000000p-4}
, {-0x1.dda65c0000000p-3}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS