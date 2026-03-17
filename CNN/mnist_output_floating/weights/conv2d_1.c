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


const float conv2d_1_bias[CONV_FILTERS] = {-0x1.7612520000000p-2, -0x1.8bd1140000000p-3, -0x1.7f4af80000000p-2, -0x1.46fc1a0000000p-5, -0x1.90c87e0000000p-2, -0x1.357dc60000000p-7, -0x1.c122ae0000000p-3, -0x1.0a51540000000p-3}
;


const float conv2d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.07be500000000p-2}
, {0x1.df97200000000p-3}
, {0x1.cf17f80000000p-6}
}
, {{0x1.4e49780000000p-5}
, {0x1.58cd460000000p-3}
, {-0x1.082a520000000p-5}
}
, {{0x1.72ca4c0000000p-3}
, {0x1.88c3be0000000p-2}
, {0x1.75fb100000000p-4}
}
}
, {{{-0x1.2fa1760000000p-3}
, {-0x1.69b69a0000000p-2}
, {-0x1.279b040000000p-1}
}
, {{0x1.1e40ae0000000p-1}
, {-0x1.1bd9300000000p-5}
, {-0x1.46f51c0000000p-3}
}
, {{0x1.5effee0000000p-3}
, {0x1.130ff60000000p-1}
, {0x1.6001520000000p-3}
}
}
, {{{-0x1.1af2c00000000p-1}
, {0x1.7ae5380000000p-4}
, {0x1.c351900000000p-2}
}
, {{0x1.2adcce0000000p-2}
, {0x1.5ca9260000000p-1}
, {-0x1.52bb220000000p-5}
}
, {{0x1.30b9ba0000000p-2}
, {-0x1.5e01800000000p-3}
, {-0x1.3755260000000p-1}
}
}
, {{{-0x1.0abb880000000p-1}
, {-0x1.51ca3e0000000p-3}
, {-0x1.17ec0e0000000p-3}
}
, {{0x1.9145240000000p-3}
, {-0x1.14ef040000000p-4}
, {0x1.10d0ee0000000p-2}
}
, {{0x1.7326720000000p-2}
, {0x1.620e3a0000000p-4}
, {0x1.6d26ec0000000p-2}
}
}
, {{{0x1.01d8260000000p-2}
, {0x1.745f4c0000000p-3}
, {0x1.7e4fe20000000p-4}
}
, {{0x1.d99bce0000000p-3}
, {0x1.6fb4ce0000000p-2}
, {-0x1.b92a000000000p-4}
}
, {{-0x1.95b11e0000000p-2}
, {0x1.3e4d920000000p-2}
, {0x1.2c36c20000000p-2}
}
}
, {{{0x1.20b2400000000p-2}
, {0x1.111f7e0000000p-2}
, {0x1.61a90a0000000p-2}
}
, {{0x1.ebc92a0000000p-4}
, {0x1.0d31380000000p-4}
, {0x1.24b2440000000p-2}
}
, {{-0x1.61908a0000000p-1}
, {-0x1.011b920000000p-1}
, {-0x1.92cc040000000p-1}
}
}
, {{{-0x1.5035cc0000000p-2}
, {-0x1.20e4700000000p-5}
, {0x1.0f6f4e0000000p-2}
}
, {{-0x1.e0b2620000000p-2}
, {-0x1.29abc60000000p-3}
, {0x1.f17e7e0000000p-2}
}
, {{-0x1.3c56c00000000p-3}
, {0x1.160df20000000p-2}
, {0x1.fe21f80000000p-3}
}
}
, {{{0x1.02f2a00000000p-2}
, {0x1.3265500000000p-5}
, {-0x1.09c3dc0000000p-1}
}
, {{0x1.61a6a40000000p-2}
, {0x1.c509b60000000p-5}
, {-0x1.22ba5e0000000p-1}
}
, {{0x1.1ae6ba0000000p-1}
, {0x1.c51d0a0000000p-5}
, {-0x1.1ec9700000000p-3}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS