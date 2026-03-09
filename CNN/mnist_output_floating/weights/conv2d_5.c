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


const float conv2d_5_bias[CONV_FILTERS] = {-0x1.c05cd80000000p-5, -0x1.624cb40000000p-3, -0x1.e82be00000000p-4, -0x1.4877080000000p-8, -0x1.04997a0000000p-2, -0x1.23886a0000000p-5, -0x1.09f12c0000000p-5, -0x1.b74f9a0000000p-3}
;


const float conv2d_5_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-0x1.0d98520000000p-3}
, {0x1.5c8ce40000000p-2}
, {0x1.a4f8440000000p-2}
}
, {{-0x1.e9f6260000000p-8}
, {0x1.df5de20000000p-10}
, {0x1.66ad440000000p-2}
}
, {{-0x1.4b37be0000000p-4}
, {-0x1.8ad0aa0000000p-5}
, {0x1.3820120000000p-4}
}
}
, {{{0x1.d5a3540000000p-3}
, {0x1.d72b3c0000000p-3}
, {-0x1.eb708c0000000p-3}
}
, {{0x1.9070aa0000000p-5}
, {0x1.b0429e0000000p-2}
, {0x1.4a65be0000000p-2}
}
, {{-0x1.ca12440000000p-2}
, {-0x1.0289e40000000p-4}
, {0x1.5981600000000p-2}
}
}
, {{{0x1.0841960000000p-3}
, {0x1.5ff8700000000p-3}
, {-0x1.5e38a80000000p-5}
}
, {{0x1.58aeea0000000p-2}
, {0x1.0c098a0000000p-2}
, {0x1.1dfa7a0000000p-2}
}
, {{-0x1.33f3260000000p-2}
, {0x1.0093f80000000p-2}
, {0x1.cf78700000000p-3}
}
}
, {{{-0x1.a2d5360000000p-3}
, {-0x1.9009da0000000p-2}
, {-0x1.f5ff860000000p-2}
}
, {{0x1.da32920000000p-2}
, {0x1.0980320000000p-5}
, {0x1.cf37c80000000p-3}
}
, {{0x1.0123900000000p-2}
, {0x1.e90dee0000000p-2}
, {0x1.9b3f720000000p-4}
}
}
, {{{0x1.ce77f40000000p-4}
, {0x1.2c75b00000000p-2}
, {0x1.3bd9c40000000p-3}
}
, {{0x1.3787e00000000p-4}
, {0x1.08091e0000000p-3}
, {0x1.fedac20000000p-3}
}
, {{0x1.35355a0000000p-2}
, {0x1.d9ff660000000p-3}
, {0x1.af44220000000p-3}
}
}
, {{{0x1.9a76080000000p-3}
, {0x1.3e63600000000p-2}
, {0x1.a0272e0000000p-2}
}
, {{-0x1.8a2a8e0000000p-4}
, {0x1.0b068c0000000p-2}
, {0x1.6be21c0000000p-3}
}
, {{-0x1.11a00e0000000p-1}
, {-0x1.6562b80000000p-1}
, {-0x1.cac82e0000000p-3}
}
}
, {{{0x1.62b61e0000000p-3}
, {0x1.5a84ae0000000p-2}
, {0x1.5cb0b20000000p-2}
}
, {{0x1.7288080000000p-2}
, {0x1.bc4c8e0000000p-2}
, {-0x1.e4408a0000000p-3}
}
, {{0x1.bda1000000000p-8}
, {-0x1.6c4a2a0000000p-2}
, {-0x1.b088f00000000p-1}
}
}
, {{{-0x1.1da21c0000000p-4}
, {0x1.fcb5160000000p-3}
, {0x1.51252c0000000p-3}
}
, {{-0x1.57722c0000000p-3}
, {0x1.c57fdc0000000p-3}
, {0x1.5df5880000000p-2}
}
, {{0x1.9f57780000000p-5}
, {0x1.5b391a0000000p-2}
, {0x1.df0f820000000p-3}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS