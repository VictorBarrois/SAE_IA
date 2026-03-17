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


const float conv2d_141_bias[CONV_FILTERS] = {-0x1.0c0e720000000p-9, -0x1.462e880000000p-7, -0x1.7de3940000000p-7, -0x1.9184dc0000000p-7, -0x1.3e312e0000000p-8, -0x1.f612e20000000p-7, -0x1.a9543e0000000p-6, -0x1.82ffae0000000p-4}
;


const float conv2d_141_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.1ab05a0000000p-3}
, {0x1.a4dd800000000p-2}
, {0x1.2ebf160000000p-1}
}
, {{0x1.76fdd00000000p-5}
, {0x1.15d6160000000p-4}
, {0x1.4f6ea20000000p-3}
}
, {{-0x1.b561ca0000000p-1}
, {-0x1.07fe8a0000000p-1}
, {-0x1.8d55ae0000000p-1}
}
}
, {{{-0x1.fb18440000000p-3}
, {0x1.0189f60000000p-5}
, {-0x1.2d8ee20000000p-3}
}
, {{0x1.2313080000000p-3}
, {0x1.08a6ba0000000p-1}
, {0x1.f270a60000000p-4}
}
, {{-0x1.bf36480000000p-4}
, {0x1.29ab640000000p-1}
, {0x1.7712240000000p-3}
}
}
, {{{0x1.ff60400000000p-5}
, {0x1.e6db940000000p-4}
, {0x1.5cf0b00000000p-2}
}
, {{-0x1.568ae60000000p-4}
, {0x1.8dc8f40000000p-2}
, {0x1.151e6e0000000p-1}
}
, {{-0x1.97db7a0000000p-3}
, {-0x1.af23880000000p-3}
, {0x1.1f1e560000000p-4}
}
}
, {{{0x1.99f3fc0000000p-6}
, {0x1.0cd2380000000p-3}
, {-0x1.6752440000000p-3}
}
, {{0x1.f469600000000p-2}
, {0x1.6f5fe00000000p-1}
, {0x1.72e1340000000p-4}
}
, {{0x1.b9a0960000000p-3}
, {-0x1.5863980000000p-4}
, {-0x1.f3b6ce0000000p-3}
}
}
, {{{-0x1.0fb7c00000000p-6}
, {0x1.3860a00000000p-5}
, {-0x1.c218420000000p-5}
}
, {{0x1.4e8e740000000p-2}
, {0x1.4bdf8c0000000p-2}
, {0x1.3d381a0000000p-4}
}
, {{0x1.5aa33e0000000p-2}
, {0x1.61c7120000000p-2}
, {-0x1.a7e6520000000p-4}
}
}
, {{{-0x1.0e8c2e0000000p-3}
, {0x1.a4def60000000p-6}
, {0x1.457f0c0000000p-3}
}
, {{0x1.ce3b980000000p-2}
, {0x1.72b5c20000000p-2}
, {0x1.6b796c0000000p-2}
}
, {{-0x1.c753340000000p-3}
, {0x1.7a82c20000000p-4}
, {0x1.d7ce880000000p-4}
}
}
, {{{-0x1.2a625e0000000p-4}
, {0x1.3f03760000000p-3}
, {-0x1.5888640000000p-2}
}
, {{0x1.be3af00000000p-3}
, {0x1.4745a20000000p-2}
, {0x1.9b2b240000000p-4}
}
, {{0x1.214a980000000p-2}
, {0x1.43f7640000000p-2}
, {0x1.e164500000000p-3}
}
}
, {{{0x1.013ec60000000p-1}
, {-0x1.0fc2f00000000p-3}
, {-0x1.ad8e720000000p-2}
}
, {{0x1.dabbee0000000p-2}
, {0x1.5ff6280000000p-5}
, {-0x1.2317520000000p-1}
}
, {{0x1.e16a9c0000000p-2}
, {0x1.44969c0000000p-3}
, {-0x1.d757060000000p-2}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS