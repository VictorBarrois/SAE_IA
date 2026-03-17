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


const float conv2d_148_bias[CONV_FILTERS] = {-0x1.48bfb80000000p-11, 0x1.91ed560000000p-6, -0x1.67fe3c0000000p-6, -0x1.54eac80000000p-8, -0x1.172d620000000p-6, -0x1.7cf8780000000p-4, -0x1.5b84e80000000p-6, -0x1.56b1d00000000p-10}
;


const float conv2d_148_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.9521ee0000000p-2}
, {0x1.00ddda0000000p-1}
, {0x1.8690b60000000p-2}
}
, {{0x1.b728700000000p-6}
, {0x1.1b64d80000000p-6}
, {0x1.9a87600000000p-9}
}
, {{-0x1.7e5ee20000000p-1}
, {-0x1.0cdaae0000000p-1}
, {-0x1.ffa75a0000000p-3}
}
}
, {{{-0x1.676daa0000000p-2}
, {-0x1.1b76900000000p-1}
, {-0x1.4b7a720000000p-2}
}
, {{-0x1.f950fe0000000p-5}
, {-0x1.d670100000000p-3}
, {-0x1.0862c20000000p-2}
}
, {{0x1.455dea0000000p-1}
, {0x1.ed95a20000000p-2}
, {0x1.2772500000000p-3}
}
}
, {{{-0x1.3839060000000p-3}
, {-0x1.cb81ba0000000p-6}
, {-0x1.2b38900000000p-3}
}
, {{0x1.6c7be20000000p-2}
, {0x1.3037300000000p-1}
, {0x1.5209360000000p-2}
}
, {{-0x1.c1c7ae0000000p-3}
, {0x1.52b1b60000000p-2}
, {0x1.4f938e0000000p-4}
}
}
, {{{0x1.052b5e0000000p-1}
, {0x1.7920140000000p-5}
, {-0x1.5559ec0000000p-2}
}
, {{0x1.0e09080000000p-1}
, {-0x1.7649ce0000000p-5}
, {-0x1.27232a0000000p-1}
}
, {{0x1.17f91e0000000p-2}
, {-0x1.4b4daa0000000p-3}
, {-0x1.7e615c0000000p-2}
}
}
, {{{-0x1.a8e3d80000000p-3}
, {0x1.4f78b80000000p-3}
, {-0x1.cb2d720000000p-3}
}
, {{0x1.2477d00000000p-1}
, {0x1.5076a00000000p-1}
, {0x1.dd9c7e0000000p-2}
}
, {{-0x1.7ab3580000000p-5}
, {-0x1.a8bf400000000p-4}
, {-0x1.cb20740000000p-4}
}
}
, {{{0x1.62af440000000p-5}
, {-0x1.7b08a00000000p-7}
, {0x1.62e5220000000p-3}
}
, {{0x1.099ec00000000p-2}
, {0x1.b3f8f00000000p-2}
, {0x1.9a1d2c0000000p-6}
}
, {{0x1.540a2c0000000p-2}
, {0x1.19abbe0000000p-2}
, {0x1.021b040000000p-5}
}
}
, {{{-0x1.f13e6e0000000p-4}
, {0x1.0135720000000p-2}
, {0x1.a4e1fc0000000p-2}
}
, {{-0x1.0b318e0000000p-5}
, {0x1.fdf5360000000p-3}
, {0x1.937bfa0000000p-2}
}
, {{-0x1.3361c00000000p-2}
, {-0x1.58c5680000000p-4}
, {0x1.15e12a0000000p-3}
}
}
, {{{-0x1.26326a0000000p-2}
, {-0x1.245e9c0000000p-1}
, {-0x1.0f705e0000000p-1}
}
, {{0x1.1e3d180000000p-2}
, {0x1.da90520000000p-3}
, {0x1.32c2640000000p-3}
}
, {{0x1.25a9380000000p-2}
, {0x1.92fa860000000p-3}
, {0x1.b359c80000000p-3}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS