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
#define CONV_FILTERS       16
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const int8_t conv2d_6_bias[CONV_FILTERS] = {-20, -2, -2, 0, -3, -4, -1, 1, -2, -3, -1, -2, -2, -7, -2, -4}
;


const int8_t conv2d_6_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{19}
, {26}
, {17}
}
, {{-18}
, {23}
, {6}
}
, {{7}
, {13}
, {-21}
}
}
, {{{2}
, {33}
, {40}
}
, {{28}
, {32}
, {7}
}
, {{-55}
, {-68}
, {-40}
}
}
, {{{-28}
, {14}
, {-34}
}
, {{24}
, {47}
, {29}
}
, {{-1}
, {-3}
, {1}
}
}
, {{{-77}
, {-32}
, {26}
}
, {{-53}
, {34}
, {49}
}
, {{-78}
, {-33}
, {14}
}
}
, {{{-34}
, {-7}
, {26}
}
, {{-42}
, {-16}
, {43}
}
, {{-27}
, {27}
, {38}
}
}
, {{{2}
, {-13}
, {-14}
}
, {{21}
, {41}
, {22}
}
, {{13}
, {26}
, {-32}
}
}
, {{{0}
, {49}
, {43}
}
, {{-27}
, {12}
, {46}
}
, {{-86}
, {-73}
, {-42}
}
}
, {{{-4}
, {-40}
, {-32}
}
, {{-32}
, {-20}
, {-18}
}
, {{40}
, {36}
, {57}
}
}
, {{{27}
, {13}
, {-53}
}
, {{35}
, {-6}
, {-37}
}
, {{42}
, {2}
, {-39}
}
}
, {{{-26}
, {1}
, {-17}
}
, {{5}
, {48}
, {45}
}
, {{-23}
, {7}
, {10}
}
}
, {{{-32}
, {-56}
, {-49}
}
, {{38}
, {30}
, {13}
}
, {{17}
, {20}
, {13}
}
}
, {{{3}
, {-25}
, {-38}
}
, {{-8}
, {34}
, {-2}
}
, {{16}
, {52}
, {20}
}
}
, {{{-17}
, {-4}
, {44}
}
, {{2}
, {25}
, {48}
}
, {{6}
, {-28}
, {-30}
}
}
, {{{-20}
, {19}
, {-31}
}
, {{30}
, {64}
, {-1}
}
, {{-27}
, {25}
, {-1}
}
}
, {{{1}
, {-21}
, {-11}
}
, {{36}
, {-31}
, {-74}
}
, {{65}
, {28}
, {-3}
}
}
, {{{-16}
, {-1}
, {-14}
}
, {{13}
, {24}
, {35}
}
, {{-13}
, {19}
, {14}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS