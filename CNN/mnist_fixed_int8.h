#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)


#define ARM_NN_TRUNCATE 1
#define RISCV_NN_TRUNCATE 1

#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT64_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%ld, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT64_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}
static inline int32_t scale_and_clamp_to_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int32_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int32_t) * 8);
  }
#else
  number = scale_number_t_int32_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int32_t(number);
#endif
}

#define NUMBER_MIN_INT8_T -128
#define NUMBER_MAX_INT8_T 127

static inline int32_t min_int8_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int8_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int8_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT32_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%d, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT32_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int8_t clamp_to_number_t_int8_t(
  int32_t number) {
	return (int8_t) max_int8_t(
      NUMBER_MIN_INT8_T,
      min_int8_t(
        NUMBER_MAX_INT8_T, number));
}
static inline int8_t scale_and_clamp_to_number_t_int8_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int8_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int8_t) * 8);
  }
#else
  number = scale_number_t_int8_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int8_t(number);
#endif
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    conv2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV2D_6_H_
#define _CONV2D_6_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_HEIGHT        28
#define INPUT_WIDTH         28
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )


typedef int8_t conv2d_6_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

#if 0
void conv2d_6(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]);               // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT

#endif//_CONV2D_6_H_
/**
  ******************************************************************************
  * @file    conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv2d_6.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_HEIGHT        28
#define INPUT_WIDTH         28
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int8_t
#define LONG_NUMBER_T int32_t


static inline void conv2d_6(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS / CONV_GROUPS], // IN

  const NUMBER_T bias[CONV_FILTERS],						                // IN

  NUMBER_T output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]) {               // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, pos_y, z, k; 	// loop indexes for output volume
  unsigned short x, y;
  int input_x, input_y;
  LONG_NUMBER_T	kernel_mac;
  LONG_NUMBER_T tmp;
  static LONG_NUMBER_T	output_acc[CONV_OUTHEIGHT][CONV_OUTWIDTH];

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        output_acc[pos_y][pos_x] = 0;

        for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
          kernel_mac = 0; 
            
          for (y = 0; y < CONV_KERNEL_SIZE_Y; y++) {
            input_y = pos_y * CONV_STRIDE_Y - ZEROPADDING_TOP + y;

            for (x = 0; x < CONV_KERNEL_SIZE_X; x++) {
              input_x = pos_x * CONV_STRIDE_X - ZEROPADDING_LEFT + x;

              if (input_x < 0 || input_x >= INPUT_WIDTH || input_y < 0 || input_y >= INPUT_HEIGHT) // ZeroPadding2D
                tmp = 0;
              else
                tmp = (LONG_NUMBER_T)input[input_y][input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][y][x][z];
              kernel_mac = kernel_mac + tmp;
            }
          }

          output_acc[pos_y][pos_x] = output_acc[pos_y][pos_x] + kernel_mac;

        }
      }
    }

    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        // Scale for possible additional precision of bias
        output_acc[pos_y][pos_x] = scale(NUMBER_T, output_acc[pos_y][pos_x],  WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

        // Scale bias to match accumulator
        output_acc[pos_y][pos_x] += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


#ifdef ACTIVATION_LINEAR
        output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
        // Activation function: ReLU
        if (output_acc[pos_y][pos_x] < 0) {
          output[pos_y][pos_x][k] = 0;
        } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc[pos_y][pos_x] > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc[pos_y][pos_x] = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
          output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        }
#else
#error "Unsupported activation function"
#endif
      }
    }
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif


  static q15_t bufferA[INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS];
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_basic_nonsquare(
#endif
                                      (q7_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q7_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q7_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q7_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
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
/**
  ******************************************************************************
  * @file    conv2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV2D_7_H_
#define _CONV2D_7_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_HEIGHT        13
#define INPUT_WIDTH         13
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )


typedef int8_t conv2d_7_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

#if 0
void conv2d_7(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]);               // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT

#endif//_CONV2D_7_H_
/**
  ******************************************************************************
  * @file    conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv2d_7.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_HEIGHT        13
#define INPUT_WIDTH         13
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int8_t
#define LONG_NUMBER_T int32_t


static inline void conv2d_7(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS / CONV_GROUPS], // IN

  const NUMBER_T bias[CONV_FILTERS],						                // IN

  NUMBER_T output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]) {               // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, pos_y, z, k; 	// loop indexes for output volume
  unsigned short x, y;
  int input_x, input_y;
  LONG_NUMBER_T	kernel_mac;
  LONG_NUMBER_T tmp;
  static LONG_NUMBER_T	output_acc[CONV_OUTHEIGHT][CONV_OUTWIDTH];

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        output_acc[pos_y][pos_x] = 0;

        for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
          kernel_mac = 0; 
            
          for (y = 0; y < CONV_KERNEL_SIZE_Y; y++) {
            input_y = pos_y * CONV_STRIDE_Y - ZEROPADDING_TOP + y;

            for (x = 0; x < CONV_KERNEL_SIZE_X; x++) {
              input_x = pos_x * CONV_STRIDE_X - ZEROPADDING_LEFT + x;

              if (input_x < 0 || input_x >= INPUT_WIDTH || input_y < 0 || input_y >= INPUT_HEIGHT) // ZeroPadding2D
                tmp = 0;
              else
                tmp = (LONG_NUMBER_T)input[input_y][input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][y][x][z];
              kernel_mac = kernel_mac + tmp;
            }
          }

          output_acc[pos_y][pos_x] = output_acc[pos_y][pos_x] + kernel_mac;

        }
      }
    }

    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        // Scale for possible additional precision of bias
        output_acc[pos_y][pos_x] = scale(NUMBER_T, output_acc[pos_y][pos_x],  WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

        // Scale bias to match accumulator
        output_acc[pos_y][pos_x] += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


#ifdef ACTIVATION_LINEAR
        output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
        // Activation function: ReLU
        if (output_acc[pos_y][pos_x] < 0) {
          output[pos_y][pos_x][k] = 0;
        } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc[pos_y][pos_x] > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc[pos_y][pos_x] = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
          output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        }
#else
#error "Unsupported activation function"
#endif
      }
    }
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif


  static q15_t bufferA[INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS];
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_basic_nonsquare(
#endif
                                      (q7_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q7_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q7_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q7_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS     16
#define CONV_FILTERS       32
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const int8_t conv2d_7_bias[CONV_FILTERS] = {34, 3, -24, -5, -9, 4, 19, -11, 6, -1, -4, 8, -7, -4, -6, 17, -15, -14, -1, -4, -9, 21, -4, -13, -1, 1, -1, -2, -12, -4, -10, 4}
;


const int8_t conv2d_7_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{31, 36, 14, -8, -9, 20, 27, -14, 39, 6, 2, 15, 36, 17, 24, 7}
, {25, 34, 4, -13, -2, -4, 20, -1, 21, -5, -17, 9, 33, 4, 21, -4}
, {14, 23, 1, 30, 25, -9, 29, 1, 36, 12, -1, 0, 21, 2, 8, -4}
}
, {{17, -10, -16, -33, -35, -2, 2, -28, 1, -29, -32, -9, -5, -5, -2, -16}
, {-13, -17, -47, -54, -42, -27, 0, -3, -5, -43, -38, -33, -31, -24, -13, -49}
, {-9, -11, -19, 24, 15, -19, 8, 1, 14, -7, 1, -61, 7, -34, -37, -18}
}
, {{-11, -3, 1, 11, -17, -3, 9, 21, -9, -7, -1, -2, -8, -8, -14, -3}
, {-50, -73, -26, 6, -52, 3, -28, 16, -1, -46, 10, -4, -64, -4, 13, -15}
, {-68, -66, -63, -34, -38, -43, -14, -29, 50, -51, -77, -79, -39, -83, -7, -55}
}
}
, {{{-2, -1, 5, 7, 0, 9, 18, 8, 22, 20, 8, 10, 8, 21, 9, 19}
, {-7, -42, -16, -10, -62, 11, -58, -9, 7, -25, 11, 1, -26, -3, 33, -19}
, {-38, -29, -42, -20, -15, -48, -19, 22, -9, -66, -5, -36, -28, -43, 3, -58}
}
, {{14, -13, 18, 32, 51, 18, -19, 15, 37, 24, 11, 14, 16, 3, 26, 14}
, {3, -25, 5, -20, -47, 6, -67, -44, 16, -21, 4, 0, -10, 2, 25, -9}
, {-81, -66, -33, 24, 3, -42, -70, -16, -8, 15, -21, -30, -28, -28, -30, -25}
}
, {{25, -8, 22, 3, 36, 19, -25, -34, 24, 28, -11, 10, 25, 20, 16, 6}
, {-13, -25, -13, -26, -68, -18, -54, -51, -1, -72, -60, -43, -47, -12, -1, -39}
, {-32, -43, -17, 12, 14, -34, -17, 17, -50, 19, -15, -6, -12, -18, -65, 0}
}
}
, {{{-2, -8, 5, 1, 8, 5, -15, 22, 9, 9, 27, 9, -12, 5, 13, 3}
, {-19, 17, -25, -19, -4, -25, 13, 27, -9, -24, 9, -21, 0, -27, 11, -20}
, {-5, 9, -13, -5, 8, -17, 13, 8, 7, -13, -20, -10, -2, -3, 28, -2}
}
, {{14, 14, 10, -15, -4, -7, 24, -10, -1, -4, -8, 17, -2, 10, 21, 4}
, {-16, -11, 5, -37, -33, 5, 12, 18, -10, -5, 33, 12, -1, 10, 27, 3}
, {-17, 8, 4, -41, -10, 3, 12, 9, 4, 6, 12, -11, 1, 4, 13, -4}
}
, {{9, 28, 14, -58, -17, 8, 8, -12, -21, 9, 12, 3, 13, 13, -5, 5}
, {23, 28, 9, -52, -21, 21, 39, -2, -17, 0, 29, 24, -2, 9, 17, 5}
, {-6, 19, 7, -59, -11, 0, 42, 9, -15, 8, 40, -4, -3, 4, 13, 3}
}
}
, {{{-6, -55, -7, -15, -6, -7, -51, 10, 8, 2, -3, 6, -3, -3, 12, -2}
, {-12, -44, -13, 5, 9, -2, -53, -5, 14, 0, -5, -4, 0, -15, -11, -6}
, {-5, 7, 3, 13, 15, -9, -28, 19, 2, -4, 26, -1, -4, 0, -1, -1}
}
, {{0, -17, 4, 14, -11, 0, -16, -3, 30, 4, -13, 4, 3, 12, 10, -5}
, {2, -62, -6, 35, 2, 4, -56, 0, 24, -8, -5, 11, -14, -9, 40, -5}
, {-9, -12, -23, -7, -26, -9, -36, -23, 13, -22, -25, -24, -9, -12, 13, -20}
}
, {{-1, -26, -6, 7, 1, -1, 16, 8, 2, -9, -42, 7, 2, -1, 13, 3}
, {13, -30, 3, 23, 10, 12, 12, 0, 29, 13, -11, 22, 0, 5, 39, 16}
, {5, -73, 12, -10, -34, 7, -32, 25, 28, -1, 40, 32, -10, 6, 52, 3}
}
}
, {{{6, -16, 4, -20, -22, 6, -7, 5, 11, -15, 5, 9, -9, 0, -2, -5}
, {-12, -18, -25, -4, -17, 6, -15, 17, 3, -35, 2, 8, -24, -12, 11, -11}
, {-23, -16, -61, 41, -27, -34, 4, -39, 5, -46, -53, -53, -26, -48, -29, -45}
}
, {{16, 16, 4, -41, -13, 2, 39, -9, -3, 14, 24, 21, 2, 6, 6, 11}
, {5, 24, 22, -10, -44, 28, 55, 2, -3, 35, 21, 23, -1, 19, 12, 24}
, {-30, -39, -15, -44, -52, -12, -3, 4, 43, -33, 26, -2, -45, -2, 48, -7}
}
, {{5, -13, 3, -28, 6, -8, 10, -12, -17, 7, 13, -10, 8, 1, -21, -1}
, {2, 19, 5, 2, 9, 4, 21, -12, 0, 16, 19, 1, 21, 8, 13, 12}
, {-2, -17, 26, -54, -30, 20, -1, -10, 1, 3, 15, 10, -13, 29, 7, 10}
}
}
, {{{2, -28, -3, 29, 7, 5, -46, -4, 25, -1, -17, 7, 20, 3, -2, 12}
, {14, -2, 10, 14, -3, 9, -20, -17, 11, 6, -2, 0, 2, 10, 2, 10}
, {3, 10, -11, -37, 5, 2, 6, 21, 1, 8, 6, 3, 2, 7, 13, 7}
}
, {{-6, 0, 8, 44, -6, 7, 19, -41, 20, -12, -19, -4, 8, 6, 21, 5}
, {11, -2, 9, 55, 15, -1, 7, 1, 5, 24, 12, 21, 3, 16, -6, -3}
, {14, -1, 16, 21, 16, 19, -4, 10, -10, 26, 20, 9, 20, 17, -10, 13}
}
, {{-3, -18, -37, 4, -38, -24, 45, -42, 2, -39, -32, -41, -24, -35, -7, -40}
, {3, 30, -34, -42, -44, -37, 45, -3, -33, -41, -21, -59, -6, -43, -58, -62}
, {-18, 40, -52, -58, -50, -45, 3, -10, -29, -61, -47, -61, -29, -39, -62, -53}
}
}
, {{{-7, -5, 2, 26, 1, -6, 22, -26, -12, 9, -34, -20, 5, 5, -20, 6}
, {14, 13, 14, 48, 25, 13, 5, -5, -2, 13, 21, 16, 15, 17, 10, 9}
, {26, 34, 31, -4, 20, 38, 10, 2, -14, 22, 12, 13, 39, 28, -10, 11}
}
, {{-24, -4, -55, 0, -18, -52, 55, 5, -55, -37, -65, -70, -6, -59, -57, -42}
, {6, 34, -23, -17, -36, -37, 41, -4, -73, -39, -50, -86, 33, -35, -80, -33}
, {-5, 37, -19, -61, -33, -39, 25, 15, -41, -45, -35, -50, -20, -19, -61, -44}
}
, {{-43, -44, -25, -48, -29, -37, -33, 17, -17, -15, 14, -10, -40, -21, -8, -4}
, {-66, -80, -18, -16, 20, -35, -89, 30, -8, -9, 42, 8, -54, -22, -6, -15}
, {-19, -123, -13, -10, -5, -6, -121, 16, -27, -3, 31, 14, -25, -11, 3, 4}
}
}
, {{{12, -11, -7, 18, 1, -10, 6, -17, -38, 3, -29, -1, 5, 1, -64, -11}
, {12, -30, 13, -9, -1, 7, -17, -10, 12, 12, 6, 19, 8, 19, -2, 2}
, {-14, -26, -10, 32, -47, -4, -12, 9, 6, -7, 6, -1, -1, -17, 11, -8}
}
, {{-3, -4, -26, 29, 16, -7, 0, -34, -1, -11, -36, -18, 10, -20, -36, -9}
, {19, -23, 7, 27, 0, 15, 6, -57, -12, 27, -40, 15, 17, 20, -2, 20}
, {7, -21, 15, -21, -75, 18, -19, -6, 5, 5, 10, 13, -15, -2, 14, -9}
}
, {{-26, -3, -25, 5, -10, -24, 5, -25, 11, -5, -62, -21, -1, -33, -7, -9}
, {15, -29, 0, 30, 28, -4, 20, -56, -2, 15, -51, 7, 20, 13, 11, 14}
, {12, -42, -4, -18, -42, 21, -27, -32, 27, -8, 2, 17, -7, 9, 46, 9}
}
}
, {{{11, -11, 7, -10, -20, 6, 15, -6, -11, 9, 12, -3, 6, 9, 7, -1}
, {-21, -25, -25, -40, -78, -12, -7, -1, -9, -41, 14, -9, -28, -17, 9, -20}
, {-7, -18, -21, 18, 14, -22, 31, 2, -3, -4, -19, -34, 8, -29, -14, -23}
}
, {{6, -28, 21, 13, 13, 14, 22, 9, 6, 21, 27, 19, 0, 16, 12, 14}
, {4, -123, 4, -5, -36, 18, -92, -2, 9, -10, 15, 4, -7, -1, 27, -4}
, {-69, -57, -38, 67, 10, -34, -35, -15, 31, -38, -25, -31, -16, -38, 27, -37}
}
, {{2, -24, 17, 36, 23, 14, 6, 3, 23, 12, 10, 12, 12, 14, 21, 6}
, {18, -27, 22, 8, 6, 32, -39, 1, 32, 15, 7, 13, -3, 17, 35, 5}
, {-22, -43, -14, 11, -10, 3, -39, 14, 28, -27, -14, -6, -13, -23, 31, -11}
}
}
, {{{-12, 10, -9, -14, 1, -8, 23, 8, -34, -7, -8, -12, -20, -8, -15, -26}
, {5, -33, -9, -16, 2, 2, -9, -3, -62, 1, -20, 1, -2, -12, -71, 14}
, {15, -27, 6, 6, 1, 12, -60, -4, -3, 4, -8, 7, 11, 13, -3, 4}
}
, {{-15, -39, -6, 17, -14, -14, -37, 18, -54, -3, -22, -19, -16, -19, -42, -5}
, {6, -62, 11, 32, 10, 10, -63, 3, -15, 20, -23, 7, 12, 16, -49, 9}
, {17, -13, 14, -10, -9, 8, -66, -35, 40, 13, -24, 15, -12, 22, 12, 0}
}
, {{1, -30, -13, 21, 10, -7, 1, -6, -67, -6, -22, 4, 2, -2, -63, 6}
, {5, -8, 1, 42, 12, 0, 3, -16, 16, -3, -33, 5, 18, 18, 0, 20}
, {6, -13, 9, -7, -23, 17, -27, -49, 13, 10, -38, 12, 0, 0, 11, 7}
}
}
, {{{-2, 1, 2, -1, -18, -1, 11, -10, 1, 0, 17, 8, -3, 15, 15, -1}
, {5, -33, -6, -1, 11, -11, -16, 3, 13, -6, 7, 11, -5, 3, 20, 1}
, {-4, 5, -5, 4, 4, -9, 7, 3, -9, -5, 23, -1, -2, 0, 4, 5}
}
, {{14, -29, -8, -17, 20, -3, -26, -18, -8, 19, -7, 2, 2, 8, -20, 11}
, {25, -42, 2, 8, 19, 15, -29, -8, 17, 11, 27, 16, 15, 12, 4, 7}
, {8, -1, -4, -32, -22, 9, -7, -25, 6, 7, -21, -7, 13, 5, 15, 10}
}
, {{15, -40, 4, 2, 19, 9, -58, 7, 2, 5, -29, 18, 0, 6, -8, 0}
, {9, -28, 5, 18, 11, 5, -80, -46, 16, 7, -50, 11, -4, 10, 6, 11}
, {5, -1, -1, 7, -24, -10, -22, -26, 31, -19, -44, -17, -12, 1, 10, -6}
}
}
, {{{22, -1, 13, 48, 17, 12, -1, -3, 29, 14, -15, 8, 1, 21, 27, 11}
, {4, -2, 3, -19, -6, 9, -12, -16, 13, -19, -78, -27, 7, -3, -1, -21}
, {-6, -9, -15, -24, -7, -23, -21, -5, -30, -21, -41, -16, -20, -21, -91, -3}
}
, {{9, 19, 20, -27, -59, 20, -13, -33, 20, -39, -13, -2, 21, 16, 32, -6}
, {-48, -34, -39, 5, 20, -66, -26, 13, 55, -17, -79, -30, -58, -41, -40, -26}
, {-5, -56, 4, 8, 3, 13, -41, 34, -34, 18, 20, 12, 11, 4, -32, 17}
}
, {{-46, 11, -45, -15, -11, -68, -34, -29, 3, -54, -81, -70, -77, -41, -56, -57}
, {-8, -67, 5, 33, 24, -12, -32, 4, -26, 17, -17, 4, -6, -11, -111, -9}
, {11, 1, 22, 25, 26, 12, 0, 23, 36, 20, 11, 10, 13, 18, 12, 9}
}
}
, {{{3, 5, 1, 26, 17, -11, 8, -34, -20, -18, -26, -32, 5, -14, -14, -25}
, {4, 5, 2, -1, 23, -5, 29, -22, -27, 23, -19, -4, 23, 1, -2, 17}
, {5, -14, 6, 31, 26, 16, 18, -8, 22, -2, -5, 16, -1, 15, 20, 14}
}
, {{-43, -17, -44, 3, -15, -33, -15, -13, 15, -69, -8, -25, -32, -37, 21, -31}
, {-2, 9, 4, -9, 12, -29, 28, -56, -50, -2, -66, -23, 15, -18, -18, -14}
, {18, 13, 11, 48, 11, 5, 52, -42, 15, 25, 0, 15, 28, 31, 28, 23}
}
, {{-35, -20, 1, -33, -25, 3, -45, 6, 6, -13, 35, 11, -44, 1, 10, -9}
, {-38, 5, -31, -27, -5, -17, 37, 0, -7, -15, 6, -14, -2, -42, 17, -19}
, {0, 22, 14, -2, 13, -5, 48, -20, -31, 7, -15, -11, 12, 1, 2, -3}
}
}
, {{{-4, -2, -2, -21, 3, -5, -14, 11, -1, -1, 21, 0, -7, -5, 9, -6}
, {-17, -34, -5, 34, -4, -1, -15, 25, 14, -9, 20, 8, -13, -4, 8, -2}
, {-9, -27, -18, 21, -27, 5, -21, -18, 12, -25, 36, 0, -16, -10, 36, -15}
}
, {{1, -29, 3, 0, 2, 6, -19, -5, 16, 5, -13, 6, 0, -10, 29, 4}
, {5, -34, 3, 1, -2, 7, -11, -11, 9, 10, -7, 15, -6, 4, 29, 8}
, {-13, -19, 15, 8, 1, 4, -24, 20, 10, 17, 36, 18, -29, 15, 13, 10}
}
, {{7, -2, -3, -47, -6, 2, -9, 1, -1, 7, 3, 1, 12, -6, -9, 6}
, {-4, -5, 10, -21, 1, -3, -29, -10, -19, 12, -23, 9, 10, 8, -18, 23}
, {19, -3, 1, -6, -2, 11, 4, -3, -35, 12, -8, 3, 20, 7, -21, 15}
}
}
, {{{-3, 10, -5, -3, -11, 1, 6, 7, -15, -1, -1, 17, -8, -5, -12, -3}
, {-4, 19, 2, -35, -15, -5, 15, -5, -57, 2, 17, 7, 2, 10, -32, 5}
, {-2, -10, 5, -24, -3, -3, 6, 0, -22, -2, 7, 5, -17, 11, -13, 8}
}
, {{-4, 15, -14, -4, -3, -18, 23, -16, -41, -18, -35, -39, -1, -23, -44, -19}
, {-12, -4, -9, 9, 10, -14, -1, 14, -65, 5, -14, 6, -6, -6, -75, -9}
, {18, -19, 9, 28, 33, 22, -7, 0, -24, 22, 5, 21, 30, 11, -21, 16}
}
, {{-27, -27, -22, -7, 19, -19, -34, 18, -33, -4, -5, -6, -12, -23, -63, -7}
, {14, -33, 16, 21, 19, 5, -21, 2, -59, 2, -12, 1, 8, 5, -36, 9}
, {5, 5, 7, 24, 12, 7, -8, -23, 30, 10, -43, -5, 14, 8, 25, 9}
}
}
, {{{11, 1, 14, 12, 29, 8, -4, 17, 13, 6, 6, 15, 29, 4, 12, 12}
, {19, 5, 7, -6, 30, 15, -13, -17, 3, -2, -24, 15, 21, 7, 15, 10}
, {8, -3, -13, 15, -17, 2, -18, -57, 41, -16, -22, -15, -15, -3, 33, -16}
}
, {{9, 10, 17, 25, 11, 11, -11, 12, 14, 17, -13, 18, 20, 11, -3, 15}
, {19, 19, 8, 27, -29, 9, -23, -84, 9, 5, -29, -16, 28, 23, 6, -2}
, {-30, -6, -30, 5, -46, -21, -28, -20, 35, -84, -74, -58, -38, -20, -8, -43}
}
, {{0, 32, -3, 7, -35, -7, 14, -30, 0, -24, -61, -29, -17, 1, -7, -17}
, {-31, 32, -23, -24, -54, -34, 10, -10, -15, -59, -67, -32, -44, -25, -36, -37}
, {-57, -43, -18, 30, -35, -33, -40, -7, -10, -29, -10, -16, -35, -17, -34, -19}
}
}
, {{{-1, 20, 5, -5, -13, 13, 6, -31, -3, 1, 34, 0, 7, 2, -7, -7}
, {2, 3, -10, 17, 10, -5, -14, -13, -29, -15, -24, -13, 0, -11, -47, -1}
, {2, -15, 3, -9, 15, -5, -19, 19, -4, -12, -7, 9, -21, -9, -23, -6}
}
, {{-16, -51, 0, -14, 2, -29, -73, 17, -41, -16, 0, -3, -30, -18, -26, 2}
, {-1, -57, 13, -24, 34, 16, -75, 30, -52, 25, 50, 22, 10, -7, -57, 18}
, {16, 19, 11, -56, 22, 21, -11, -4, -11, 9, 31, 12, 31, 19, -19, 8}
}
, {{16, 2, 15, -35, 12, 26, -13, 26, -44, 19, 21, 25, 27, 6, -35, 9}
, {18, 22, 2, -6, -6, 7, -19, -47, -7, 4, -13, -4, 21, 5, -24, 2}
, {4, 1, 1, -48, -36, -9, -26, -37, -4, -29, -61, -18, -20, 0, -15, -10}
}
}
, {{{18, -21, -3, 13, -11, 5, -7, -21, 14, -10, -20, 2, -7, 3, 16, -6}
, {7, -22, -12, -27, -12, -7, -33, 0, -12, -9, 2, -1, 0, 9, -19, 9}
, {-9, 1, -10, -28, -14, -3, 2, 4, -25, 3, 22, 13, 1, 8, -6, 18}
}
, {{13, 16, 9, 8, -11, 1, 15, 3, -22, 3, 2, 6, 2, 2, 7, 14}
, {10, 26, 10, -34, -16, 11, 21, -12, -17, 10, 2, -1, 7, 11, 20, 7}
, {-1, 23, -2, -80, -20, -6, 42, 3, -3, 6, 0, 6, -2, -14, 2, -3}
}
, {{-2, 22, -17, 12, -3, -21, 34, -80, -22, -13, -97, -40, 8, -12, -4, -10}
, {14, 21, 10, -5, -24, 7, 52, -37, -17, 10, 3, 3, -1, 6, -1, 3}
, {5, 5, 15, -35, -18, 11, 13, 3, -3, 11, 29, 4, 3, 13, 8, 9}
}
}
, {{{-11, -7, -20, -15, 14, -10, -9, -4, -27, 8, -8, 3, -4, 5, -22, -6}
, {5, -1, 3, -6, 18, 6, -22, 20, -13, 5, -4, 8, 14, 4, 1, 11}
, {22, 18, 15, -7, 5, 21, -8, -35, 8, 17, 3, 18, 45, 21, -6, 18}
}
, {{8, -2, 14, -5, 20, 3, -31, 34, -7, 15, 7, 9, 13, 8, -9, 1}
, {8, 26, 6, 17, 15, 9, -13, -17, 3, 9, 13, 6, 26, 5, 15, 4}
, {-16, 6, -6, -54, -54, 0, -25, -90, 16, -48, -62, -38, -6, -10, 14, -33}
}
, {{6, 31, 7, -5, -18, 5, -7, -39, 11, -2, -30, -4, 12, 6, 1, 5}
, {-6, 24, -23, -42, -87, -32, -6, -26, 2, -55, -28, -29, -23, -10, -24, -39}
, {-38, -26, -14, 8, 7, -44, -34, 11, -13, -26, 23, -16, -47, -25, -23, -9}
}
}
, {{{7, -4, 7, 1, -12, 6, 1, -4, -16, 2, 9, 11, -2, 3, 9, 7}
, {-3, 11, 13, 5, 4, 14, 47, 0, -25, 15, 25, 16, 16, 4, 4, 12}
, {6, 8, 4, -53, -23, 13, 40, 18, -42, 19, 22, 16, 8, 27, 7, 14}
}
, {{9, 8, -15, 15, -8, -9, 18, -16, -9, -9, -47, -16, -3, -6, 0, -10}
, {7, 19, -7, -19, -24, -13, 37, -40, -66, -13, 56, -32, 11, -2, -43, -11}
, {17, 19, 14, -6, 2, 15, 30, -21, -21, 16, 51, -3, 7, 16, -20, 8}
}
, {{-22, 13, -8, -17, -19, -12, 9, 18, -8, -33, 51, -3, -18, -19, -4, -12}
, {-36, -55, -53, -10, -2, -41, -54, 3, -48, -24, -6, -18, -50, -49, -18, -19}
, {5, -16, -18, -3, 9, -4, -3, 2, -21, -14, 4, -8, 5, -20, -28, 2}
}
}
, {{{5, 10, 11, -10, 6, 4, 3, 3, -4, 1, 16, -6, -3, -4, -10, -5}
, {9, 8, -6, -37, -6, -8, 17, 18, -6, -8, -3, 3, -7, 6, -8, -3}
, {-3, -7, -1, -18, 11, -2, 4, 13, -28, 5, 16, 8, 13, -10, -6, 8}
}
, {{11, 26, 6, -33, 10, 8, -11, 25, -2, 1, 3, 10, 7, 1, -3, 0}
, {21, 17, -4, -32, 19, 6, 1, 24, -12, 10, 19, 17, 10, 7, -33, 14}
, {18, 41, 4, -13, -1, 5, 13, -45, 4, 23, -15, -21, 22, -3, -7, 11}
}
, {{-1, 8, -3, -5, 3, -1, -3, -18, 13, 2, -13, -9, -5, -1, 4, -4}
, {11, 5, -2, -4, -18, -2, 10, -52, 10, -4, -67, -22, 12, -12, 10, -11}
, {-1, -23, -26, 22, 5, -31, -1, -9, 7, -22, -47, -15, -29, -28, 15, -24}
}
}
, {{{6, 11, 13, 8, 7, 8, -6, -5, 23, 4, 2, -1, 6, 9, 28, 8}
, {10, 19, 16, 56, 15, 1, 24, 4, 18, 1, -15, 1, 2, 18, 9, 3}
, {19, -1, 10, -5, 27, 11, 7, 17, 0, 11, 16, 11, 9, 1, -1, 17}
}
, {{-1, 22, 1, 44, -20, -12, 39, -17, -11, 26, -6, -11, -3, 1, -36, 8}
, {5, -8, 12, 45, 1, 22, 17, 9, -10, 25, 52, 9, 11, 5, -14, 14}
, {6, 18, 5, -1, -25, 26, -8, -45, 0, 5, 12, -19, 17, 8, -27, -10}
}
, {{-22, 3, -46, -24, -33, -50, 46, 28, -29, -74, 36, -30, -33, -41, -16, -38}
, {-14, -5, -78, -49, -42, -82, 30, 29, -23, -114, 4, -37, -35, -75, -41, -82}
, {-37, -46, -70, -101, -46, -61, -47, 49, -23, -71, 25, -14, -74, -61, -62, -47}
}
}
, {{{3, -4, -18, -9, -15, -6, -16, 17, -9, -26, 5, 9, -19, -5, 16, 5}
, {-22, -2, -28, -27, -15, -22, -6, 29, -15, -53, -5, 5, -50, -17, 3, -8}
, {-29, -27, -38, -39, -9, -22, -3, 31, -24, -24, 11, 11, -48, -19, 4, -20}
}
, {{12, 24, 14, 7, -12, 12, 6, 19, -36, 19, -6, 15, 29, 11, -11, 12}
, {25, 25, 13, 11, -1, 12, 35, 8, -15, 24, 25, 24, 18, 21, 4, 13}
, {11, 39, 22, 1, 8, 18, 19, 9, -22, 26, 10, 8, 22, 26, 6, 25}
}
, {{-6, 25, -12, -8, -30, -16, 21, -49, -19, -18, 1, -32, 7, -23, -9, -19}
, {-2, 13, -18, 7, -22, -27, 19, -44, -29, -31, -88, -30, -1, -9, -32, -17}
, {12, -1, -14, 0, -7, -10, 19, -24, 3, -20, 19, -15, 5, -13, -6, -20}
}
}
, {{{-4, -1, 0, -5, -14, -8, 1, -2, -5, 6, 13, -2, -4, 10, 8, -1}
, {-1, -31, 0, -8, -2, 8, -24, -2, -4, -13, -6, 9, -15, 1, 31, -6}
, {-7, -4, -10, 28, -5, -16, 1, 0, 18, -28, -5, -7, -7, -14, 38, -20}
}
, {{0, 4, -9, -2, -13, -14, 4, -25, -21, 8, -17, -10, 5, 4, -32, 1}
, {10, -1, 15, -10, -13, 4, 30, -19, -22, -10, 6, 5, 13, 9, 8, 13}
, {-5, -7, 19, -54, -24, 19, 25, 1, 20, 4, 28, 15, -4, 30, 16, 8}
}
, {{-4, -1, -6, -37, -21, 2, -10, -37, 0, -17, -27, -9, -19, -2, 7, -11}
, {0, 47, 1, -12, 1, -17, 51, -71, -48, -6, -95, -32, 9, -17, -28, -15}
, {9, 31, 17, -19, -3, 6, 49, -28, -8, 6, 44, 6, 8, 19, 8, 13}
}
}
, {{{-3, -12, -21, -2, 5, -11, 11, 1, -64, -4, -5, -5, 3, -12, -99, 0}
, {21, -48, 16, -3, 13, 26, -40, -12, 0, 17, 12, 10, 5, 11, -14, 9}
, {4, 11, 10, 11, -52, 14, -3, -53, 8, -13, -17, -14, 7, 0, 30, -3}
}
, {{-13, -41, -13, 14, 12, -14, -28, 16, -96, 1, -4, -5, 10, -12, -70, -4}
, {21, -55, 28, 46, 21, 21, -95, -42, 31, 29, -9, 22, 17, 25, 10, 16}
, {-12, -29, -5, 7, -86, -6, -50, -27, 18, -45, -3, -9, -16, -15, 25, -15}
}
, {{-10, -47, -6, 34, 3, 14, -45, -4, -13, 1, -18, 5, 20, -6, -8, 16}
, {15, -18, 7, 27, 6, 21, -14, -55, 22, 12, -47, 12, -7, 12, 19, 5}
, {-6, -9, -22, -22, -21, -14, -18, -1, -3, -23, 8, -14, -5, -19, 1, -19}
}
}
, {{{-28, -11, -25, 55, 19, -33, 35, -22, -42, -11, 18, -55, -6, -36, -34, -26}
, {15, 32, 12, 29, -3, 4, 24, -44, -35, 27, -34, -53, 28, 3, -69, 9}
, {4, 23, 5, 27, -10, -3, 21, -41, -8, -10, -58, -33, 20, 6, -48, -4}
}
, {{-43, -4, -62, -41, -23, -61, 9, 42, -30, -46, 50, -16, -72, -53, -20, -36}
, {-56, -9, -77, -85, -20, -55, 2, 17, -42, -57, 9, -14, -76, -54, -11, -27}
, {-28, -45, -24, -44, -11, -21, -34, 22, -44, -19, 34, -1, -44, -34, -31, -12}
}
, {{10, -5, 13, 9, 13, 18, -12, 8, 8, 7, -3, 7, 15, 8, 10, 12}
, {10, 16, 8, 9, 21, 19, 12, 21, 16, 24, 14, 20, 2, 15, 15, 16}
, {11, 29, 15, 1, 12, 18, 8, 29, -12, 6, 6, 18, 13, 12, 11, 8}
}
}
, {{{-54, -13, -53, -11, -24, -29, 10, 10, -13, -80, 6, -14, -65, -50, -4, -52}
, {-16, -5, -48, -25, -51, -61, 23, -7, -58, -77, -48, -70, -24, -72, -42, -38}
, {5, 23, -42, 13, -22, -45, 40, -46, -40, -36, -85, -52, 0, -41, -12, -18}
}
, {{-6, -2, 20, -5, -8, 17, -27, 25, -8, 6, 30, 10, -2, 12, -6, 16}
, {-21, -35, -8, 9, -12, 5, -39, 16, -14, -4, 23, 18, -19, -2, 10, -7}
, {-24, -21, -14, -19, -14, -8, -1, 20, -1, -2, 1, 17, -29, -5, 31, -9}
}
, {{6, 26, -2, 7, 13, 4, 8, -4, 11, -2, 24, -4, 25, 9, 2, 8}
, {28, 20, 25, -2, 5, 12, 30, 8, 2, 8, 5, 23, 22, 16, -4, 9}
, {3, 29, 5, -21, -4, 4, 39, 14, -6, 9, 25, 14, 6, 11, 11, 20}
}
}
, {{{1, 3, 2, -17, 0, -8, 12, -6, -22, -7, 6, 10, 2, 8, -3, 4}
, {4, -24, -4, -5, -1, 2, 7, -1, 16, 4, 21, 11, 1, 5, 29, 12}
, {-1, -39, -10, 10, -4, -3, -17, 7, 13, -8, 5, 0, -13, 0, 14, 2}
}
, {{-11, -12, 4, -19, 8, -5, -4, -17, -51, -4, -7, -6, -7, 4, -46, -11}
, {17, -27, 3, -15, 8, 2, 2, 5, 6, 9, -10, 18, 8, 16, -1, 11}
, {-4, -19, 13, -19, 13, 18, -17, -3, -17, 15, 9, 12, -5, 17, 10, 7}
}
, {{-7, -21, -8, -33, 14, 11, -23, 42, -44, -5, 3, -6, 11, -22, -51, 6}
, {19, -15, 13, -5, 13, 10, -26, -15, 5, 19, -37, 2, 7, 3, -23, 1}
, {22, -12, 12, -13, -9, 11, -10, -41, -3, 9, -33, -11, 12, -7, 15, 7}
}
}
, {{{-12, 5, -2, -6, 9, -11, 13, 11, -3, -9, 9, -10, -6, -4, 12, -4}
, {0, 1, -2, -45, 11, 0, 3, 29, 13, -4, -1, 7, -24, -2, 14, 6}
, {-9, -15, -4, -29, 17, -3, 2, 25, -18, 14, -3, 2, 0, 2, -4, 4}
}
, {{3, 25, 8, -25, 7, -1, -24, 14, -16, 3, 26, 6, 0, -2, -19, 4}
, {3, 33, 19, -28, -1, 2, 4, 7, -7, 3, -11, -3, 32, 11, -5, 0}
, {4, 25, 12, -55, -13, 9, 8, 14, -16, -1, 7, -1, 21, 6, 1, 9}
}
, {{7, 14, -5, -10, -12, 10, -21, 6, 5, 6, 11, -1, 3, 2, 2, -3}
, {-6, 35, 6, 0, -16, -6, 30, -15, 5, -3, -10, -18, 9, 8, -7, -10}
, {-12, 28, -16, -1, -3, 0, 46, 15, 10, -11, -14, -7, 10, -10, 20, -1}
}
}
, {{{-7, -13, -9, 29, 4, 3, -6, 2, -20, -9, -7, 5, -5, -4, -18, 2}
, {14, -2, 6, -8, 3, 7, 1, -8, -19, -1, -26, 3, -5, 0, -5, 0}
, {4, -16, 1, -27, -1, -13, -2, 14, -21, -3, 11, 0, -16, -9, -2, -1}
}
, {{24, -15, 7, 5, -3, 11, 5, -1, -34, -12, -3, 11, 6, -9, 10, 3}
, {7, -6, 10, -8, -5, 14, -16, 10, -41, 5, -17, -3, 3, -7, -35, 17}
, {19, -2, 2, -9, -2, 21, 5, 7, -51, 22, 3, 22, 22, 10, -39, 14}
}
, {{3, 29, 6, -4, 7, -4, 12, -8, -26, 10, 22, 3, 5, 5, -16, -5}
, {-3, 18, -7, 21, 2, -5, 13, -43, -8, 1, -10, -31, 7, -3, -12, 3}
, {22, 12, 4, -1, 10, -13, 25, -39, -15, -13, -45, 15, 6, 7, -4, 3}
}
}
, {{{0, 0, 1, 23, 23, 3, -25, 12, -2, 9, 14, -4, 14, -8, -16, 6}
, {-1, 7, 4, 16, -6, -12, -24, -26, 10, -16, -38, -31, 5, 3, -20, -15}
, {-15, -20, -12, 2, 4, -15, -17, -8, 0, -12, -22, -7, -29, -14, 3, -11}
}
, {{-5, -25, -6, 13, 5, -2, -48, 13, 1, -5, -17, -1, -1, 8, 21, -12}
, {-9, -55, -22, -44, 2, -13, -72, 47, -3, -16, -2, 6, -19, -17, -18, -3}
, {-2, 8, 0, -18, 22, 7, -59, 22, -15, 20, 31, 15, 5, 0, -11, 3}
}
, {{14, 13, 8, -28, 16, 1, -14, 29, -6, 12, 8, 10, -9, 1, -9, 5}
, {19, 29, 19, -23, 12, 23, -24, 48, -24, 30, 9, 23, 29, 18, -8, 19}
, {9, 2, 5, -19, 2, 12, -12, 4, 1, -1, -10, -3, 8, 8, -11, -1}
}
}
, {{{-16, 0, -12, 12, -44, 6, 26, -2, 11, -16, 67, 24, -10, -4, 25, 1}
, {-30, -31, -27, 17, 2, -55, -2, -4, 33, -29, -17, -43, -13, -38, 10, -30}
, {9, -26, 7, 11, 28, 4, -6, 21, -45, 17, -19, 11, 6, 17, -67, 10}
}
, {{13, -19, 22, -41, -10, 19, -39, -16, 24, -19, 23, 1, 3, 20, 27, 14}
, {-43, -60, -27, 41, 2, -33, -16, -18, 12, -5, -46, -36, 9, -34, -10, -15}
, {13, -26, 19, 54, 23, 11, 1, 10, -16, 26, -38, 7, 3, 9, -5, 13}
}
, {{1, 0, -15, -32, -21, -16, -25, -14, 12, -29, -20, -6, -33, -4, -4, -28}
, {-21, -11, -20, 3, -14, -37, 24, -1, -13, -11, -30, -19, -10, -30, -46, -11}
, {24, -1, 7, 28, 13, 15, 10, -17, 2, 21, -32, 11, 21, 11, -2, 7}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    conv2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV2D_8_H_
#define _CONV2D_8_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_HEIGHT        6
#define INPUT_WIDTH         6
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )


typedef int8_t conv2d_8_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

#if 0
void conv2d_8(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]);               // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT

#endif//_CONV2D_8_H_
/**
  ******************************************************************************
  * @file    conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv2d_8.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_HEIGHT        6
#define INPUT_WIDTH         6
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int8_t
#define LONG_NUMBER_T int32_t


static inline void conv2d_8(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS / CONV_GROUPS], // IN

  const NUMBER_T bias[CONV_FILTERS],						                // IN

  NUMBER_T output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]) {               // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, pos_y, z, k; 	// loop indexes for output volume
  unsigned short x, y;
  int input_x, input_y;
  LONG_NUMBER_T	kernel_mac;
  LONG_NUMBER_T tmp;
  static LONG_NUMBER_T	output_acc[CONV_OUTHEIGHT][CONV_OUTWIDTH];

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        output_acc[pos_y][pos_x] = 0;

        for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
          kernel_mac = 0; 
            
          for (y = 0; y < CONV_KERNEL_SIZE_Y; y++) {
            input_y = pos_y * CONV_STRIDE_Y - ZEROPADDING_TOP + y;

            for (x = 0; x < CONV_KERNEL_SIZE_X; x++) {
              input_x = pos_x * CONV_STRIDE_X - ZEROPADDING_LEFT + x;

              if (input_x < 0 || input_x >= INPUT_WIDTH || input_y < 0 || input_y >= INPUT_HEIGHT) // ZeroPadding2D
                tmp = 0;
              else
                tmp = (LONG_NUMBER_T)input[input_y][input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][y][x][z];
              kernel_mac = kernel_mac + tmp;
            }
          }

          output_acc[pos_y][pos_x] = output_acc[pos_y][pos_x] + kernel_mac;

        }
      }
    }

    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        // Scale for possible additional precision of bias
        output_acc[pos_y][pos_x] = scale(NUMBER_T, output_acc[pos_y][pos_x],  WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

        // Scale bias to match accumulator
        output_acc[pos_y][pos_x] += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


#ifdef ACTIVATION_LINEAR
        output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
        // Activation function: ReLU
        if (output_acc[pos_y][pos_x] < 0) {
          output[pos_y][pos_x][k] = 0;
        } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc[pos_y][pos_x] > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc[pos_y][pos_x] = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
          output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        }
#else
#error "Unsupported activation function"
#endif
      }
    }
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif


  static q15_t bufferA[INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS];
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_basic_nonsquare(
#endif
                                      (q7_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q7_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q7_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q7_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS     32
#define CONV_FILTERS       64
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const int8_t conv2d_8_bias[CONV_FILTERS] = {21, 6, 17, 7, 26, 13, 5, 7, -3, -6, -7, 10, 13, -9, 7, 11, 0, 13, 1, -9, 15, -5, 12, -11, 21, 11, 13, -4, 13, 24, 18, -5, 0, 20, 26, -11, 19, -5, -13, 10, 3, -12, -4, 0, 3, 12, -8, 28, -10, 3, 1, 5, 9, 0, 0, -7, 2, 0, -3, 3, -1, -11, 8, 0}
;


const int8_t conv2d_8_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-10, -8, 11, 5, 17, 6, -37, -12, 0, -13, -2, -7, -13, 5, 0, -3, -5, 0, -16, -14, 7, -6, 3, 15, -5, -8, -2, 11, 5, 6, -1, -2}
, {22, -5, 10, 12, 9, 11, 14, 8, 6, 11, 7, -5, -21, 2, -20, 3, 3, 2, 1, 9, 8, 17, 11, -2, 2, 1, -12, 1, 3, -5, 1, -14}
, {3, -13, -25, -36, -64, 4, 32, -28, -7, -23, -31, 7, 2, -29, 15, -20, -33, -3, -4, 18, 1, 18, 1, -9, -26, 8, -43, -34, -4, -6, -54, -7}
}
, {{-6, 0, 1, 5, 33, -22, -15, 0, 47, -5, -1, 18, 20, -2, 5, 7, -18, -1, 26, -3, -7, -15, -7, 12, 14, 5, 7, 5, -11, -2, -7, 15}
, {8, 26, -4, 1, -10, -23, -54, 20, 11, 13, 2, 30, 14, 7, 15, 18, -29, 4, 7, -23, -31, -42, -37, 17, 9, 20, 0, 31, -19, 4, -2, 12}
, {1, -17, -11, -10, -14, -23, 18, -3, -16, -36, -14, -18, 29, -29, -29, -10, -42, -32, -11, -9, -26, 9, -47, -47, -20, -7, -18, -39, -33, -35, -28, 12}
}
, {{15, 20, 4, -5, 9, 1, 3, 18, 3, 1, 2, 17, 0, 9, 10, 13, 8, -5, 7, 1, 11, 9, 14, -4, 16, 25, 6, -7, 12, -1, 7, 18}
, {23, 34, 0, 8, 9, 7, -31, 12, 34, 13, -2, -6, 20, -2, 10, 1, -19, -8, -15, 2, -1, 6, -4, 9, 11, -59, -30, 17, -10, -3, -21, 24}
, {-6, 5, 5, 5, -6, -33, 7, 2, -8, -25, -3, 2, -10, -3, -13, -27, -15, -18, -48, -40, -19, -16, -11, -1, -2, 2, -1, 1, -23, -19, -3, -3}
}
}
, {{{-26, -2, -13, -12, -27, 0, 5, -23, -35, 8, -11, 7, 15, -9, 14, -10, 10, 10, -2, -11, -13, 14, 5, -5, -3, -9, -5, 3, -7, 9, -10, 13}
, {-22, -16, 0, 5, -22, 1, -3, -3, -12, 18, 11, -2, -20, 2, -2, -12, 15, 4, -1, -18, -5, -21, -5, 13, 0, -3, -15, 16, 4, 11, 2, -7}
, {6, -2, 9, 18, 0, -6, 4, -13, -10, 2, -7, -48, -67, -22, -30, 8, 18, 4, 28, -25, 16, -7, 16, -15, -8, 13, 13, -36, 17, -20, 21, -46}
}
, {{-9, -62, -8, -19, -38, 10, 19, 5, -27, -9, -20, -20, 18, -23, -2, -8, -20, -12, 0, -3, -6, 8, -9, 8, -7, 12, -7, -2, 1, -4, -5, 9}
, {-5, -52, -16, -34, 1, 9, 24, 3, -27, -13, -19, -8, 21, -10, -5, -11, -13, 5, 3, 19, -3, 7, 4, 11, -21, -6, -23, -8, -9, -1, -20, 1}
, {9, 16, 8, 1, 8, -10, -14, 24, 7, 9, 7, -48, 42, 14, 7, -7, -14, 2, -23, 8, -11, -46, 4, 12, 18, -40, -7, 22, -5, 9, -13, -32}
}
, {{-19, 1, 1, 12, -14, 6, 20, -18, 5, -3, 2, 24, -8, 1, -7, 12, 14, 1, 19, 2, 2, 21, 13, -4, 0, 10, 4, -5, 19, 13, 0, 8}
, {-51, -7, 7, -11, -30, 11, 6, -22, -22, 2, -2, 19, -19, -15, 2, 18, 7, -4, 14, -28, 14, 6, 23, -2, -6, 34, 12, -13, 13, -1, 16, 8}
, {5, 6, -6, -5, -9, -5, -39, -12, -21, 0, 2, 28, -13, 3, -5, 12, 7, -9, -4, -26, 4, -10, -33, -16, 21, -3, -16, -3, 2, -13, 0, 11}
}
}
, {{{19, 23, -23, -2, -17, -14, -14, 19, 19, 11, 15, 27, -31, -32, 2, 8, 12, -2, 0, -22, 16, -8, -43, -9, 21, -10, -22, 3, 11, -10, -19, 3}
, {-4, 6, -41, 2, 0, 7, 10, -2, 10, 0, -15, 17, 22, -17, 14, 3, -48, -11, 11, 35, -8, 0, -63, -21, 8, 7, 2, -4, -12, -19, -9, -8}
, {-10, -7, 12, -6, 3, -34, -10, 2, -3, 7, 8, 28, 14, -7, 9, -14, -5, -7, 1, 1, 20, -27, -25, 0, -2, 5, -15, 3, 9, 0, -13, 15}
}
, {{-6, 29, -56, -30, -26, -4, -18, -15, 2, -41, -28, 15, -15, -43, -20, 36, -32, -38, 4, -25, -48, -10, -55, -28, 23, -2, -44, -34, -60, -40, -46, 25}
, {-45, 21, -16, 8, -19, -66, -1, 2, -16, -3, 6, 9, -14, 11, 1, -20, 24, -33, -8, -30, 0, -46, -41, -15, 5, 3, 13, 2, -8, 6, 11, 13}
, {6, 12, -1, 4, -2, 10, -25, -24, 13, 10, 0, 22, -28, -12, 8, 11, 9, -22, 4, -16, 19, 4, -6, -24, 4, 20, -12, 4, 13, 11, 4, 0}
}
, {{22, 29, -11, -26, -6, -39, -19, -59, -8, -6, 8, 14, -4, 10, 11, -8, -1, -57, -6, -26, 16, -30, -23, -32, -19, 8, -13, 3, 2, 9, 4, 13}
, {4, 7, -6, 21, -17, -32, -75, 5, -10, 12, 5, 23, -5, -8, 11, 7, 11, 18, -5, -17, 10, -50, 0, -24, 5, -51, -26, -1, 7, 2, -5, 3}
, {13, 3, -14, 2, -13, -19, 10, -19, 7, -16, -15, -3, -19, -14, -10, 2, -17, -10, -4, -14, -34, -11, 0, -7, -7, -7, -22, -29, -20, -28, 9, -7}
}
}
, {{{9, 17, -12, -3, -3, 5, 10, 3, -3, -11, 6, -10, 21, 4, 4, -6, 10, -9, -4, 10, -10, 14, 9, -8, 3, -18, 5, 0, -5, 2, -8, 4}
, {-3, -32, 9, 15, 22, 19, 1, 7, 4, 21, 7, -6, 0, 8, 11, 0, 10, -11, 9, 10, 10, 0, 25, 9, 12, 0, 15, 10, 3, 7, 10, -1}
, {22, 0, -2, 4, 10, -2, 7, 8, 8, -10, -8, -38, -6, -6, -7, -6, -21, -1, -2, 15, 3, 7, 0, -10, 0, 0, 7, -13, -4, -7, -8, -41}
}
, {{-8, -17, 17, 8, 0, 15, 29, 0, -12, -21, -5, -36, -10, -17, -29, -29, -8, 4, 5, 3, 4, 6, 5, -2, -1, 16, 16, -14, 5, -17, 3, -30}
, {-13, 9, -15, -18, 3, -7, -19, 4, -26, -6, 0, 6, 18, -8, 13, -3, -6, -7, -16, 2, -4, -23, 0, -9, 4, 16, -8, 10, 0, 15, -2, 17}
, {14, 25, 2, 13, 5, -13, -23, 15, 17, 5, 12, 7, -9, -9, -28, 22, -7, -33, -3, -58, -21, -4, -22, -34, 10, -48, -6, -4, -33, -50, -39, 32}
}
, {{5, -22, 11, -9, 5, 5, 10, -12, -18, -4, -15, 11, 0, -5, 8, -10, 19, 11, -7, 0, 3, 18, 20, -12, -21, 8, 0, -5, 12, 5, 22, -10}
, {3, 17, -20, 0, -38, 9, -20, 6, -4, 11, -3, 16, 2, -16, 17, 6, 8, 7, -2, 2, 4, -11, 3, -6, 0, -15, -11, -4, -2, -9, -21, 26}
, {16, -7, -12, -23, 2, -20, -32, -9, -1, -15, -27, 19, -25, -4, -16, -11, -25, -14, -48, -44, -42, 1, -43, -30, -2, -8, -32, -21, -69, -20, 1, 3}
}
}
, {{{-4, -13, 3, -20, -14, 4, -6, 2, -21, -3, -24, 18, 8, -6, 10, 1, -13, 5, -8, 2, 2, -2, 7, 3, -10, -13, -8, -8, -8, 0, -14, 11}
, {-66, -22, -2, 2, -24, -39, 31, 1, -33, 7, 22, 15, -24, 16, 2, -2, 12, 4, 15, -15, 20, -33, -21, -27, 10, -8, -13, 19, 9, -11, 2, 1}
, {10, -2, 3, -5, 10, 15, -22, -7, -5, 15, 7, 9, -28, 6, -40, 13, -4, 3, 6, -14, 14, 8, 1, -3, 23, -45, -20, -1, 8, 1, -27, -9}
}
, {{-14, -72, 6, -6, 8, 2, -16, -1, -30, -4, -2, 13, 26, -1, 9, -1, 4, -4, 2, 0, 9, -6, 3, 2, -8, 3, 10, -5, -6, 0, 6, 15}
, {0, 14, -10, 13, -10, 2, -4, 5, 1, 5, 7, 13, -8, -5, -13, 16, 10, 9, 30, -2, 19, 20, 4, 9, 1, -27, -14, -1, 21, 17, -23, -17}
, {23, -7, -1, 17, 12, -5, -9, -4, -2, -51, -13, -22, 19, -6, -31, -9, -56, 4, -33, -2, -20, 3, 11, 7, 10, -6, 3, 3, -21, -31, -4, -27}
}
, {{6, 7, -8, 10, -5, 9, 9, -8, 4, -3, -10, 5, 20, -8, -6, -11, -15, -4, -8, 25, -21, 0, -4, -1, 5, 3, -7, -14, -5, -12, -9, 4}
, {-34, -12, 9, 5, 15, -21, -8, -10, 8, -7, 12, 10, -4, 6, 7, -24, 9, -6, -11, -12, 4, -29, 9, -6, 9, 12, -3, 15, -9, -2, 15, -3}
, {-13, 1, 2, 12, 13, -16, -1, 17, 2, -27, 14, -1, 19, 19, -15, -2, -7, -5, -4, -26, -8, -32, -34, 12, 12, -16, -10, 8, 1, -8, -15, 12}
}
}
, {{{-6, -15, 1, -6, -3, -4, -19, 25, 4, 9, 13, -20, 6, 4, -6, -7, 8, 2, 3, -8, 12, -12, -16, 17, 14, -15, -11, 14, 5, 19, 5, 1}
, {-15, 14, -9, -1, 7, -31, -17, 13, 5, 1, 16, 6, -7, 0, -7, 20, -9, -11, 6, -30, -3, -40, -9, -4, 16, -18, -7, 1, -14, -8, -13, 7}
, {9, 20, 10, 1, 4, 6, -61, 7, 3, 28, 4, 34, -2, 2, 30, 9, 31, -26, 9, -43, -10, -13, 11, -12, 1, -3, -3, 20, 17, 0, 0, 35}
}
, {{14, -9, 5, -10, -8, 15, -7, 3, -9, -2, 5, 0, -7, -3, 3, 28, -5, -16, -13, -4, -5, 5, -10, -8, 2, -7, -8, 3, 10, 4, 7, -1}
, {10, 10, -8, -17, -26, 25, 4, -13, -13, -5, -4, -29, -6, -21, -6, 11, -5, 8, 4, 7, -3, 3, -4, -7, -4, -32, -20, -13, -2, -17, -20, 4}
, {16, -5, -21, -25, -16, 18, 11, -6, -17, -9, -27, -12, -7, -42, -31, 10, -24, 5, 24, -4, 7, 20, 9, -2, -5, -6, -40, -31, -10, -12, -33, -18}
}
, {{9, 4, 8, 23, -1, -13, -15, 10, 12, 2, 13, 14, -5, 7, -3, 5, 0, 11, 8, -12, -4, -15, 14, 1, 11, 13, 10, 6, -5, 0, 7, 3}
, {1, -6, 7, 14, 1, -27, 26, -16, -10, -2, -4, 26, -49, 18, -2, -16, 6, 8, -28, -24, 1, -26, 31, -4, -22, -1, 1, -1, -1, 13, 9, -19}
, {-19, -4, 3, 18, 12, -5, 28, 0, 18, -10, 2, -38, -40, 20, -14, -48, 2, 1, -45, -25, -28, -32, 33, 15, -24, 11, 4, 15, -9, -10, 2, -39}
}
}
, {{{-23, 9, 0, 6, -7, 0, 0, -11, -16, 13, -15, 31, 7, 4, 8, -6, 21, -9, 16, -13, 2, -1, 4, -12, 10, 13, 0, 7, -4, 2, 9, 24}
, {-11, 5, 3, 13, 8, 19, -9, -15, 6, 6, -7, -8, -23, -7, 8, 14, 12, 12, 20, 3, 6, 9, -10, -6, 4, -10, -5, -3, 10, 6, 18, -17}
, {16, -15, -13, 4, 4, 7, -9, -17, -2, -26, -34, 5, -34, -28, -40, 9, -39, 17, 4, -1, -10, 22, -9, -19, -21, 31, -3, -55, -22, -26, 24, -55}
}
, {{-13, -16, 14, 3, -6, 1, 15, 2, -10, -9, -8, -31, 3, 10, -18, -26, -9, 11, 0, -4, -1, -5, -7, 11, -8, 37, 14, -1, -5, -5, 8, -9}
, {1, -10, -5, -4, 11, -10, 1, -2, 13, -1, -17, -4, -12, -4, 2, -31, 14, 8, -23, 13, -19, -6, -2, -8, -24, 19, 8, -5, -5, -8, 4, -18}
, {-38, 23, 5, 5, 18, -27, -17, 18, 9, 7, 25, -26, 42, 19, 12, -27, -9, 6, -24, 13, -19, -28, 3, 15, 12, -32, 1, 22, -17, 2, -11, 23}
}
, {{-26, -13, 6, -2, -9, -15, -10, -2, 3, 5, 16, -6, -14, 6, -1, -51, 12, 6, -15, -20, -5, -21, -9, 0, -7, 3, -6, 9, 2, 6, 2, -8}
, {-59, -2, 5, -16, 0, -9, -26, -7, -16, -8, 10, -32, -24, 5, -3, 3, 4, 3, -7, -30, 3, -4, -17, -14, -5, -9, -8, 10, 7, 3, 3, -15}
, {-8, 7, -3, -21, -20, 1, -28, -4, -13, 7, -2, 14, -8, -4, 8, 10, 31, -6, 7, -27, 12, -5, -11, -23, -16, -10, 3, 7, 13, 25, 28, -3}
}
}
, {{{-6, -10, -13, -2, -32, -17, 8, -12, -40, 2, 1, 10, 2, 4, 1, -1, -14, -5, 14, 17, -1, -2, -30, -13, -9, -2, -7, -12, -10, 0, -15, -10}
, {7, 28, 1, -19, 8, -1, -5, 14, 12, 3, 14, 19, -22, -1, 10, 7, 4, 1, 17, -2, 11, -3, 2, -4, 1, 9, 8, 11, -3, 5, -2, -5}
, {27, -7, -8, -8, -9, -11, 8, 22, 6, 28, 19, 9, 16, -6, 37, -8, 17, -4, 11, 16, 13, -4, 4, 11, 10, -6, -8, 8, 14, 3, -26, 23}
}
, {{-2, -4, -4, -30, -30, 4, -19, 8, -19, 14, -1, 6, 7, -10, 3, -11, 8, -5, -4, -16, -2, 7, 4, -11, 4, -16, -38, 4, -3, 3, 6, 10}
, {33, 37, -14, 3, 3, 20, 2, 10, 17, 8, 15, -3, 15, 0, 10, 24, 4, 3, -21, 0, -7, 22, -9, -5, 11, -11, -26, 11, 0, 13, -7, 23}
, {-5, 7, -31, -10, -17, -17, -23, -14, -5, -2, -6, 12, 17, -20, 14, 3, -42, -38, -9, -21, -33, -14, -52, -23, 4, -35, -71, -27, -26, -9, -85, 16}
}
, {{20, 27, -2, -17, -34, -1, 19, -13, -22, 5, 5, 25, 18, -7, 16, 22, 18, -8, 19, 9, 0, 12, -2, -15, -9, 16, -6, 12, 1, 3, 11, 5}
, {20, -6, -31, -21, -17, 8, -7, -11, 2, -2, 3, 27, -46, -5, -24, 4, -9, 4, 14, -7, -16, 14, -51, -21, 2, -9, -19, -7, -29, -34, -46, -13}
, {10, 12, -43, -41, -22, -12, -13, -7, -23, -15, 0, 8, -2, -9, 2, 10, 1, -52, -37, -17, -11, 5, -26, -17, -51, -20, -54, 2, -9, 3, -23, 18}
}
}
, {{{2, 13, -22, -4, -7, -4, -6, 19, -8, 10, -9, 26, 1, -2, 21, 10, 18, 3, 24, -11, -5, -4, -19, -26, -6, -2, -11, 5, -1, 3, 14, 4}
, {-11, 26, -4, 1, 6, -16, 2, 10, 17, 9, 11, 34, 5, 0, 7, -2, -5, -9, 25, 15, -3, -18, -43, -28, 1, 10, 14, 2, -14, 13, 8, 5}
, {7, 10, -16, -16, -37, -15, -14, 2, -7, 22, 4, -14, -45, -28, -35, 20, 0, -15, 28, -16, 14, -20, -38, -33, 17, -31, -24, -12, -10, -34, -34, -20}
}
, {{4, 6, -5, -11, 8, -13, -41, -3, -1, 6, -8, 1, -5, -2, -10, 3, -6, -8, 3, -16, -22, -10, -21, -6, 8, 13, 5, 10, 9, 4, -2, 1}
, {29, 12, 13, 16, 3, 0, -79, 7, 24, 25, 0, -27, -2, 20, 10, 11, 1, 5, -8, -8, -15, -22, 0, 0, 13, -15, -15, 2, -10, 7, 0, 19}
, {-33, -3, 3, 4, 11, 1, -12, -11, 10, -50, 0, -39, -33, -8, -43, -25, -11, -2, -22, -59, -16, -7, 5, -11, -14, -5, -11, -21, -6, -10, 12, -37}
}
, {{10, 13, -26, -16, -11, -22, -9, 9, -14, 5, 4, 5, 7, -30, 20, 21, -6, -1, -1, 5, 1, -22, -14, -13, -4, -54, -43, 0, 6, 13, -38, -4}
, {16, -17, -10, -8, -5, 8, -16, 18, 3, -2, 5, -39, 8, -11, -18, -4, -21, 13, -2, 7, 10, 8, 14, 15, 4, -25, -7, 5, -6, -2, -43, -13}
, {14, -14, -2, -40, 13, 3, 16, 0, -20, -31, -11, -22, -5, 8, -3, 9, -21, 9, -2, 10, -13, 12, 25, 9, -37, 3, -9, -17, 9, 6, -24, -23}
}
}
, {{{4, 17, 4, -27, -8, 9, 3, -2, -8, -10, 4, 0, 6, -1, -4, -6, 20, -1, -5, 10, 15, 10, 16, -4, 21, -7, 0, -4, -1, -10, -3, 0}
, {-4, 42, -11, 17, 8, 7, 0, 18, 13, 22, 11, 11, 16, -13, 16, 26, 14, 2, 10, -4, -2, -17, -15, -5, 24, -12, -2, 6, 10, 12, -2, 21}
, {16, 10, -50, -27, -37, -6, -1, -20, 7, -11, 7, -5, -28, -49, -42, 8, 4, 3, 17, -8, 10, -6, -25, -18, -2, -24, -32, -35, -3, -26, -83, -31}
}
, {{1, 31, -4, -3, -2, 2, -1, -17, -3, -13, 5, 9, 8, -3, 12, 38, 9, -15, 12, 18, 12, 19, 4, -26, 17, 14, 8, 1, 0, 0, 9, 4}
, {3, 18, -9, -12, -16, 16, 19, -17, -8, -7, -5, 29, -10, -18, -50, 3, 23, -18, 20, -14, 3, 5, -17, -41, 4, -3, -8, -7, 3, -49, -30, -17}
, {1, -17, -11, 9, -10, -12, -8, -31, -22, 19, -19, 12, -59, -23, 3, -2, 32, -83, -23, -60, -16, -1, -78, -61, -42, 10, 0, -1, -21, 1, 20, -6}
}
, {{1, 9, -2, 7, -26, -4, -27, 7, -1, 25, 11, 38, -25, 5, 2, 9, 11, -2, 8, -26, 1, 2, -5, -13, 8, -14, -3, 4, -2, 9, 9, 28}
, {4, -29, 15, -12, -12, 7, -18, -48, -19, -2, -9, 1, -73, -6, -13, -26, 23, -15, -23, -37, 15, 11, 18, -15, -48, 5, -5, -2, 6, 6, 16, -15}
, {-67, -54, 1, 7, 6, -8, 16, -11, -18, 6, 3, 5, 9, 13, 15, -13, 13, 17, 14, 24, 21, 8, 26, 14, -5, 37, 27, 18, 7, 16, 15, -13}
}
}
, {{{3, 14, -2, -27, -10, 17, -6, 11, -1, -31, -12, -37, -4, -12, -28, 9, -22, 16, 17, 9, 3, 4, 2, 13, 7, -27, -22, -24, 3, 5, -41, -11}
, {5, -36, -23, -33, -6, 5, -7, -3, 5, -14, -27, 14, 2, -2, 6, 6, -15, 10, -8, 0, -8, 5, -11, -1, -10, -5, 5, -4, -16, 9, -19, 17}
, {-28, -9, -5, -9, 9, -10, -1, 15, -15, 7, 4, 10, 5, 10, 12, -19, 24, 12, 0, 8, 9, -9, 2, 6, 0, 4, 4, 7, 3, 5, 2, 1}
}
, {{-8, -13, 15, 7, 25, -12, 4, -3, 30, -13, -24, -13, -38, -24, -22, -35, -39, -6, -43, -23, -55, -16, -30, 2, -8, 24, 25, -22, -50, -36, -4, -29}
, {-17, -38, 4, 5, 23, 7, 16, -48, 17, -51, -55, -33, 9, -41, -27, -67, -45, -31, -44, 15, -24, -1, -7, -13, -39, 28, 5, -55, -40, -43, -12, -14}
, {4, -45, -18, -20, -5, 15, 8, 4, -2, -13, -29, -11, 18, -30, -10, -13, -24, -9, -4, 8, -1, 3, 11, 17, -22, 4, -29, -10, 5, -4, -18, 21}
}
, {{-4, 8, 11, 40, 26, 3, -1, 3, 36, 1, 12, 6, 18, 4, 10, -1, -2, 9, -3, 10, 3, 3, 8, 7, 8, 18, 11, 5, -10, 10, 7, 17}
, {-17, 26, 19, 19, 12, 8, -3, 24, 43, 9, -7, -18, 20, 13, 8, -30, -6, 10, -13, 8, 9, 6, 15, 15, 12, 24, 22, 8, 5, 14, 7, 0}
, {5, -22, 9, 19, 6, -2, 26, -11, -2, -1, 3, -21, -29, 9, -24, 0, 2, 6, -11, -14, -9, 14, 6, -4, -30, 8, 10, -7, -5, 1, 3, -14}
}
}
, {{{-12, 8, 18, -2, 6, -37, -16, 9, 4, 4, 9, -2, 13, 10, 21, -16, 8, -5, -23, 8, 14, -30, 18, 7, 19, 13, 5, 19, 23, 3, 11, -9}
, {-20, 16, 0, 24, 14, -3, -36, -7, 24, -1, 4, 2, 3, 15, -11, -22, 4, 17, -6, -15, -6, -29, 7, 10, 4, -7, -7, 8, 1, 2, -10, 11}
, {0, -12, 0, 4, 6, 13, -5, -9, -7, 7, 4, -13, -20, 6, 1, -4, 14, -9, 9, -21, -4, -9, 10, -16, 6, 3, 3, -1, 9, 9, -1, 1}
}
, {{17, 2, 8, -4, -9, 11, -23, 4, 1, -17, 0, -35, -8, -2, -7, 9, 6, 19, 7, 2, 5, 1, -3, 12, 4, -12, 2, 7, 1, 6, -2, -19}
, {10, -29, 19, -14, 5, 11, 29, -10, 6, -4, -6, -38, 11, -4, -18, -19, -2, 11, 5, 3, 18, 19, 14, 19, -26, 11, -13, -5, 19, -2, -14, -15}
, {2, -18, -4, -27, 8, 18, 17, -15, -22, -33, -15, -25, -2, -11, -1, 1, 4, 4, 15, 8, 10, 4, 22, 11, -41, -3, -4, -13, 17, 11, -26, -26}
}
, {{-9, 0, 15, 8, -3, -8, -9, 6, 18, -16, -15, -45, -13, -11, -36, -29, -26, 2, -14, -20, -17, -17, 20, -14, 6, -12, -11, -13, -5, -24, -1, -42}
, {4, 7, -3, 10, -9, -34, -3, -1, 27, -20, -1, 8, -8, -9, -5, -24, -8, -18, -31, -9, -17, -24, 37, -3, 5, 5, 6, -14, -8, -18, -27, 1}
, {2, 8, 6, 6, 0, 3, -24, -2, 3, 2, -7, -2, 0, -4, 9, -14, 3, -4, -16, -15, 5, 1, 30, 15, -13, 14, 14, 5, 7, 7, 1, 24}
}
}
, {{{14, 30, -16, -3, -15, 6, -15, 15, 45, -3, 18, -16, -13, -24, -9, 9, -25, 1, -2, -1, -2, -10, -15, -10, 11, 5, -31, 6, 9, -13, -30, -19}
, {7, -35, -40, -8, -25, -1, 6, -10, -22, 0, -20, 11, 5, -17, -4, -3, -17, -8, 13, 20, -3, -1, -36, 1, -22, -3, -8, -9, 6, 6, 10, 12}
, {-30, 8, 4, -7, 3, -65, 3, 6, -3, 4, 15, 25, -7, 10, 24, -2, 12, -23, 12, -9, 10, -27, 3, -15, 5, -5, 7, 10, 12, -3, 4, -6}
}
, {{-6, 45, 1, 5, -9, -8, -2, -9, 38, -40, -16, -11, -37, -20, -28, 12, -25, -36, -4, -23, -34, 4, -29, -34, 26, 4, 3, -25, -47, -37, -2, -2}
, {-79, -4, -19, 2, -7, -62, -88, 9, -30, 15, 10, 8, 4, 10, 19, -6, 1, -14, 5, -22, -4, -47, -19, -2, 9, -1, -5, 6, 9, 7, 17, 8}
, {11, 15, -6, 13, -1, 4, -46, 1, 9, 15, -8, 20, -60, -4, -16, 6, 0, 0, 8, -17, 11, 9, -20, -24, 8, -42, -6, 0, -13, -18, 0, -15}
}
, {{-4, 24, 24, 14, 17, -11, -18, -12, 22, -9, 1, 2, -17, 16, 3, 9, 7, 1, 6, -27, -11, -9, 6, 2, 1, -32, -5, 6, 2, 4, -4, -6}
, {0, 5, 0, 23, 10, 13, -41, 1, -11, 2, -6, -5, 17, 8, 9, -7, -17, -7, -16, -6, -11, -12, -22, 8, -1, -35, -45, -5, -4, -10, 1, 9}
, {-2, -2, 12, -6, 3, 4, -42, 0, 6, -14, -13, -47, 0, -2, -23, -23, -18, 19, -48, -22, -9, -8, 13, 17, -30, 18, -14, -4, -4, 3, 19, -11}
}
}
, {{{-2, -27, 19, 11, 34, 5, -7, 15, 6, -17, -24, -6, -3, 0, -16, -2, 3, -5, -4, 5, 2, -9, 30, 2, -21, 9, 14, -1, 8, -3, 10, -37}
, {-29, 3, 7, 15, 10, -50, -11, -14, 19, -2, -2, -1, -6, 6, -14, -13, -14, -31, -32, -10, -32, -62, -17, -4, -20, 11, 6, -13, -28, -17, 7, 4}
, {-6, 18, 19, 12, 24, -8, -24, 2, 19, 7, 4, 11, -6, 13, -5, -26, 0, -21, -35, -58, -5, -54, 9, -10, 1, 9, 7, 12, -12, 1, -4, 24}
}
, {{11, -10, -8, -20, -26, -3, -6, -5, -35, 11, 20, 11, -1, -11, 4, -21, 13, 4, 2, 2, 7, -8, -4, -3, -10, -10, 6, 1, 10, 3, -4, 1}
, {-21, 13, -4, -17, -7, -20, -41, 16, 0, -1, 19, -15, 0, 5, 15, 6, 12, 5, -8, -8, 13, -18, -7, 0, -15, -32, -19, 15, 10, 2, -27, 7}
, {-23, 12, 2, 5, -6, 2, 6, 5, 1, 10, -9, -4, -21, 12, 1, -17, -17, 5, -18, -17, -10, -1, 7, 1, 15, 1, -13, -4, 3, 23, 1, -17}
}
, {{21, 0, 7, -25, 2, 6, 7, -5, -7, -2, -15, -22, -14, 2, -32, 36, 9, 3, 16, -11, -2, 23, -2, -15, -15, -12, 2, -6, -3, -4, 7, -16}
, {15, -12, -20, -28, -15, 8, 8, -4, -27, 2, -29, -2, 8, -14, -5, 16, 0, -1, 16, 11, 18, 3, -3, 5, -8, -50, -44, -8, -3, 22, -30, 0}
, {2, -21, -7, -3, 8, 19, 28, 7, -2, -1, -6, -34, 8, 1, 16, -28, 0, 21, 0, 22, 10, 28, 13, 22, -15, -9, 4, 5, 8, 20, -5, -16}
}
}
, {{{10, 36, -11, 2, 11, 16, 15, 16, 24, 19, 14, 4, -10, 15, 0, 11, 4, 13, -2, -4, 1, -6, 18, 16, 10, 8, -9, 25, 5, 13, -1, -6}
, {6, -1, -68, -25, -45, -3, -4, -3, -2, 6, -6, 8, 22, -24, 15, -6, -18, -5, -8, 18, -10, -11, -12, -10, -9, 1, -11, -28, -4, 0, -34, 20}
, {2, 14, -37, -9, -31, -12, -24, 10, -5, 3, 14, 0, 16, -17, 10, 24, 6, 1, 29, 9, 16, 0, -25, 1, 11, -37, -48, 6, 14, -5, -61, -1}
}
, {{18, 4, -7, -4, 7, -14, -3, -12, -7, 14, -3, 37, -28, -3, -14, 34, 8, -13, 13, -21, -31, -5, -22, -9, 3, 2, 6, -12, -7, -10, 12, -2}
, {-23, -49, 6, 13, -4, -12, 5, -34, -16, 0, -15, 11, -10, 7, -8, -15, -2, -18, -8, -27, -20, -21, -22, -31, -23, 11, 2, -6, -14, -12, 8, -9}
, {-11, -9, 15, 14, -9, 0, -7, -24, 19, 2, -6, -2, -30, 10, 11, -11, 8, -12, -11, -37, 6, -22, 22, -7, -11, 21, -3, 6, 12, 17, 7, -6}
}
, {{3, 13, -4, 11, -4, -1, -13, -9, -7, 8, 14, 15, 7, 2, 19, 11, 2, -13, 11, -3, 7, -23, -11, 5, 5, -17, 7, -3, 2, 5, 9, 6}
, {-11, -27, -11, -13, -30, -3, -29, 2, -42, 7, 10, -10, -14, -17, -1, 2, -18, 10, 4, -4, 9, -11, 8, -4, 11, -44, -12, -13, 3, -17, -37, 16}
, {7, -4, 5, -12, 1, -1, 14, 12, -35, -2, 1, -7, -26, 13, -24, 11, -8, 1, 4, -9, 6, 21, 15, -8, -3, 9, -7, -3, 5, -16, 1, -39}
}
}
, {{{-5, 18, -13, -40, -32, -14, 14, -22, -53, 9, -9, 14, 10, -21, 10, 6, 27, -11, 13, 10, 6, -9, -15, -37, -13, -3, 0, 6, -3, 8, 9, 4}
, {5, 9, -4, -3, -4, 5, 6, 4, -17, 20, 13, 9, -7, 1, 10, -12, 15, -7, 9, -5, 17, 2, -7, 9, 14, -11, -10, 8, 1, -2, -7, -15}
, {5, 6, 2, 2, 21, -10, -14, -5, 6, -1, 3, -5, -5, -3, 8, 13, -5, -2, 20, 18, 24, 3, -9, 0, 5, -2, 8, -14, 17, 5, -12, -3}
}
, {{-1, -39, -15, -16, -27, 4, -10, 26, -29, 19, -4, -19, 7, -5, 13, -7, 5, 0, -5, -8, 7, 16, -4, 2, -9, -23, -13, 3, 5, -1, 5, 9}
, {14, -13, -30, -11, -1, 16, 4, 11, 5, -19, -14, 3, -1, -8, -2, -3, -37, 11, 13, 12, 4, 14, -5, -2, -8, -17, -24, -8, 12, -10, -51, -20}
, {-27, 21, -2, 3, -18, -44, -12, 4, 16, -14, -9, -8, 25, -16, 17, -18, -32, -20, -63, 12, -31, -42, -30, 5, -1, -2, -4, -10, -36, -4, -10, -8}
}
, {{-14, -37, 6, 0, 3, 7, 36, -17, -4, -16, -25, 21, -6, 7, -8, -16, 12, -9, 2, 15, -31, 33, 24, -24, -34, 20, 25, 4, -12, -16, 26, -25}
, {-9, -3, -8, -16, -33, -32, 29, -39, -19, -4, 13, 9, -6, -10, 5, -6, 15, -25, 11, -9, -5, -17, -11, -47, -3, 16, 11, -12, -7, -11, 2, 10}
, {-6, 13, -16, 2, -23, -32, -54, 18, 5, 7, 17, 24, 9, -14, 1, 10, 4, -23, 8, -13, 1, -24, -31, -18, 11, -63, -63, 0, 0, -5, -32, 16}
}
}
, {{{8, 2, 6, -3, -3, -18, 11, -17, -7, -7, -18, -5, 14, -2, 0, 6, 7, -5, 8, -15, -9, 2, 0, -4, -2, 15, 8, -19, -9, -11, -2, 6}
, {-36, -25, 3, 1, 6, -6, 5, 12, 9, 8, 19, 3, -6, 12, 6, -19, 17, 8, -23, -13, 1, 2, -2, 13, 19, -2, 3, 19, -5, 1, 8, 0}
, {-34, 9, 5, 6, -8, -27, -44, 4, -1, 19, 12, -2, 5, 9, 5, 10, 32, -8, 12, -30, 21, -52, -3, 6, 12, -1, 5, 12, 15, 19, 10, 13}
}
, {{-24, -1, -6, -49, -27, 1, 21, -43, -9, -28, -26, -6, -13, -22, -4, -8, 22, -3, 8, 6, 0, 19, 8, -8, -45, 13, 22, -17, 3, -2, 1, -16}
, {-28, 26, -35, -17, -20, -16, -10, 7, -15, 1, 0, 9, -5, -3, 8, 10, 0, -8, 4, 2, 3, -14, -25, -14, 10, 9, -18, 3, 5, 17, 3, -8}
, {-6, 10, -5, 6, -7, -8, -67, 12, -4, 7, 18, 8, -12, 10, 10, 9, -1, -13, 3, -25, -1, -25, -25, -24, 19, -76, -46, 15, -6, -18, -11, -8}
}
, {{1, -14, 12, -4, -15, 5, 17, -33, -13, 1, -19, 12, 0, -4, 10, 6, -4, 19, 12, 15, 4, 17, 24, 4, -23, 16, 15, 5, 10, 10, 12, 0}
, {13, 9, -8, 9, -16, -10, -43, 0, -13, 8, 2, -1, -2, -7, 11, -24, 8, -1, -13, -2, -1, -23, 3, 5, 0, -27, -6, -4, 9, 4, -8, 13}
, {-10, -6, -3, 1, -2, -23, -53, 9, 0, -2, -7, -14, -2, -18, -27, 0, -12, 6, 1, -38, -17, -24, -9, 23, 16, -5, -14, -12, -3, -7, -5, -17}
}
}
, {{{11, 9, 5, 3, -8, -27, -32, -13, 15, -10, 9, 17, 0, -9, 24, -15, 19, -21, 0, -5, 17, -23, 6, 8, -6, -16, -7, 5, 4, 0, 4, 9}
, {-46, 12, -5, 0, -15, -33, -52, 0, -13, 5, 13, 33, -9, 3, 15, 25, 0, -1, 18, -21, 18, -45, -1, -19, 7, -7, -2, 0, 20, 12, 6, 4}
, {21, 8, 9, -1, -8, 12, -8, -7, -2, -2, 4, 11, -43, 2, -19, 15, 8, -3, 19, 9, 19, 4, 7, -27, 2, -11, 16, -14, 16, -19, 0, -14}
}
, {{-7, -3, -3, 3, -3, 7, -10, 13, -14, -8, 4, 3, 6, -3, 6, 9, -9, -1, -5, 6, 7, 2, 1, 17, 7, -35, -18, -11, -1, 2, -13, 13}
, {4, -30, -1, 18, 13, -21, -60, 14, -7, -1, 8, -15, 16, -3, -2, -16, -20, 4, -12, 2, -2, -38, -21, 15, -5, -61, -33, 4, 14, 8, -20, 16}
, {-11, 4, 9, 5, 5, 1, -10, 20, -5, -21, -4, -39, -9, -6, -1, -28, -10, 13, -15, -5, 13, -8, 16, 6, -3, -16, -6, 11, -3, 3, 13, -18}
}
, {{11, 7, -32, 26, 7, -7, -10, 9, 5, 6, -12, 9, 17, -16, 8, -4, -28, -16, -4, 7, -9, -3, -12, 5, 18, -35, -27, -20, -13, -9, -20, 5}
, {11, -31, -19, -9, 12, -16, -1, 14, -26, -7, -29, -10, 16, -7, -7, -50, -36, -15, -46, 14, -15, 0, -23, 16, -24, -15, -36, -18, -20, -13, -15, -12}
, {-15, 2, 6, 22, 11, -11, -28, 7, 16, 3, 5, -19, 36, 13, -1, -10, -19, -1, -13, 12, 3, -25, -17, 12, 3, -11, -12, 18, 0, -7, -17, -2}
}
}
, {{{20, 12, -2, 10, -13, 12, 8, 13, -3, 8, -9, -21, 4, 14, 8, 6, -9, 5, -1, 4, -4, 17, -6, 2, 3, -13, 0, 10, -4, 0, 0, -12}
, {23, 16, 8, 6, 6, 2, 11, 10, -7, 3, 16, -5, 12, 10, -2, 12, 4, 14, 10, 15, 23, 20, 2, 10, 0, -1, 7, 15, 3, 1, 4, -1}
, {-12, -2, 4, 5, 1, 9, -5, 18, 4, -7, 0, -14, -18, 6, -20, -12, -1, -2, -25, -8, 6, -7, 17, 3, -18, 18, 3, -6, -5, -7, 2, -34}
}
, {{3, -10, -2, 17, 14, 12, -3, 19, 10, -3, 0, 8, 0, -2, 12, 17, 0, -2, 9, 1, -15, 1, -10, 2, -8, 5, 9, -4, 6, -9, -3, -14}
, {16, 11, -18, -31, -24, -6, -27, 14, -14, 5, 2, 1, 11, -8, 2, 10, -5, -8, 9, 1, 6, -12, -12, 3, 4, -6, 6, 13, 17, 3, -7, -6}
, {-23, -18, -3, 16, 5, -18, -31, 16, 2, 3, 6, -12, 2, 0, 9, -37, -9, 14, -25, -4, -2, -35, -4, 9, 14, -24, 1, 4, 2, 6, -15, 1}
}
, {{12, 16, -2, 8, 18, 19, 6, -13, 8, 12, 22, 15, 1, -2, 11, 14, 11, 9, 9, 10, 7, 15, 18, 0, 22, 26, 31, 21, 15, -2, 13, 1}
, {40, 18, -32, -31, -60, -7, -14, -42, -2, -38, -46, -2, -14, -60, -12, 13, -30, -17, -10, -1, -11, 5, -23, -35, -5, -36, -39, -68, -20, -12, -48, 7}
, {-52, -14, -32, -2, 8, -10, -28, 19, 5, 4, -19, -3, 25, -8, 16, -57, -25, 8, -55, 18, 5, -35, -32, 33, 6, -35, -59, 8, -8, 21, -50, 22}
}
}
, {{{-3, -2, -6, -9, -21, -20, -4, -17, -12, 0, -7, 9, 2, 4, -3, -8, -8, -6, -7, 0, -10, -12, -9, -17, 0, 8, 20, 0, -2, -20, 6, -6}
, {-43, -11, 17, 2, 9, -3, 7, -15, -4, 1, 2, 3, -26, 6, 4, 0, 20, -1, 34, -1, 14, 9, 12, 10, 5, 14, 10, 4, 26, 8, 8, -28}
, {-3, -14, 9, -1, 29, -8, -1, -3, 16, -36, -8, -56, -20, -1, -20, -34, -5, 13, -20, 32, -3, 19, 17, 23, -27, 21, 25, -14, -8, 7, 1, -48}
}
, {{-27, 9, -2, -11, -30, -20, -8, 17, -9, 20, 8, 7, 14, 0, 9, -32, 9, -5, -16, -14, -8, -23, -4, -5, -14, -11, 0, 14, -8, 4, -6, 3}
, {-23, -12, -5, -11, -10, -7, -19, -18, -38, -1, -3, -13, -30, -3, 9, -17, 5, -1, 13, -6, 0, -1, 1, -11, -20, -25, -8, -4, 2, 2, -4, -18}
, {-33, 19, -5, -17, -4, -18, -15, 12, -12, -11, 4, 15, -11, 5, 8, -2, 11, -10, 6, -11, 6, -36, -7, -2, 17, -4, -8, 4, 4, -5, 0, 25}
}
, {{-13, -44, -3, -35, 7, 21, 7, -1, -27, 3, -35, -25, 24, 1, -6, -21, 2, 10, -12, 7, -2, 8, 2, 18, -12, -36, -7, 8, 15, 9, -14, -1}
, {-20, -38, 15, -7, 0, 17, 19, -10, -18, -35, -11, -46, -2, 13, -29, 7, 1, 17, 19, 3, 8, 26, -2, 1, -19, -5, -1, -12, 3, 8, 2, -9}
, {17, -9, 10, -28, -21, 26, 3, -37, -24, -21, -21, -7, -39, -20, -53, 22, 32, -2, 20, 2, 5, 32, 0, -18, -19, -27, -10, -34, -1, -13, 13, -60}
}
}
, {{{18, 11, -12, 6, 5, 19, -12, 23, 27, -39, -13, -16, -10, -24, -35, 5, -32, -2, -12, -3, 11, 12, 20, -7, 10, -1, -8, -28, -35, -29, -23, -47}
, {0, -12, -13, -23, -20, -2, -7, 10, -6, -18, -17, 7, 3, -16, 2, -63, 1, -7, -20, 1, -7, 5, -6, -2, -33, 1, -8, -4, 0, -13, -12, -6}
, {-23, -10, 4, -10, 21, 1, 2, -10, 9, -2, -10, 15, 1, 1, 10, -31, 11, 3, -43, 3, -22, -14, 1, 5, -28, 21, 16, 1, -6, 20, 15, 17}
}
, {{-6, 14, 3, 4, 18, -5, 2, 16, 23, 8, 2, 2, -35, 5, -5, 11, 5, 2, -1, -14, 1, -4, -14, 7, 11, 6, 16, 14, -1, -19, 13, -20}
, {16, 27, -25, 4, -12, -14, -12, 10, 29, -13, -26, -4, 31, -28, 12, -13, -46, -32, -20, -12, -24, -2, -17, -18, 1, -27, -22, -30, -27, -10, -25, 37}
, {-13, 26, -1, -3, 3, 3, 2, 13, 16, 11, 1, -11, 28, 14, 14, -14, -17, 9, -3, 17, 4, 8, -3, 16, 7, -28, -34, 9, 14, 20, -23, 13}
}
, {{0, 24, 1, 16, 10, -4, 14, 6, 11, 10, 5, -9, -4, 14, -3, 11, 35, 2, 1, 1, -9, -8, 5, -2, 2, 27, 11, 9, 1, 5, 2, -21}
, {-24, 15, 5, 6, 5, -27, 4, -6, 3, -6, 2, 1, -6, -2, -19, -19, -17, -22, -48, -16, -35, -35, -10, -19, -9, 15, 14, -14, -24, -26, 4, -5}
, {8, 11, -1, 4, -3, -14, -30, -18, -8, 1, 18, 12, 5, -7, 15, 12, 8, -27, 15, 0, 6, -4, -50, -28, 6, -2, -5, 2, 10, 3, 8, 27}
}
}
, {{{-5, -11, 15, 3, 15, 18, 18, -10, 8, 7, -4, -19, 17, 7, 0, 1, -3, 14, 10, 19, 10, 23, 9, -5, 7, 7, 6, 6, -15, 9, -3, 4}
, {-8, -19, 4, -13, 11, 11, 3, 2, -29, 11, 18, 2, 0, 12, 6, -14, 20, 13, -1, 7, 15, -1, 19, 8, -13, 0, 21, 1, 7, 21, -2, -9}
, {12, -10, 0, -2, 9, 12, -9, 6, 3, 4, 3, -19, 10, 16, -12, 10, -4, 19, -6, 19, 0, 19, 17, 16, -2, 2, 19, 5, -8, -16, 5, -32}
}
, {{-8, -25, -13, -59, -34, 17, 8, -7, -49, -21, -38, -4, 23, -16, 6, -8, -2, 18, -15, 10, 2, 17, 26, 11, -24, 0, -23, -4, 3, -3, -24, -11}
, {-9, 19, -27, -16, -6, 2, -8, 17, -24, 5, 2, 2, 14, -11, 15, 7, 11, -9, 1, 3, 5, 0, -24, -2, -3, -12, -32, 3, 3, 10, 3, -2}
, {0, 18, -23, 1, -13, 15, -10, -1, 4, -1, -11, 0, 7, -25, -4, 7, -37, 3, 5, -2, 3, 2, -4, -2, 16, -34, -14, -30, -7, -7, -70, 15}
}
, {{3, -11, -36, -7, -42, -38, 1, -26, -40, 11, 3, 6, 17, -7, 15, 15, 13, -59, 11, -3, -8, -14, -27, -63, 7, 5, -7, 15, -4, -4, 13, 9}
, {12, -9, -52, -30, -52, 45, -4, -23, -30, 8, -28, 28, 1, -28, -41, -1, -6, -10, 17, 5, 1, 40, -34, -30, -8, -25, -23, -25, 4, -24, -34, -19}
, {13, -12, -67, -57, -31, 3, -18, -24, -40, -37, -41, 17, -20, -40, -13, 1, -13, -50, -45, -26, -17, 25, -47, -43, -51, -37, -53, -32, -41, -17, -40, -9}
}
}
, {{{7, 36, 10, -10, 0, 5, -6, -1, -3, -7, 13, 25, -4, 20, 9, -18, -3, 7, 3, -12, 7, -6, 10, 6, 13, 3, 6, 7, 10, 12, 3, 17}
, {15, 13, -15, 5, 8, -17, -19, 11, 6, 11, 5, -5, -4, -13, 21, 12, 5, -3, 10, 20, 4, -18, -12, -15, 29, -16, 7, -7, 6, -4, -5, -5}
, {0, -6, -40, -12, -18, -3, 10, 7, -30, 11, -6, 8, 20, 3, 7, -11, -18, 3, 1, 19, 14, -4, 15, -2, -1, 3, -11, 13, -15, 22, -43, 16}
}
, {{1, 36, -7, -4, -6, -4, -8, 6, -7, 4, 8, 10, 5, 10, 1, 15, 12, -14, 18, -10, -6, 5, -25, -21, 11, -10, -13, 5, 8, 0, 11, 1}
, {-10, 11, -9, 7, -10, 3, 15, -20, 1, -5, 4, 29, -40, -3, -22, 10, 3, -36, 22, -31, -9, 6, -8, -33, 9, 5, 0, -9, -17, -29, -2, 16}
, {-18, -18, -15, -8, -24, -18, -18, -18, -23, 15, -9, 25, -16, 4, 13, 13, 4, -43, 25, -38, 23, -27, -37, -38, 4, 30, 5, -13, 3, 4, 4, -3}
}
, {{-10, 12, -9, -1, -25, 7, -33, 0, -1, 7, 13, 17, -24, 2, -11, 6, 10, 17, 5, -16, 9, -17, -13, -4, -1, -21, 7, 2, 13, 12, -2, -2}
, {-26, -6, -6, -7, -22, -23, -5, -26, -25, -4, 9, 16, -24, -2, -9, -1, 7, -5, 5, -17, 2, -17, -10, -32, -18, 0, -13, -5, -8, 0, 17, 9}
, {6, -2, 4, 4, -38, -4, -33, -7, -29, 15, 8, 16, -26, -3, 8, 13, 18, 3, 10, -22, 4, -9, -17, -7, 2, -34, -16, 2, 7, 6, 3, -1}
}
}
, {{{-21, -38, -6, -21, -19, 1, 16, -7, -35, 6, -8, 7, -1, -8, 22, -10, 13, -2, 4, -5, 2, 11, 13, -9, -22, 0, -7, 1, 0, 18, 9, 3}
, {-4, -7, -17, -6, -7, -7, -12, 5, -8, -1, -15, 3, -17, 2, -19, -9, -7, -1, -6, 0, -12, -4, -26, -2, -14, -2, -11, -2, -12, 8, 4, -14}
, {-18, 4, 7, 12, 19, -25, -1, -11, 6, -10, -8, 3, -31, -1, -18, 3, -21, -23, -10, -24, -11, -13, -30, 11, 1, 18, 13, -13, -26, -22, 8, -3}
}
, {{-44, -14, 1, -6, 17, -19, -52, 2, 1, 24, 0, -14, -15, 1, 3, -19, 12, 6, -22, -41, -19, -17, 18, 6, 19, -20, -18, 16, -1, 4, -3, -1}
, {-17, -14, -7, -4, 6, -20, -44, -1, -1, 4, 10, -4, -41, 7, 0, -25, -9, 6, -18, -55, -5, -53, -13, -6, -5, -8, -3, 12, -12, -1, 8, 7}
, {-43, -12, 18, 1, 14, 25, -16, -35, 3, 1, 3, -22, 11, 6, -2, -23, 13, 6, -19, 17, 18, -1, 24, 16, -46, 7, 14, -4, 17, 27, 2, -15}
}
, {{3, -9, -11, -21, -33, -7, 13, 12, -12, 20, 17, -3, -3, 11, 5, 9, 3, -6, -5, 6, -5, -3, -20, 14, -3, -5, -22, 11, 0, 8, -1, -5}
, {0, 0, -12, 2, 2, -2, -27, 9, -9, 12, 4, -12, 4, -4, 6, 2, -26, 15, -7, 2, 11, 0, -13, 21, 11, -34, -28, -1, -11, -1, -48, 19}
, {20, -15, 5, -5, 3, 22, 13, 1, -5, -30, -4, -27, 9, 2, -9, -19, -34, 11, -11, 17, 0, 22, 24, 21, -9, 3, 3, 0, 8, 11, -8, -4}
}
}
, {{{20, 29, -9, 9, -1, 2, 4, 16, 5, 4, 3, 30, 14, 7, 23, 11, -2, 6, -1, -5, 24, 11, 15, 11, 16, -1, 11, 1, -6, 16, -3, 25}
, {27, 5, -13, 3, -7, 13, -10, 23, -8, -7, -12, -13, 1, -5, 1, 5, -19, 8, -14, 1, -2, -5, -6, 12, -1, -26, -27, -8, -2, 2, -37, 7}
, {-2, 5, 9, 1, 4, -33, -6, 12, 1, -7, 6, -14, 24, 2, 4, 3, -21, -12, -6, -1, -13, -19, -9, 1, 4, -2, -6, -16, -5, -23, -5, 0}
}
, {{18, 13, -1, -6, -3, -14, 17, 17, 22, 4, 3, -41, -25, -5, -31, 2, -14, -2, -1, -15, -22, -6, -14, -10, -3, -1, -14, 15, -22, -28, -14, -18}
, {3, -7, -12, -4, -24, -19, 8, -2, -41, -6, -9, 13, 4, 6, 14, -9, -11, -34, -22, -8, -28, -8, -73, -41, -13, 9, 0, 2, -21, -2, 12, 9}
, {-43, -25, 17, 6, 18, -24, -29, 4, -8, -2, -1, -8, -16, 5, -2, -2, 10, 6, 2, -23, 15, -34, -13, 2, -1, -2, 3, 12, 9, 9, 15, -10}
}
, {{-5, 7, -16, -11, -17, -31, -39, -4, -10, 3, 2, 27, -14, 16, -4, 5, -9, -24, -22, -50, -34, -18, -26, -14, 18, -8, -12, 0, -20, -3, 8, 18}
, {-8, -27, -6, 30, -49, -33, -89, -2, -49, 9, 32, 8, -1, 6, 2, -1, 18, 1, 18, -36, 1, -49, -23, -12, 25, -32, -30, 14, 11, 8, 2, 2}
, {-17, -7, 14, 8, -15, 16, -17, -1, -2, 3, 0, -4, -1, 11, 2, -21, -1, 7, -3, -6, 10, -12, 8, 14, 6, -20, -25, 4, 13, 18, 4, 7}
}
}
, {{{13, 19, -2, 7, 16, 0, 0, 18, -7, 17, 2, 1, -15, -7, 7, -2, -21, 16, 2, 3, -5, -4, -1, 16, 9, -9, -10, 9, -17, -11, -11, 30}
, {-34, -30, 18, -5, 13, -40, 2, -56, 2, -20, 2, 12, -43, -15, -6, -53, -2, -47, -56, -34, -8, -50, -18, -29, -42, 17, 17, -1, -10, -17, 7, -17}
, {-47, 11, 3, -5, -12, 7, -5, -4, -16, 16, 11, 10, -13, 8, 19, -2, 23, -3, 14, 3, 29, 14, 16, -11, -7, 24, 18, 12, 29, 22, 19, 2}
}
, {{-26, 5, -9, 4, -8, 4, -17, 25, 0, 2, 0, 18, 3, 0, 12, -36, -12, -12, -24, -22, 1, -22, -2, 2, -14, -9, -39, -6, -3, 1, -22, 10}
, {-9, -20, 9, 16, -39, 25, -14, -13, -21, -7, -7, -20, -13, -17, -33, 14, -4, 12, -4, 0, 3, -3, 2, 8, -1, 23, -11, -24, -4, 2, 2, -3}
, {19, -14, 2, -9, -5, 13, 24, -29, -9, -34, -26, -19, -3, -41, -32, 0, -42, 6, -2, 12, -4, 11, 7, -6, -34, 20, -2, -39, 3, -10, -5, -37}
}
, {{-3, -18, 17, 1, -4, -8, 7, 1, 1, -5, -9, -20, -1, 7, 0, -13, -6, 3, -3, 14, 6, 0, 11, -3, -7, 6, 0, 3, 10, 7, 19, -2}
, {0, 14, 5, 6, 7, -13, 15, 8, 24, -7, 9, -26, -11, 15, 3, 3, 8, 3, -10, 3, 6, 0, 5, -4, 13, -16, -1, 13, 8, 2, -10, -17}
, {8, -15, 7, 6, 21, 0, 17, 19, 8, -27, 22, -15, 0, 4, -3, -14, -5, 5, -13, 7, 4, 19, 4, 3, -21, -14, -13, -2, 1, 0, -18, -11}
}
}
, {{{5, -12, -9, -30, -27, -13, -16, 15, -17, 16, -13, 2, 9, 5, -5, -11, 8, -4, -2, -17, -1, -7, -17, 3, -17, -8, -16, 0, 11, 3, -4, 29}
, {-42, 20, 1, 3, 0, -2, 4, -3, -4, 8, 26, 30, -12, 13, 13, 18, 13, 0, 11, -23, 17, -5, 2, -8, 18, -4, 0, 13, 21, 14, 10, 11}
, {23, 27, 8, 4, 17, -5, -15, 1, 10, 29, 27, -3, -4, 2, -2, 28, 22, -11, 10, -5, 26, 1, 14, 1, 32, -38, 5, 11, 7, 7, -27, 3}
}
, {{31, -27, -1, -5, -12, 0, -11, 4, 0, -7, -4, -5, 7, -4, 3, 23, 7, 7, 14, 3, 4, 22, 16, 1, -14, -9, 10, -9, 5, 4, 9, 18}
, {21, 9, -15, -10, -17, 3, -1, 8, 2, 2, -3, 2, 3, -21, -3, 22, -8, -4, 2, -7, 6, 2, -34, -9, 10, -17, -64, -10, -3, 10, -60, -5}
, {-7, -5, -44, -19, -39, -11, -8, -30, -11, -32, -29, 17, 5, -22, -7, -18, -41, -44, -26, -8, -28, 6, -29, -16, -14, -13, -48, -36, -36, -28, -23, 10}
}
, {{45, -2, -27, -10, -8, -7, 7, -5, -18, 1, -4, 21, 8, -26, -4, -5, -7, -15, 3, 10, -7, 18, -45, -14, -3, -11, -12, -4, -12, -4, -18, 0}
, {-18, -6, 0, 4, -6, -29, -8, 6, -7, 0, 0, 18, -7, 12, -7, -8, -16, -10, -26, -22, -19, -29, -21, -15, -2, 16, -1, 5, -20, -14, 3, -6}
, {7, -2, -9, 0, 9, -13, 32, -12, 15, -13, -16, -11, -19, 8, -11, -16, 0, -24, -38, -30, -17, -23, -1, -10, -23, 9, 17, -4, -14, -4, 4, -2}
}
}
, {{{19, -11, -12, -6, 4, 7, 5, 8, -1, 18, 9, -11, -7, -2, -12, 5, 9, 7, 19, -4, 13, -3, -25, -1, 0, -10, -25, 6, -4, -4, -9, -5}
, {1, -4, 0, 7, 10, 24, 13, -7, 11, -9, 0, 5, -4, -14, -30, 8, -36, -7, 1, 16, -24, 27, -34, -16, -8, 4, 12, -4, -37, -34, 6, -24}
, {14, -18, -15, -10, -15, 3, 16, -9, -15, -19, -38, -5, -24, -15, -39, -18, -16, -20, -2, 11, -27, 4, -44, -25, -22, 19, 15, -25, -47, -60, 5, -51}
}
, {{-7, 10, -9, 0, 16, -25, -14, 28, 28, -12, -16, 4, 6, -2, 3, -20, -20, -5, -27, -1, -39, -25, -34, -5, -4, -11, -28, -10, -21, -1, -21, 12}
, {-22, 11, -1, 8, 0, -36, -33, 17, 28, 16, 5, 24, -2, 11, 19, -13, -15, 0, -24, -37, -36, -49, -26, -4, 25, 4, -12, 3, -30, -2, 3, 30}
, {-12, 8, 2, 15, 19, -26, -3, 19, 16, 16, 6, 6, -26, 7, -7, 0, -3, -25, -4, -31, -19, -48, -26, -19, 13, -8, 12, 6, -17, -7, -13, -1}
}
, {{-1, -7, 18, -17, -6, 13, -44, -5, 4, 12, 5, -12, -8, 6, 8, -26, 0, 1, -5, -18, 9, -15, 20, 3, -1, 19, 14, 7, 16, 21, 16, -2}
, {12, 9, 5, 4, 8, 5, -51, 1, 22, 20, 7, 14, 6, 9, -3, -11, 7, 6, -10, -17, -3, -27, -11, 13, 15, -13, -31, 13, 9, 17, -9, 20}
, {-18, 2, 10, 15, 21, 1, -28, 15, 2, 10, 0, -52, 1, 12, -27, -11, -8, 5, -6, -26, -10, -23, 2, 14, -6, -7, 1, -3, -7, -5, 2, -10}
}
}
, {{{-5, 0, -2, -6, 1, 2, 14, -24, -18, -1, -18, 17, 2, 0, 23, 1, -1, 0, -19, 6, -10, -9, 14, 6, -31, 14, 13, 2, 9, 3, 0, 5}
, {0, 7, 10, 4, 2, -37, -21, -16, 10, -9, 1, 7, 4, 8, 7, 3, -10, 1, -3, -28, 0, -41, -21, -15, 0, 9, 5, -9, -14, -11, 3, 0}
, {16, -8, 0, 9, -20, 15, -8, -15, -3, 11, -24, -1, -13, -14, 7, -4, 6, -1, 11, 7, 19, 14, 14, -19, -13, 2, 15, -13, 4, 16, 10, 2}
}
, {{-11, -3, -18, -13, -36, -4, 3, -7, -62, 1, 6, 4, -7, -21, 6, 4, 5, -1, 0, -10, 8, -7, 1, -17, -5, 8, -16, -12, 3, 4, -7, 0}
, {-27, -33, -5, -1, 27, -1, -32, 18, -26, 1, 7, -7, 21, 0, 11, -16, 5, 4, -7, -4, 7, -7, -5, 8, 3, -45, -18, 15, 13, 16, -10, 10}
, {-21, 5, 5, 9, 13, -36, -62, 15, 10, -4, 5, -27, 28, 11, -3, -33, -5, 5, -43, 3, -36, -61, -9, 14, -7, -48, 7, 9, -3, 9, 0, -4}
}
, {{-14, 24, 4, 5, -18, 14, -12, 0, -2, 9, -10, 11, -4, 19, 7, 8, 11, 8, -2, -11, -8, 1, -3, 17, -1, 13, 0, 9, 10, 3, 12, 2}
, {-20, -14, -7, -12, -12, -7, 10, -23, -24, -5, -18, 9, 3, -10, 2, 12, 20, -8, 10, 10, 10, 15, -16, -16, -11, -12, -6, -16, -4, -8, 15, 5}
, {-17, 14, -23, -3, -20, -9, -14, -2, -4, 12, 12, 16, 14, 10, 26, 3, 11, 3, 2, 9, 7, -14, -23, -15, 9, -10, -41, 14, -1, 8, -5, 24}
}
}
, {{{5, 13, -25, 0, -23, -27, -9, 11, 13, 13, 9, 28, 15, -25, 3, -12, 1, 5, -16, -8, 10, -28, -44, -5, -5, 15, -2, 3, -2, 3, 3, 5}
, {-4, 5, -13, -8, -6, -10, -14, 35, 1, -5, 4, -1, 24, -1, 2, -20, -16, 8, -15, 11, 13, -16, -32, 21, 12, -22, -16, -6, -1, 1, -25, 3}
, {-18, 5, -3, 1, -1, -24, -10, 10, 2, 14, 6, 10, 10, -2, 2, -8, -11, 1, -7, -12, -4, -35, -10, -2, 16, -7, -16, 3, -27, 9, -10, 20}
}
, {{-7, 21, 1, 15, -13, -21, 9, 19, 35, 13, 6, -7, 28, -1, 7, -2, -6, -13, -6, -8, -14, -29, -51, -13, 13, -4, -8, 1, -14, -13, -1, 20}
, {-9, 27, -5, 16, -19, -22, -6, 0, 4, 6, 3, 2, 5, 4, -1, -7, -15, -51, -1, -30, -34, -18, -11, -28, 1, -6, -13, -8, -34, -17, 0, 18}
, {-8, 14, -6, -9, -6, -25, -31, 6, 6, 22, 5, 33, 11, 9, 23, 6, 12, -18, 10, -33, 12, -36, -36, 6, 10, 11, -6, 10, 5, 2, 6, 6}
}
, {{4, 13, 5, 18, -7, -39, -38, 0, 15, 4, 13, -10, -18, 2, 1, 8, 18, -8, 5, -67, -16, -61, -39, -20, 5, -31, -26, 10, -3, -1, 8, -7}
, {-82, -24, 1, 15, -18, -29, -45, 5, -22, -5, 8, 6, 0, 6, -9, 0, -1, -9, 7, -33, -15, -57, -12, -10, 17, -10, -13, 0, -10, 4, 17, 13}
, {-2, 4, 9, 8, -13, -21, -52, -2, 8, 12, 7, -9, -38, 4, -39, -2, -2, -2, 3, -48, -15, -30, -63, -22, 14, -56, -31, -3, -5, -26, 0, -18}
}
}
, {{{0, 5, 11, -1, 30, 0, -4, -7, 8, -12, 8, -5, 20, 14, -3, -6, -11, 21, -9, -8, -2, -22, 18, 7, -25, 1, 13, 6, -4, 5, 4, -18}
, {-3, -2, -5, 3, -14, 1, 20, -11, 3, 0, 13, -31, -8, -12, 1, 13, 0, 8, 13, 15, 3, 14, -19, -26, -7, 6, -4, -6, 2, -2, 0, -11}
, {22, -18, -15, -28, -36, 16, 34, -24, -28, -17, 0, 1, -27, -6, -16, 20, 9, 16, 16, 23, 16, 20, 28, -5, 11, -8, -31, -15, 23, 4, -1, -40}
}
, {{-18, -40, -1, -14, 8, -6, -10, -16, -29, -3, 3, -10, -7, -6, 9, 6, 18, 10, 6, 15, 11, -2, 0, -8, -13, 14, 8, 13, -3, 2, 13, -16}
, {-23, 4, 2, 8, 0, -3, -41, 15, 15, -21, 6, -43, -22, 6, -5, -26, -7, 5, -26, -11, -8, -37, 1, 6, -5, -11, -2, 5, -4, -3, -15, -28}
, {11, -3, 6, 2, 15, -38, 10, -4, 8, -65, -12, -51, -22, -14, -46, -23, -56, 10, -61, -18, -67, -48, -11, 6, -34, -4, -8, -37, -16, -40, -5, -21}
}
, {{2, 5, 6, 6, -21, 10, -4, -1, -3, 1, -19, 13, -19, 4, -15, -18, 15, 14, -7, -7, 9, 13, -3, 8, 6, 4, -13, 9, -2, -2, 11, -17}
, {4, 10, 5, -14, 6, -27, 28, -2, -1, 5, 23, -33, -26, 7, -16, 12, 31, 2, 5, -7, 2, -5, -1, -19, 0, -1, -10, 7, 9, -24, 0, -30}
, {-8, 6, -10, -3, -2, -17, -15, 14, 5, 2, 11, 25, 4, 10, 23, 16, -14, -28, -12, -36, -20, -17, -27, -7, 5, 6, -2, 13, -18, -6, 2, 16}
}
}
, {{{-20, 0, -1, -14, -19, 6, -20, -5, -19, 8, 2, 1, -3, 1, -7, -45, 8, -3, 1, -14, 2, -2, 0, -9, -6, -13, -11, 2, -1, 3, 10, 7}
, {-19, 21, 8, 9, 9, -27, -16, 11, 10, 7, 10, 12, -29, 9, 1, 0, 5, 1, 18, -26, -7, -41, -16, -7, 3, 14, 2, 5, -3, 5, 7, -3}
, {-10, 2, 0, 0, -22, -14, -1, -25, -2, 7, 3, -8, -24, -5, 9, -13, 11, -21, -6, -22, 12, -18, -13, -29, -9, 3, -7, -10, -2, -17, 11, 1}
}
, {{15, -33, -31, -61, -33, -4, -13, 18, -30, 0, -3, 0, 19, -6, 4, 14, -1, 1, -14, 2, -8, 2, -6, -5, -15, -35, -35, 2, -10, 10, -22, 1}
, {-39, 12, -17, 6, -5, -57, -64, 6, 10, 2, 12, 5, 12, 1, -4, -4, -15, 6, -13, -10, -1, -68, -34, 4, 17, -40, -48, 7, -1, 12, -24, 13}
, {0, 18, 4, -3, -9, -2, -18, -2, -1, -2, 9, 1, -30, 5, -6, 10, 14, -19, 15, -18, 2, -10, 5, -16, 9, -40, -17, 2, 1, -2, 15, 12}
}
, {{0, -46, 4, -43, 4, -14, 10, -34, -11, -21, -65, -2, 6, -1, 14, -15, 0, -15, -35, 10, -11, 2, 6, -20, -75, 25, 38, -7, -15, 9, 1, 11}
, {22, -1, 1, 5, 17, -1, -33, 8, 17, 19, 4, -19, 12, 20, -2, -26, -10, 13, -38, -1, -7, -4, -11, 10, 15, -63, -60, 15, -1, 3, -8, 20}
, {-26, -21, 5, 6, 8, -1, -34, 16, -2, 0, -3, -31, -21, 0, -7, -36, 2, 21, -20, -20, 12, 3, 14, 2, -5, 10, 13, -11, 12, 0, 21, -21}
}
}
, {{{-2, 4, -1, -12, -7, -5, -4, -24, 3, -5, -26, 10, 8, -8, 23, -2, -1, -3, -27, -3, 2, -11, 10, -8, -35, 11, 4, 0, -20, 16, 7, 4}
, {13, -36, 8, 9, -8, 0, 6, 6, -15, 5, -17, -19, -8, -4, 2, -7, 1, 15, 20, -6, 4, 13, 30, 4, -6, 15, 17, 0, -1, 1, 4, -18}
, {9, 4, 3, 15, 19, -23, -8, 25, -1, 10, 0, -24, 19, 0, -3, 10, -9, 13, -13, 3, -24, -7, 0, 18, -7, 5, -7, -2, -6, -25, -11, 10}
}
, {{3, 8, 4, 2, -2, 6, -5, -1, -24, 3, 0, 10, -1, 5, 7, -21, 11, 8, -12, -11, 1, 10, 2, 25, -28, 19, 4, 5, 3, 9, 11, -1}
, {-28, -57, 2, -32, -16, 4, 32, -12, -64, -10, -12, -18, -9, 4, 4, -29, 16, 12, -13, 17, 5, -3, 8, -8, -24, 14, 6, -2, 3, 10, -2, -20}
, {-43, -3, -4, -18, 6, -30, -39, -15, -10, 12, 18, -4, -6, 11, 18, 2, 18, -5, 17, -9, 14, -39, 3, -22, -2, -26, 2, 15, 5, 4, 1, -8}
}
, {{9, -17, -6, -26, -8, -15, 16, -5, -27, -6, -37, 7, 9, 7, 23, 5, -12, -1, -10, 22, -13, 7, 12, -12, -30, 1, 8, 14, -11, -1, -3, 9}
, {-50, -27, 2, 7, -27, -9, -34, 5, -27, -2, 15, -7, 6, -10, -5, -12, 2, 3, -1, -20, 9, -14, 9, 6, 13, -37, -9, 6, 9, -1, 1, 3}
, {-10, -1, -4, 7, -11, 4, -24, -9, -5, -11, -14, 6, -41, -21, -23, 4, -4, 3, 3, -19, 5, 0, -13, 5, -1, -43, -71, -21, 15, 2, 5, -21}
}
}
, {{{22, -7, 6, -1, 10, -11, 3, -1, -3, -7, -1, -7, -3, 20, 1, -24, -4, 11, -19, 11, -8, -9, 25, 16, -7, -4, 9, -12, 15, -2, -5, 10}
, {0, 2, 2, -1, -3, -1, 23, -23, -2, -5, -8, 9, 6, 4, 1, 10, -7, -6, 23, 7, -9, -9, 14, 2, -6, -10, -17, -3, 10, -2, 1, 5}
, {27, -5, 4, -8, -2, -12, -20, -6, -5, -5, -2, -7, 7, -3, -7, 5, -9, -3, 0, -4, 3, -19, 4, -7, 10, 4, -13, 8, 0, -3, 8, -21}
}
, {{-21, -14, 8, 4, -2, -22, 15, 10, 5, -9, 1, 7, 19, -4, 16, -30, -18, 1, -18, 4, -5, -20, 3, 14, -1, -1, 1, 11, -15, -13, -9, -3}
, {-19, -61, 15, -6, -5, -10, 1, -11, -39, -2, 15, -2, -25, 20, -9, -18, 23, -3, 17, 6, 13, -10, -11, 2, -21, -5, 12, 5, 19, 8, 17, -40}
, {-20, 3, 7, 5, 12, -8, -5, 0, 5, -2, -6, 9, 1, 9, 15, 16, 4, 0, 13, 7, 10, -18, 17, 5, 2, -13, 20, 8, 13, 20, 15, 0}
}
, {{-26, -8, -42, 3, -17, -3, -33, 4, -25, 14, -14, -2, 16, -3, 16, -32, -32, -8, -21, 2, -7, -39, -26, 9, 18, -55, -76, -7, 3, 8, -34, 8}
, {-36, -20, 13, 19, -7, -38, -74, 1, -11, -24, -2, -18, -7, -6, -20, -39, -12, -2, -30, -34, -3, -59, -2, 1, -7, -50, -43, -1, -23, -2, -2, 16}
, {-14, 2, -4, 17, -2, 6, -20, 21, 5, 10, -5, -5, 1, 2, -4, 6, 1, 9, 9, -9, 11, -24, -28, 3, -1, -9, -24, -1, -2, -10, -9, 7}
}
}
, {{{28, 35, 1, 7, 8, 22, 13, 13, -3, 19, 19, 2, 22, 11, 25, 19, 9, 9, 18, 16, 22, 22, 3, 6, 23, -15, 4, 10, 6, 14, -12, 10}
, {13, 14, 10, -13, 5, 12, 3, -2, 12, -1, 9, 0, 6, -7, 12, 6, 5, 9, -11, 19, -6, 16, 6, 8, -2, -9, 7, 0, 2, 5, -13, 11}
, {34, -1, 1, -2, 2, -18, -12, 3, 7, -11, -5, -14, -5, -19, -26, 16, -19, -9, 17, 0, -2, -4, 13, 6, 4, 7, -8, -8, -5, -16, -13, -7}
}
, {{7, 14, 5, 16, 12, 15, 15, 13, 27, -5, -6, -10, -17, -10, -34, 9, -16, 15, 5, 0, 4, 3, 10, 14, 5, 13, 4, 9, 0, -12, -10, -32}
, {16, 7, -19, -4, -12, -8, 19, -33, -5, -12, -43, 1, 7, -31, -3, 11, -22, -12, 8, -4, -22, 9, -10, -20, -42, 5, -7, -19, -24, -5, -17, 8}
, {-52, -18, 0, 2, 8, -52, -4, -4, -19, -9, -11, -12, 6, -6, 3, -51, -3, -13, -48, -13, -18, -35, -39, -2, -8, 4, 8, 14, -12, -1, -1, -7}
}
, {{24, 1, -24, -21, -23, -7, 31, -14, -15, -10, 2, 10, -14, -5, -20, 0, -14, -16, -17, 3, -8, 30, -13, -16, -3, -7, -29, -17, -13, -19, -27, -23}
, {-31, 1, -12, 3, 1, -25, -41, 7, -29, 18, 20, 25, -7, -12, 31, 1, 17, -12, -25, -38, 9, -34, -4, -13, -15, 3, -5, 13, 1, 20, 11, -3}
, {-11, 16, 0, -7, -15, -21, -16, 4, -18, 12, 7, 1, -5, 8, 9, 4, -2, 6, -8, -3, -9, -28, 0, -10, 1, -18, 5, -2, 2, 4, -8, -16}
}
}
, {{{-6, 14, 2, 7, 10, -6, -1, 3, -16, -11, -4, 11, -20, 5, 1, 5, 10, -1, 6, -1, 10, -5, -17, -2, -4, -9, -17, 1, 12, -8, 1, -4}
, {-8, 22, 5, -14, 12, 15, -23, 2, -11, 8, 7, 9, -6, 14, 12, -6, 15, -8, 12, -8, 0, -21, -13, -6, 4, 18, 12, 20, -4, -3, 14, 12}
, {11, 4, 11, 6, 10, 2, 4, -24, 8, -3, -18, -22, -10, 11, -9, -4, -11, 2, -6, -1, -5, 11, 4, -9, 3, 9, 18, -30, -14, -15, 8, -14}
}
, {{4, -5, -22, -34, -29, 23, -18, 6, -45, 22, 5, 32, 18, -13, 20, -12, 8, -2, -7, -5, 4, 6, 4, 12, -11, -2, -25, -7, -1, 5, -14, 15}
, {2, -31, -5, 3, 23, 9, 7, -3, -14, 6, 1, -2, -10, 25, -5, -21, 8, 8, 13, 1, 2, 16, 10, 0, 5, -3, -3, 9, 2, -7, 9, -4}
, {-18, -37, 8, 0, -1, 9, 2, -10, 18, -6, -3, -56, 6, -3, 9, -39, 4, 9, 1, 26, 10, 11, 5, -6, -16, 13, 17, 7, 6, 9, 20, -57}
}
, {{11, -21, -9, -30, -21, 33, 28, -20, -13, -9, -26, -21, 5, -29, -36, -2, -1, -10, -1, 8, -1, 25, -5, -23, -43, 18, -11, -4, -1, -14, -11, -18}
, {0, -29, -28, -53, -55, -7, 15, -62, -45, -41, -30, 18, -6, -51, -9, 22, 12, -7, 7, 5, -12, -2, -33, -19, -71, 14, -1, -37, -16, -20, 0, -14}
, {-22, -18, -4, -23, -28, -10, -10, -21, -52, 0, 4, 15, -11, -12, 9, -5, 1, -13, 10, -15, 9, -15, -10, -15, -13, 13, -16, 2, 17, 3, 8, 2}
}
}
, {{{6, 6, -34, -7, -28, 2, 13, 1, -26, -9, -2, 26, -5, -7, 12, 31, -1, -37, 5, -16, -12, 13, 22, -33, 8, -15, -33, -4, -22, 0, -8, 32}
, {23, -13, -17, 0, -38, -12, 2, 21, -15, 12, -8, 7, -2, -2, -6, 14, -17, -14, -4, -10, -7, 7, -9, -15, 15, -11, -27, -9, -5, 2, 0, 22}
, {25, -3, -19, -12, -42, -19, -11, 6, -12, 4, 2, 18, -4, -1, 10, 0, -4, -35, 1, -16, -5, -22, -44, 4, 12, -25, -38, 5, 2, -13, -25, 4}
}
, {{8, -24, 0, 22, 3, 0, -17, 3, 9, -10, -6, -15, -14, -4, -15, -11, -1, 17, -13, -1, -12, 12, 9, 7, -18, 9, 6, -12, -11, -3, 4, -7}
, {-4, -24, 3, -13, -13, 0, -1, 15, -36, -10, 8, -9, -8, -8, -5, -14, -1, 6, -12, -5, 5, -8, 13, -17, -2, -5, -19, -13, 1, 6, -10, -9}
, {-20, 8, 6, 8, 15, -18, -54, 0, 1, 1, 18, 7, -27, 4, -6, 15, 7, -11, 12, -49, 15, -37, -4, -9, 18, 12, 14, 14, 4, -9, 14, 7}
}
, {{22, -25, -8, -25, -25, 18, 14, -7, -29, -4, 9, 14, -21, 11, -11, 17, 3, -1, 15, -5, 7, 24, 19, 13, 7, 17, 9, -7, 11, -4, -3, -18}
, {12, -5, 3, 8, -11, -12, -11, -6, -48, 8, -2, 5, 4, -10, 18, -19, -5, -3, -20, 1, 7, -14, 17, 10, 3, -56, -14, -7, 0, 9, -10, 0}
, {-24, 4, 5, 19, -3, -21, -65, 10, 4, 10, -11, -6, 0, 11, -14, -14, -27, 13, -24, -52, -33, -73, -34, 23, 6, -33, -26, -1, 0, 2, -16, -4}
}
}
, {{{7, -2, 8, 7, 12, 13, 13, -21, -5, -11, -11, 0, -3, 6, -5, 5, -1, 10, 11, 22, 1, 20, 11, 1, 2, 1, -1, -5, -7, 17, 16, -3}
, {-1, -17, -13, -10, -13, -1, 16, 4, -27, 11, 9, 3, -12, 13, 9, 9, 22, 18, 14, 0, 13, -13, 3, -11, 11, -7, 7, 2, 3, 10, 6, -4}
, {26, -4, -1, 17, 9, -6, 0, 18, 0, 8, 12, 7, -10, 7, 3, 9, -3, -5, -4, -6, 4, 11, -10, 9, 13, -2, -3, 15, -6, -12, -2, -5}
}
, {{0, -7, -18, -21, -12, 16, -7, -1, -15, -1, -10, 5, 9, -18, 12, 1, 8, 10, -11, 10, 9, 9, 13, -1, -24, 7, -1, 6, 10, 8, -2, -4}
, {-17, 17, -30, -15, -28, 6, 0, 0, -5, 10, 3, 24, 8, -9, 4, 9, 4, 2, 15, 1, 8, -4, -10, -13, -4, -16, -23, 3, 2, 11, 1, 8}
, {1, 2, -3, 6, -6, 21, 0, 6, -12, 6, 5, 4, -9, -5, 10, 14, 7, -7, 14, -4, 19, 3, 9, -3, 3, 21, -5, 9, -3, 5, 10, 0}
}
, {{12, 20, -57, -12, -40, -33, -16, -11, -19, 7, -8, 9, 31, -32, 23, 9, -8, -67, 1, -4, -8, -20, -65, -49, 14, -18, -21, 2, -22, -6, -14, 11}
, {22, -11, -64, -52, -61, 46, -25, 3, -17, -4, -19, 8, 15, -45, -15, 17, -11, -8, 10, 1, -2, 31, -3, 0, 3, -40, -33, -26, -9, -12, -57, 4}
, {3, -4, -55, -51, -29, -12, 2, -19, -43, -43, -51, 23, -6, -51, -13, 14, -51, -2, 2, 1, -17, -8, -32, -45, -34, -1, -11, -58, -32, -32, -35, -12}
}
}
, {{{-2, -32, -24, -49, -12, 25, 4, -2, -64, -1, -21, -27, 24, -8, 5, 22, -1, -8, 5, 10, 1, 18, 11, 10, 1, -5, -42, 18, 4, 9, -5, 7}
, {-12, -13, -40, -16, -34, 28, -21, 15, -28, 9, -7, 15, 12, -17, -6, 27, 9, 15, 14, -2, -4, 13, -6, 2, 9, -29, -56, -6, -6, 3, -26, 21}
, {-8, 19, -29, -5, -25, 10, -6, 5, 6, 3, -2, -17, 0, -17, -27, 19, 16, -2, 12, -4, 14, -3, -34, -13, 0, -12, -40, -1, -6, -28, -20, -6}
}
, {{-21, -30, -3, 0, -15, -23, 6, -25, -7, -6, -5, 0, -31, -7, 5, -21, 18, -26, 12, -31, -9, -13, -8, -11, -10, 15, 7, 1, 5, 1, 10, -22}
, {0, -14, -10, -9, -13, -29, 37, -21, -24, 0, 1, 3, -37, -6, -14, 26, 27, -11, 1, -29, -4, 18, -10, -28, -23, 21, 5, -4, -11, -7, 8, -15}
, {-3, -6, 1, 8, -9, 16, 0, -7, -16, -2, -7, 5, -41, 8, 1, 6, 6, -17, -20, -15, -1, -3, 26, 3, -14, 15, 11, -11, -6, 17, 16, -16}
}
, {{-17, -22, 9, 4, 5, 0, -34, 2, -14, 6, 11, -2, -34, 20, -1, 6, 7, 16, 11, -8, 7, -6, 7, 2, 1, 2, 23, 5, 7, 9, 19, 1}
, {-10, -30, 10, 0, 11, 6, -17, -5, -16, 3, -4, -7, 0, 17, -18, -14, -2, -3, 2, 5, -3, -1, -4, 14, -7, -8, 0, -4, 6, 5, -1, -7}
, {-16, -29, 2, 10, 27, 13, 21, 8, -7, 6, 0, -20, 30, 13, -4, -25, 1, 7, -12, 16, 7, 8, 11, 13, -6, 30, 10, 3, -3, 15, 18, -4}
}
}
, {{{12, -5, -27, 4, -17, -12, -6, 19, -6, 15, 9, 3, 2, -4, 28, -9, 14, -1, -20, -9, -4, 2, -10, 10, 11, -8, -17, 6, 3, -10, -5, -6}
, {-5, -40, -63, -14, -40, -12, 12, -9, -23, -11, -13, 2, 8, -7, 0, -12, -32, -18, -11, 2, -16, -8, -18, -7, -12, -5, -35, -4, 0, 0, -30, 11}
, {-6, 20, 0, 9, -24, -19, -4, 12, 11, 17, 14, 15, 31, -4, 9, 7, -6, -8, 1, 4, 1, -13, -34, -6, 5, -12, -29, -4, 2, -4, -11, -7}
}
, {{1, 2, 1, 24, 9, -3, -17, 15, -8, 8, 9, 0, -13, -7, -25, -10, -5, 3, -2, -5, -1, -3, 6, 12, 15, 12, -2, -4, -3, -3, -1, -20}
, {-26, 1, -1, -5, -14, -5, -8, 2, -35, -5, -11, -5, -15, 11, 2, -32, 3, 1, -39, -12, 4, 7, -1, -1, 9, 9, 6, -4, -3, 3, 8, 13}
, {0, 6, 0, 7, 10, -11, -69, 6, 3, 6, 3, 11, -22, 19, 5, -2, -4, 8, 7, -28, 8, -20, 9, -10, 24, -20, -16, 2, 13, -3, 2, 26}
}
, {{10, 28, -9, -14, 10, 18, 14, -4, 17, -12, -18, 14, 2, -6, 5, 13, -15, 16, 4, 11, 5, 0, 34, 2, 12, -3, -13, -7, -8, -5, -36, 15}
, {-26, -3, -46, 21, -58, -29, -21, 14, -37, 5, 12, 21, 2, -21, 7, 4, -24, -11, -5, 13, -7, -20, 7, -14, 24, -38, -45, -4, -7, 2, -27, 13}
, {17, 3, -16, 9, -15, 0, -27, 7, -9, 15, -7, -8, -11, -15, -16, 8, 8, 4, -9, -14, 20, -6, -2, 13, 12, -38, -45, 12, 10, 0, -5, 11}
}
}
, {{{-4, -4, 10, -3, -1, -48, -8, -9, -17, 1, 8, -1, 6, -1, -1, -21, 16, -13, -4, 4, -1, -18, 6, -17, 11, -7, -1, 21, 3, 13, 12, -1}
, {0, -4, 17, 11, 10, 11, 10, -11, -3, 8, 7, -6, 18, 13, 10, -15, 2, 3, 2, 17, -1, -8, 7, 3, 4, 10, 10, 6, 23, 8, 12, 11}
, {14, 13, 2, 12, -1, 21, -15, -21, 8, -4, -6, -25, -3, 18, 13, 0, -7, 14, 8, 8, 8, -5, 36, 13, -3, 3, -1, 8, 9, -5, 12, -13}
}
, {{13, -2, 2, 9, -17, 6, -25, 16, -14, 8, 3, 12, 7, 2, 5, 13, -23, 6, 8, -2, -1, -7, 5, 11, 13, -33, -3, 19, -2, 3, -1, 10}
, {19, -63, 6, -5, 10, 15, 25, -15, -28, -51, -6, -50, -8, -5, -27, -20, -10, 4, 18, 11, 7, 18, 6, -3, -32, -4, -17, -36, 9, -2, -30, -45}
, {1, -4, 0, -19, -3, -1, 4, -8, 3, -51, -14, -18, 0, -4, -3, -18, -7, 2, -11, 11, -4, -9, 2, 12, -34, 16, 0, 8, 0, 7, -22, -20}
}
, {{-5, 15, -5, 8, 6, 4, 17, 2, 31, -13, -21, -29, 23, -14, -21, -16, -20, 6, -13, 15, -16, 7, 14, 2, 10, -3, 1, -19, -12, -12, -5, -33}
, {3, 4, 0, 7, -4, -11, 2, -16, 3, -24, 8, 3, -6, -3, -10, -15, -27, -25, -24, -5, -10, -22, 8, -1, -15, -11, -6, -25, -8, -19, -11, 9}
, {8, 5, 5, 10, 10, 15, -9, 7, 13, 3, -5, -4, 7, -4, 0, -6, 3, -2, -13, -9, -6, 9, 8, 12, -2, -29, 5, -2, 0, -5, -6, 13}
}
}
, {{{-18, 7, -30, 3, -30, -7, -2, 7, 6, 5, 1, 10, -20, -13, -3, -6, -4, -10, -5, -29, -8, -8, 2, -22, 5, -7, -30, -8, -25, -8, -23, -11}
, {1, -7, -14, -1, -35, 9, 3, -9, 3, 0, -20, -29, -40, -36, -6, 1, -26, -20, -13, -6, -7, -5, -10, -35, -29, -14, -16, -13, -6, -4, -5, -17}
, {11, -36, -11, 11, -30, 40, 2, -15, -11, -37, -19, -58, 9, -43, -41, -2, -49, 11, 5, 11, 16, 19, 3, -7, -24, 7, -14, -62, -1, -16, -4, -10}
}
, {{-16, -54, 6, -9, 12, 17, 14, 3, -52, 5, 18, 7, -4, 12, -2, -7, -1, 15, 19, -2, 24, 5, 12, 5, -23, 10, -5, -5, -1, 2, -8, -3}
, {-2, -38, 4, -4, 10, 9, 34, -2, 8, 1, -6, -28, -2, 4, -2, -19, 13, 12, 0, 30, 16, 43, 1, 3, -16, -4, 14, 7, 11, 12, 0, -37}
, {-43, 0, 6, 15, 7, -20, -20, -5, 17, -36, 9, -82, -35, -1, -39, -56, -17, 18, -32, 3, -18, -40, -1, 5, 0, -7, 11, 2, -7, -22, -12, -49}
}
, {{-7, 28, -19, -7, -12, -3, -4, -3, -2, 18, -11, -4, 9, -3, 15, 8, 0, 6, 12, 6, -3, 18, -2, 10, -5, -9, -9, 3, -1, -4, -16, 6}
, {-33, -38, 4, -13, -20, 24, 17, -2, -49, 13, 9, -2, -17, 14, 1, 24, 19, 5, 23, -16, 4, 21, -6, -14, -3, -1, 7, 10, -4, 5, 12, -18}
, {-3, -1, -12, -6, -10, 4, -6, -1, -2, 2, 11, 1, 2, 12, -3, 10, -1, -12, 6, 7, 7, 16, -5, -10, 17, -28, -7, 18, 5, -2, -17, -5}
}
}
, {{{17, -16, 4, 5, 6, 16, 18, 6, 12, -8, 32, -12, -11, 0, -25, 3, -7, -8, 31, 0, -7, 27, -10, -19, 22, -15, 4, -5, -9, -32, -5, -8}
, {6, -22, -6, -9, -18, 23, 21, -39, -2, -20, -46, 0, -4, -18, -26, 20, 5, -2, -2, 11, 2, 16, 23, -11, -41, 9, 18, -22, -14, -44, 24, -20}
, {-11, -36, 7, 3, 4, -3, 10, 1, 12, -39, -44, -1, -3, -16, -15, -26, -35, -11, -23, 5, -18, -7, 2, -8, -19, 18, 9, -10, -26, -18, 9, -17}
}
, {{-10, -3, -11, -15, -4, -39, -29, 21, -11, 0, -2, 10, 2, -2, -4, -8, -9, 10, -25, -26, -3, -25, -10, -7, -13, 13, 4, 12, -16, -1, -1, 8}
, {-17, -51, 0, 0, -15, -4, -29, -6, -43, 9, 8, 9, -3, 11, 1, -25, 11, 18, -3, -16, 10, -3, 15, -1, -24, 5, 0, -4, 2, 11, -6, 4}
, {-18, -26, 8, -1, 19, 11, 21, -13, 0, -1, 2, -53, -5, 3, -5, -13, -15, 19, -4, 12, 7, -3, 10, 2, -15, 7, 16, 1, 8, 6, -6, -24}
}
, {{-14, -27, 3, -1, 0, 1, -17, 10, -12, 12, -6, -8, 10, 6, -2, -21, -11, 19, -16, -5, -26, -9, -17, 8, 11, -22, 8, -5, -1, -8, -1, -4}
, {-25, -20, -3, -1, 4, 9, 7, 8, 0, 2, -3, -23, 12, 8, -22, -17, -4, 2, -12, 9, 4, 14, -6, 10, -13, 9, 3, 11, 16, 14, 5, 12}
, {-11, -28, 2, -10, -4, 5, -1, -2, -20, -18, 2, -3, -5, 2, 3, -10, 8, 2, 11, 15, 9, 4, -2, -7, -10, 6, 7, 14, 4, 21, 7, -2}
}
}
, {{{-9, -3, 13, 13, 16, 17, -3, 14, 4, -10, -3, -4, -8, -13, -3, 16, -1, 5, 2, 1, -9, -10, 14, 7, 16, 14, 20, 1, 11, 1, -1, -9}
, {13, -1, -15, 6, -8, -11, -8, 2, -13, 1, -2, -3, 2, 14, -15, -20, -21, -18, -18, -10, -10, 6, -9, -13, 0, -6, -17, -1, -9, 1, -9, 29}
, {-12, -9, 27, 15, 4, -5, -3, -8, 7, -11, 2, 16, -8, -6, -1, -2, -10, -7, -12, -7, -5, -34, 38, -12, 3, 15, 9, -1, 14, -15, 12, 8}
}
, {{12, 13, -4, -12, -10, 12, 12, 8, -1, 10, 5, -5, -9, 3, -1, 5, 3, 6, -13, 12, 10, 2, 0, -12, 7, 16, 6, 7, 0, 2, 2, -9}
, {-31, -44, 2, -19, -28, -17, -74, 6, -62, 15, 17, -1, -29, 11, 5, -23, 11, 11, -4, -22, 9, -34, -3, 2, -3, -11, 6, 6, 15, 7, 12, 11}
, {7, -22, 6, -2, 7, 16, 8, -18, -28, 2, 6, -16, -24, -9, -14, 3, 18, 14, 7, 3, 31, 11, 13, -4, -24, -46, -14, -10, 13, 4, -10, -36}
}
, {{6, -16, 5, -21, -13, -3, 6, 0, -5, 8, 13, 13, -1, 6, 2, -5, 3, 3, -8, -9, -7, -3, 1, 15, 20, -23, -20, 1, -5, 6, -2, 0}
, {-1, 11, -2, 11, 5, 6, -28, 1, -4, 4, -19, -23, 10, -13, -14, -14, -25, 13, -12, -4, 4, -11, -13, 24, 12, -43, -25, -10, 5, 6, -24, -2}
, {2, -4, 4, 12, 23, 17, 4, 10, 16, -47, -9, -44, 21, -3, -17, -22, -38, 9, -16, 24, -1, 3, 11, 16, -17, -2, 0, -13, -5, -8, 2, -15}
}
}
, {{{-15, 2, -9, 17, -11, 13, 6, 0, 17, 1, 15, 0, 7, -10, -4, -14, 4, 1, -14, -3, 2, -5, -18, -4, 12, 1, 3, 8, 6, 7, 3, 10}
, {-14, 19, -39, -15, -39, 15, -23, 17, -6, -19, -13, 2, -2, -30, -3, 7, -21, 0, 15, 8, -2, -13, -25, 6, -4, -14, -28, -24, -14, 8, -35, 7}
, {-28, 8, -6, 14, 6, -22, 9, 2, 1, 4, 7, 15, -9, -5, -16, -16, -6, 15, -8, 17, -27, -26, -41, -5, 20, -1, -15, 6, -36, -9, 6, 18}
}
, {{-3, 16, 6, 14, 14, -20, -27, 12, 11, 9, 31, -11, -21, 2, -10, 12, 1, -4, 16, -30, 5, -20, -26, -12, 11, -33, -4, 6, -8, -36, -8, -9}
, {-14, -13, -19, -5, -22, -36, -8, -9, -32, 1, -18, -10, 2, 3, -15, -21, -18, -48, -58, -33, -41, -14, -15, -11, 0, -18, -25, -13, -27, 1, -6, 19}
, {0, 5, -3, 9, -5, -1, -17, 23, 3, 31, 11, 32, 19, -5, 36, 21, 12, -3, 16, -2, 12, -9, -11, 0, 27, -28, -1, 11, 4, 5, 8, 24}
}
, {{-36, 17, -5, 3, -13, 6, -29, 8, 7, -5, 8, 14, -4, 19, -21, 4, -1, -4, 9, -23, 13, -37, -5, 7, 26, 4, 11, -3, 1, -15, 3, -2}
, {-41, -30, 15, 33, -1, 12, -16, -13, 3, -2, 4, -18, -4, 18, -4, 15, 9, 0, 5, -14, -7, 14, 8, -5, -8, 26, 25, 2, -3, 2, 20, -6}
, {10, 5, 4, 1, 11, 9, 22, -14, 10, -20, 6, -26, 4, 0, -29, 5, 4, -1, 6, 9, -13, 22, 8, -7, 13, 7, 6, -1, -13, -30, 6, -33}
}
}
, {{{17, 32, -22, 12, -4, 4, -3, 3, 9, 5, -14, 11, 1, -2, 1, 4, -12, 0, 4, -6, 3, -12, -34, -6, 9, 13, 4, 2, -19, 0, -5, -3}
, {-3, -9, -24, -8, -34, -19, -23, -9, 6, -15, -21, -10, -3, -36, -16, -9, -5, -9, -5, -22, -17, -18, -30, -37, 6, 1, -22, -20, -42, -5, -21, 11}
, {-15, -9, -21, 3, -2, -40, 5, -3, -16, 0, 2, 34, -1, -7, 7, -9, -12, -19, -32, -25, -19, -24, -55, -43, -10, 16, -6, -3, -41, 3, 1, 18}
}
, {{-4, 25, 4, 7, 14, -16, -2, 11, 12, 18, 12, 25, 28, 8, 20, 16, 9, 6, 0, -5, -6, 11, -5, 13, 6, 10, -6, 3, 13, 4, 1, 26}
, {-7, 28, 1, -18, -13, -18, -17, -50, 5, -28, -15, 19, 12, -6, -9, 26, -28, -54, -12, -37, -52, -7, -18, -43, 12, -22, -6, -37, -37, -25, -7, 28}
, {-35, -3, -21, 2, -15, -74, -31, 13, -10, 11, 18, 26, -7, -6, 14, 22, 18, -23, 2, -28, 7, -66, -48, -26, 16, -13, -19, 8, 3, 13, 0, 17}
}
, {{10, 16, 10, 15, 4, -3, -4, 11, 10, 5, 17, -16, 13, 8, 4, -4, 19, 10, 9, -1, -1, -17, -8, -8, -2, -4, -11, 6, 12, 8, 17, -12}
, {14, 2, 24, 2, -8, -37, 8, -27, 9, 5, 1, -7, -4, -16, 2, -27, 4, -9, -29, -11, 3, -30, -14, -50, -27, 7, -8, -10, -9, -10, 5, -14}
, {0, 4, -12, -2, -21, 15, -70, 15, -4, 25, 15, 21, 11, 8, 17, 16, 17, -7, 5, -16, 4, -20, -16, 0, 8, -45, -34, 25, 7, 16, 10, 18}
}
}
, {{{-22, -5, -4, -1, -26, 13, -1, -13, -22, 26, 9, 4, 10, -9, 15, 4, 38, -7, 6, -21, 5, -28, 13, 12, -1, -19, -18, 9, 13, 17, 0, -1}
, {-30, 0, -12, -4, 4, 0, -39, -2, -11, -6, 0, 5, -10, -15, 5, -19, -4, 4, -6, -22, 10, -38, -8, -6, 1, -17, -10, -3, -11, 10, -14, 6}
, {8, -4, -8, 1, -17, -7, -9, 1, -4, -11, 1, -16, -37, -10, -16, 18, -12, -14, 15, -10, 5, -7, -9, -9, 13, 6, 4, -15, -10, -22, 13, -14}
}
, {{-5, -33, -5, -19, 2, 15, -1, -11, -26, 4, -19, 0, 15, -6, 8, -26, -17, 8, -25, 21, -1, 14, 8, 13, -43, 10, -13, -1, 7, 10, -5, -4}
, {0, -23, 6, 16, 7, 14, -4, 4, 1, -7, 9, -35, 17, 7, -1, -39, 5, 8, -18, 8, -3, 4, 12, 12, 5, -10, -13, 15, 0, 8, 9, 7}
, {-32, -6, 1, -5, 1, -4, -4, 3, 9, -18, -2, -47, -20, 7, -26, -31, -15, 7, -13, 8, -9, -5, 5, 16, 19, -15, 15, 10, -9, -12, 1, -22}
}
, {{9, 8, 4, -15, -26, 6, 15, -50, -30, -16, -6, 9, -20, 5, -12, -1, 12, -2, 7, -3, 8, 9, 6, -24, -21, 10, 18, -7, -1, 6, 1, -9}
, {-15, -30, 4, -26, -10, -5, 11, -7, -54, 5, -3, -12, -9, 4, 10, -13, 13, 14, 9, 11, 11, -5, 1, 4, -11, -3, 12, 12, 11, 20, 14, -15}
, {-12, -11, -4, 4, 15, -4, 3, -2, 11, 15, 11, -17, -3, 0, 2, -5, 2, 7, -11, -2, 10, -14, -12, -2, 3, -6, 5, 5, -2, 0, -10, -18}
}
}
, {{{17, -15, 3, 8, 22, -16, 0, -9, -12, 1, -2, -5, -35, 4, -20, 16, -14, 20, 5, -5, 6, -16, 7, 12, -13, 18, 15, -1, 2, -5, 0, -3}
, {-4, -10, -10, -21, -48, 4, 6, -11, -20, -16, -6, -2, 8, -23, 6, -6, -16, -33, -1, 16, -9, 13, -14, -44, -17, 8, -14, -2, 3, -1, -1, 11}
, {2, -18, -16, -27, -38, 27, 12, 5, -32, 1, -3, 32, -13, -10, -10, 10, 20, 8, 20, 12, 24, 25, 10, -9, 2, 7, -32, 5, 18, 16, -7, -3}
}
, {{-10, -19, 5, 7, 19, -4, -10, 4, 11, 8, 3, -3, 13, 14, 15, -18, 8, 0, 26, -1, -10, -15, 11, 1, 1, -10, 12, 12, -2, 12, 3, 6}
, {1, 1, 16, 17, 12, -28, -34, 6, 15, -8, 3, 12, -8, -1, -27, -14, -13, 17, -20, -26, -3, -65, -5, 25, 2, 16, 11, 0, -3, -9, 16, -23}
, {18, -13, -6, -9, -31, -32, -3, -21, 1, -40, -51, -10, 5, -54, -19, -12, -29, -38, -22, -2, -15, -4, -38, -23, -19, -1, 2, -55, -16, -9, -9, -31}
}
, {{-18, -11, 5, -4, -13, 14, 8, -4, 5, -4, -1, -12, -3, -17, 3, -42, 6, -10, -16, -3, 4, -14, -2, 5, 12, 17, -3, 3, 4, -3, 19, -10}
, {-24, 11, -12, 1, 17, -33, -29, 9, 17, -4, 9, -18, 12, 7, 4, 0, -3, 2, -14, -1, -5, -27, -20, -4, 7, -34, -23, 16, 3, -2, -21, 15}
, {-17, 0, 12, 18, 18, -40, -19, 19, 15, -25, 19, -1, 8, 15, -26, -15, -29, -16, -17, -72, -22, -41, -25, 12, 11, -11, -12, 19, -17, -16, -12, 20}
}
}
, {{{-13, 2, -11, -3, -4, -3, -3, 24, 11, -13, -7, -39, 16, -9, -32, 1, -7, -5, -11, -14, 9, 3, 8, 4, 2, -24, -8, -4, -4, -8, 5, -1}
, {2, 5, -21, -9, -14, -3, 9, 2, -28, -3, 4, 21, -11, 7, 6, 3, -11, -1, -33, -1, 9, 1, 8, 12, 20, -5, -25, 4, 14, 5, -13, 11}
, {-8, 7, 11, 10, 21, -23, 9, 16, 14, 1, -3, -12, 11, 0, -9, -4, -5, -2, -3, 4, 9, -20, -2, -1, -4, 6, 7, 1, 7, -5, 8, 7}
}
, {{-12, 8, 8, 15, 14, 11, 25, 15, 15, -39, -5, -49, 6, 11, -33, -26, -10, -5, -4, 10, -3, 9, 4, -6, 24, 5, 2, -11, 0, -15, -10, -27}
, {-8, -10, -6, -25, -12, -19, -4, 25, -37, 20, -10, -9, 5, 1, 12, -25, 6, 7, -23, 6, 4, 7, 9, 5, -2, 7, 4, -3, 2, 7, 8, 0}
, {3, 0, 12, -2, 2, -18, -34, 6, -2, 24, 6, 30, 4, 17, -3, 5, 13, 0, -5, -27, 11, -38, 9, 6, 12, -75, -19, 9, -3, -3, -2, 31}
}
, {{6, 7, -12, -16, 10, 12, 18, -15, 14, -46, -19, -24, -16, -14, -14, 14, -27, 12, 10, 9, -2, 28, 16, -5, -2, -6, -5, -26, -29, -18, -7, -15}
, {-10, 1, -57, 11, -42, -32, -30, 12, -1, 13, 7, 7, 15, -10, 3, -28, -14, -18, -23, 4, 12, -36, -16, -7, 14, -22, -29, -5, -16, 2, -30, 11}
, {11, 13, -19, 21, -15, 2, -65, 19, -5, 7, -3, 2, -27, -32, -29, 0, -42, 24, -2, -28, -4, -41, -49, 4, 8, -73, -72, -17, -25, -11, -56, 13}
}
}
, {{{22, 21, -2, -12, 6, -6, -13, -6, 15, 5, 12, 5, 10, -10, 11, 13, -10, -16, -1, -3, -23, -6, -30, -24, 10, 2, -8, 0, -29, -1, -1, 3}
, {20, 7, 5, 3, -4, 22, 8, 24, 11, 8, 6, 10, -1, 10, -5, 10, 7, 5, -1, 6, -10, 21, -1, -2, 24, 4, -12, 2, -3, -6, -8, -7}
, {-6, 3, -2, 2, 2, 6, -2, 1, -12, 12, -10, 19, -19, -9, -12, 3, 3, -16, -2, -8, 15, -6, 7, -18, 13, 7, 8, -25, -6, -8, -5, -6}
}
, {{-1, 17, 0, 8, 11, -1, -3, 8, 26, 3, 3, 7, 4, 6, 5, -6, 3, 2, -2, -1, 0, -13, 6, 18, 11, -20, -1, 3, 4, 1, -3, 5}
, {24, -9, -8, -21, 4, 16, 1, -4, -25, -10, -1, -22, -2, 7, -12, -11, -6, 12, -3, 13, 14, 20, -6, 6, -6, 6, 0, 1, -2, -1, 0, 7}
, {-2, -6, 3, -34, 2, 10, 16, -3, -27, 5, 0, -12, 5, 12, -3, 4, 11, 11, 11, 22, 20, 28, 10, 1, -26, 48, 11, -14, 6, 19, 34, -1}
}
, {{0, -2, -3, 1, -1, -12, 3, 0, 17, 7, -4, 11, -8, 10, 8, 3, -8, -10, -15, -17, -21, -25, -17, -10, 11, 11, 10, 4, -18, 0, 2, 3}
, {19, -4, 10, -11, 9, -13, -4, -15, 9, -10, -19, -16, -35, -7, -54, -8, -41, -10, -8, -9, -52, 9, 17, 9, -7, 16, 24, -10, -33, -24, 1, -12}
, {14, -12, -19, 0, -19, -14, -3, -61, 8, -71, -67, -31, -17, -65, -65, -12, -86, -1, -19, 3, -61, -3, -39, -43, -76, -9, -11, -90, -95, -90, -22, -50}
}
}
, {{{4, -42, -26, -9, -10, -6, 7, 17, -30, 7, -3, -3, -25, -16, -6, -11, 2, 0, -18, -7, -1, -4, 2, 3, 6, 6, -16, -4, 2, -12, -2, -33}
, {9, 6, -7, 6, 1, -9, -4, -7, 28, -28, -15, 10, -11, -10, -12, 18, 0, -21, -4, 0, -16, 14, 9, -12, -30, -7, -4, -12, -15, -29, 5, -6}
, {-20, 13, -7, 8, -14, 0, 20, 6, -10, -2, 8, -3, 8, -5, 6, -23, 13, -8, -3, 7, 21, -19, 15, 3, -6, 13, -7, -1, 6, 1, -3, -12}
}
, {{-15, -15, -3, 2, 5, -14, -22, 9, 0, 18, 14, 3, -22, 11, -10, -2, 2, -12, -3, -19, 14, -28, -3, 2, 0, -3, -1, -3, 1, -7, 0, -7}
, {-13, -25, -4, 6, 14, 4, -6, -2, 25, 10, 2, -14, 3, 5, -1, -27, -1, 4, -11, -2, -2, 19, 6, 7, -19, -5, 5, 10, 4, 13, 14, 13}
, {-33, -13, 8, 2, -6, 28, 17, -30, -7, 1, -6, -10, 12, 1, 3, -9, 0, 6, 6, -8, 12, 26, -2, 2, -4, 15, 3, -9, 19, 13, 3, -21}
}
, {{13, -2, 10, 6, 16, 12, 27, -4, 26, -9, -2, -56, -9, 14, -39, -3, -6, 20, -9, -6, -14, 25, -2, -4, 0, -29, -12, 0, -3, -4, -12, -43}
, {12, -43, 7, -6, -5, 6, 23, -29, -8, -38, -16, -23, 10, -22, -13, 8, -9, 13, 0, 12, 3, 19, 13, -10, -27, 28, 7, -20, -5, 3, -9, -9}
, {18, -22, -9, -29, -10, 4, 34, -33, -6, -34, -11, 11, -6, -11, -1, -11, -15, 12, -3, 10, -3, 12, -7, -3, -45, 15, 14, -12, -5, 4, -2, 5}
}
}
, {{{2, 5, -34, -9, -67, -3, 10, 0, -3, -12, 9, -5, -39, -35, -52, 21, -32, -14, 25, -11, 11, 10, -1, -32, 20, -13, -27, -11, -18, -39, -10, -23}
, {17, -1, -42, 3, -54, 14, 11, -13, -8, -32, -54, -29, -10, -62, -31, 12, -21, -3, -18, -6, -9, 19, 10, -14, -38, -10, -24, -54, -19, -24, -20, -11}
, {-3, -38, -17, 8, -29, 5, 14, 20, -17, -3, -27, -19, 1, -20, -11, 26, -32, -1, -13, -15, -16, 16, -24, -30, 2, -7, -11, -11, -30, -41, -12, -15}
}
, {{-13, 7, 9, 24, 9, 0, -49, 5, 23, -8, 27, 15, -12, 5, 11, -9, -2, 11, -13, -37, 13, -16, 11, -5, 1, -16, -9, 3, 7, 8, -7, 5}
, {-57, -47, 5, 10, -4, 0, -20, -16, -12, -2, -7, -1, -18, 6, 6, -35, 5, 4, 6, -7, 13, 1, 15, 4, -20, 2, 6, 7, 13, 8, 3, 6}
, {-25, -19, 5, 1, 9, -1, 11, -6, -12, -13, -4, -22, -29, 0, -17, 4, 1, 4, 22, 18, 12, 10, 9, 8, -18, 28, 18, 9, 15, 6, 16, -33}
}
, {{13, 5, -19, 4, -16, 1, 3, 8, -6, 9, 7, -6, 14, -1, 10, 0, -12, -24, -12, -4, -12, -6, -7, 1, 8, -24, -38, 5, -12, 6, -13, -8}
, {5, 5, 10, 2, 13, -1, -24, -1, -8, 7, 0, -1, 12, -1, -4, -16, -8, -6, 0, -15, -2, -15, 18, 7, 13, -28, -24, 7, -3, 2, -10, 9}
, {-3, 3, -2, 9, -11, 6, -11, 8, 8, -7, 11, 2, -16, 2, 3, 2, -6, 5, 7, -18, 3, -3, 10, -9, 10, 2, -16, -4, -7, -7, -6, 2}
}
}
, {{{19, 13, 14, 4, 11, -26, -24, 9, 33, -6, 12, -19, 16, 5, -18, -14, -27, 5, -16, -7, -11, -27, -25, 8, 1, -14, -1, -1, -8, -16, -9, 23}
, {-20, -25, 17, 14, 8, -22, 11, 3, -8, -17, 5, 1, -1, 16, -4, -27, 4, -6, -38, -6, 6, -5, 18, -4, 1, -16, -11, 7, 12, 7, 9, -2}
, {-31, 17, 9, -2, 3, 20, -6, 9, -2, 3, -2, 10, 21, 15, 37, 0, 7, 12, 14, 7, 14, 7, 17, 1, -1, -5, 5, 6, 11, 8, 16, 6}
}
, {{8, -36, 10, 7, 22, 6, 3, 7, 18, -19, -3, -14, 9, 11, -15, -14, -22, -3, -6, -2, 7, 7, -5, 14, -3, 5, 12, -7, 0, -2, 1, -5}
, {10, -14, 11, -7, -31, 24, 8, 13, -15, -22, -14, -27, 11, -4, 0, 8, 3, 17, 8, 10, 7, 17, 14, 3, 2, 31, 17, -20, -2, 4, 15, -13}
, {23, -18, -30, -46, -35, 10, -7, -13, -20, -40, -18, 7, -4, -45, -16, 14, -23, 7, 19, 9, -2, 15, 4, -10, -22, -30, -47, -35, -3, -17, -51, -13}
}
, {{-19, -14, 1, 8, -7, -3, 10, -5, 1, -3, 3, 4, -20, -6, -19, -37, -6, -21, -5, 7, -18, -1, -2, -20, -8, -1, 3, 1, -8, -12, 6, -10}
, {3, 27, 0, 0, 9, 1, 39, -4, 10, -16, 9, 20, -11, -6, -7, 11, 12, -14, -2, 3, 0, -2, 43, -9, 5, 14, 6, 8, 3, -13, -10, -5}
, {12, 2, 7, 1, 18, -15, 24, -6, 2, -23, -16, 5, -26, -1, -17, -15, -30, -11, -53, -42, -42, -4, 21, 11, -35, -15, 0, -4, -48, -19, -12, 10}
}
}
, {{{7, 15, 1, -22, -26, -12, -3, 2, 4, 6, -2, 13, -4, -7, 2, 10, 33, 2, -2, -8, 5, -3, -28, 0, -2, -8, -15, -4, 1, 7, -26, 15}
, {1, 29, -12, 1, -10, -12, 2, 12, 18, 12, -5, 23, 14, 1, 7, 16, 4, -16, 6, 1, -13, 3, -8, 14, 7, -13, -16, 9, -10, 18, 0, 0}
, {3, 13, -22, -7, -8, -13, 12, 1, 3, -1, 10, 13, -39, -37, -25, 1, -3, -23, -5, -2, 0, -16, -30, -47, 12, -36, -18, -5, -23, -36, -35, -11}
}
, {{1, 30, -1, 20, -1, -19, -10, 7, 18, 10, 10, -3, -7, 1, 7, 28, 9, -19, -17, -15, -14, -1, -21, -26, 19, 10, -7, 11, -9, 1, -1, 4}
, {-3, 34, -15, 2, -12, -19, -30, 2, -9, 14, 1, 27, -20, -1, 17, 25, 9, -28, 4, -48, -14, -21, -34, -40, 6, -26, -24, 8, -11, -13, -4, 11}
, {-7, -3, -13, -12, -5, -13, -27, -16, -9, 24, -19, 9, -24, -10, 9, 20, -18, -78, 9, -75, -23, -31, -102, -49, 20, 9, -5, -9, -44, -5, 5, 2}
}
, {{-19, 12, 18, 3, -6, -10, -22, -3, 4, 0, 0, 20, 1, 19, 11, 18, 3, 3, 0, -19, -10, -7, -24, 10, 10, -6, -2, 10, 2, 4, 3, 4}
, {-14, -5, 19, 22, -4, 6, -17, -5, 3, 10, -8, 12, -18, 10, -2, -20, 6, 10, -25, -35, -16, -28, 0, 3, 1, 5, 5, -2, -10, 1, 19, 1}
, {-25, -24, 13, 22, 6, -27, -21, -13, -1, 11, 5, -7, -6, 12, 28, -52, 10, 7, 5, -37, -4, -45, 6, 8, 5, 12, 22, 12, 18, 21, 39, -5}
}
}
, {{{-17, -3, 4, -3, -2, -6, -1, -4, 10, 2, 12, -26, -10, -3, -16, -22, -16, 16, -4, -8, 2, -16, 4, -8, 14, -15, -24, -6, -14, -13, -11, -22}
, {-40, -3, 10, 0, 9, -10, 2, -3, 15, -12, -11, -16, -5, 7, -6, -9, 5, -12, -15, -20, -6, -16, -4, 4, -2, -5, 9, 4, -10, -7, 6, 6}
, {2, -19, 16, 4, -5, -14, -22, -8, 4, -15, -5, -10, -1, -10, 11, 1, 9, -6, -3, -3, 10, -5, 17, -9, -21, 7, 14, -8, 13, 9, 16, -1}
}
, {{-6, -51, 9, -26, 4, 11, 8, 10, -12, -10, 4, -14, 4, 2, 7, -20, 9, 12, -18, 13, 19, 6, 16, 22, -17, 27, 0, 9, 15, 20, 13, -3}
, {-37, 2, 7, -6, -12, -6, -52, 2, -29, 17, 13, 8, -41, 6, 9, 24, 18, 20, 13, -29, 19, -36, -6, -10, 2, -7, -11, 9, 21, 23, 12, 7}
, {13, 16, -21, -11, 2, 10, -3, -17, 4, 22, -16, -6, -30, -12, -23, 8, -6, 5, 15, 13, 5, 6, 4, 3, 3, -55, -46, -17, 1, -4, -41, -3}
}
, {{-14, -16, -23, 1, -13, -39, -22, 1, -6, 8, -3, 11, 1, -28, 7, 1, -8, -11, 3, 6, -20, -28, -31, -12, -11, -20, -29, -8, -4, -7, -32, 8}
, {13, -4, -18, 11, 7, 13, -23, 14, 16, 2, -2, -16, 8, -10, -1, -13, -21, -12, -6, 2, -1, -4, -9, 5, 3, -61, -50, 13, -10, -7, -21, -3}
, {28, 0, 8, 1, -3, 7, 27, -39, 11, -55, -13, -16, -3, -14, -44, -14, 10, -9, -29, 20, -3, 32, 26, -12, -39, 0, 22, -27, -8, 1, 8, -10}
}
}
, {{{-26, -39, 6, 9, 6, 11, -2, -1, -11, 6, 3, -8, -12, -15, 15, -7, 19, -14, 6, -12, 1, -7, -10, 0, -10, 1, -6, 8, 13, 6, 15, -12}
, {-11, -4, -13, -1, -7, -21, -42, 7, -11, -5, -3, 2, 8, 6, -10, -3, -14, -18, -10, -5, -8, -19, -21, -1, -9, -4, 5, -11, -7, -15, -4, 3}
, {-15, 6, -9, 22, -12, -10, -6, 13, 4, 20, -1, 1, -31, -6, -21, 12, 1, -17, 12, -8, 14, 1, 13, 0, 15, -12, -21, -13, 0, -23, 0, 14}
}
, {{1, -24, 5, 2, 24, 3, -20, -15, 0, 5, 15, 2, 1, 16, 6, -25, -4, -2, -3, 0, 5, -6, -13, 7, 6, -20, 7, 2, 4, 2, 1, 7}
, {-20, 2, 19, 4, -5, 1, -15, 6, 20, -6, 2, -55, 6, 3, -6, -8, 4, -1, -17, 0, -4, -18, 2, 14, -5, -16, 0, 9, -12, 5, 9, 6}
, {-33, -5, 6, -12, 7, 15, 40, -4, -9, -1, 3, -44, 2, 10, -11, -18, 5, 13, -16, 3, 2, -2, 25, 0, 7, -18, -13, 0, 1, 15, 12, -19}
}
, {{6, -2, -7, 5, -31, 8, 0, -1, -7, 1, 3, -22, -22, -7, -4, 8, 13, -11, 13, -16, -1, 3, -32, -23, -10, -17, -14, -10, 0, -5, -11, -3}
, {-3, -18, -1, -34, 11, 24, -3, 4, -29, -15, -9, -31, 3, -7, 0, -4, -14, 8, 9, 11, 5, 7, 1, 17, -12, -8, -17, -10, 3, 8, -29, -21}
, {4, -17, 19, -7, -1, 18, 8, -5, -12, -5, -10, -3, -6, -13, -2, -14, -3, 15, 18, 16, 28, 18, 6, 4, -15, 13, 26, -12, 4, 1, 13, -17}
}
}
, {{{9, -8, -3, 3, 1, -8, -4, 8, 1, 5, -2, -14, -2, -3, -1, 15, 4, 21, 6, -4, -10, 5, -1, 11, 0, 8, 4, 0, -2, 15, -2, -2}
, {-12, -6, -14, -22, -16, -20, -6, -3, -24, -7, 8, -5, 20, -4, 5, 0, -1, -13, -12, 19, -8, -32, -15, -9, 0, 0, -9, -10, -7, -4, -12, 0}
, {-5, 8, 2, -12, -8, 10, -34, 21, 10, 21, -1, 29, 14, -1, 24, 9, 17, 2, 9, -12, 15, -8, 11, -2, 18, -32, -7, 7, 1, 8, -11, 44}
}
, {{-27, -4, -2, -8, -3, -2, 5, -20, 2, -4, -4, 15, 1, 1, 6, -6, 4, -1, 13, 11, 4, -11, 8, 17, 10, 8, 10, 5, 6, 0, 7, -6}
, {13, 5, 22, 12, 2, -3, 10, -37, 1, -11, -8, -5, -30, -5, -11, 10, 7, -2, 3, -11, -18, -5, 8, -2, -18, 25, 21, -14, -6, -23, 24, -12}
, {7, -5, 0, -13, -7, -6, 18, -14, -4, -37, -34, -37, 7, -42, -48, -10, -43, -27, -22, 7, -34, 17, -64, -7, 1, -3, -8, -40, -41, -54, -9, -34}
}
, {{-20, 4, -9, 8, -5, -9, -21, 12, 10, 13, 4, 1, -9, 13, -3, -15, 4, 3, -4, -16, 1, -22, -17, -4, 4, -26, -33, 0, 0, 4, -7, 14}
, {-41, -47, 1, 14, 24, -23, -35, 10, 15, -6, 1, -20, -1, 14, 5, -47, 4, 12, -22, -16, 3, -39, -1, 16, -2, 8, -9, 13, 12, 14, 8, -14}
, {-37, -9, 10, 18, 15, 1, -40, 12, 15, -22, 13, -38, -2, 41, -21, -42, -1, 16, -14, 1, -1, -43, -3, 33, 3, -15, -11, 26, 6, 24, 6, -3}
}
}
, {{{15, 9, -11, -18, -24, -4, -9, -16, 0, -1, 3, 11, -13, -11, -11, 14, 6, -11, 5, 8, 9, 0, -11, 7, 14, -3, -7, 6, 12, -6, 4, -2}
, {9, 31, -24, -4, -6, 8, -4, 16, 6, 31, 3, 12, 8, 0, -4, 21, 9, -7, 11, -10, 6, 0, -1, 1, 31, -8, 7, 6, -6, -2, -17, 12}
, {-2, 13, -12, -13, 8, 1, 15, 10, -5, 3, -3, 27, 1, -6, 10, 6, 4, -8, 7, 0, 0, -8, -8, 0, -3, 25, 7, -7, -4, 5, 4, -1}
}
, {{8, -7, 6, 9, 7, 2, 4, -9, 22, 1, 25, -32, -33, -4, -12, 18, 31, -27, 3, -14, 0, 3, -39, -38, 22, -12, 0, 2, -13, -24, 6, -9}
, {11, -8, -26, -37, -34, 1, 1, -48, -15, -60, -53, 3, 2, -38, -23, 8, -41, -41, -6, -6, -33, 3, -29, -42, -21, -1, -22, -44, -36, -23, -12, 3}
, {23, -33, 1, -7, -6, 14, 13, -5, -24, -4, -29, 0, 2, -7, -8, 9, -5, 2, 17, 4, 0, 18, 9, 12, -18, 3, 19, -11, 7, 8, -5, 12}
}
, {{-4, 15, -6, 24, 1, 21, 12, -4, 3, -14, 10, 0, -25, 1, -33, 29, -6, 23, 15, 2, 3, 11, -3, 2, 29, 5, 0, -11, 4, -16, -1, -31}
, {-8, -22, -5, 4, 4, 18, -16, -25, 8, -15, -33, -15, -15, -3, -10, -8, 8, -2, -3, -3, -10, 18, 23, 9, -42, 13, 12, -7, -8, 7, -1, -4}
, {18, -20, 15, -2, 15, 3, 45, -34, 17, -33, -20, 1, 7, -14, -7, 18, -6, 8, 4, 19, 1, 11, 31, -13, -9, 38, 23, -19, -2, -4, 1, -2}
}
}
, {{{17, 20, -1, 6, -23, -3, 4, -15, -6, 4, 1, 8, -22, -1, -26, 11, 0, 4, -4, 4, -20, 8, -38, -20, -11, 5, -20, -10, 6, -2, -7, -12}
, {4, -13, -5, -5, -4, 7, 5, -12, -13, 12, 3, 10, -6, 1, 8, 4, 22, -5, 5, -5, -10, -1, -7, -2, 1, 6, 1, 3, 12, 10, 7, -7}
, {-20, 0, 7, -1, 3, 3, -10, 7, 5, -5, 4, -2, 13, 15, 17, -14, 6, -1, -16, 2, -4, -23, 12, 5, -5, 8, -2, 9, -1, 2, 1, 13}
}
, {{-15, -14, -17, 1, -28, -11, -11, 14, -22, 10, 21, 25, -2, -2, 15, -5, -12, -5, -3, -16, 1, -19, -13, 3, -11, 8, -16, 2, 0, -3, 2, 18}
, {-1, -19, 3, -12, -12, -1, 7, -39, -33, -11, -14, -40, -5, -12, -29, -5, 2, -2, 7, 3, 5, 3, -4, 4, -9, -32, -10, -33, 14, -21, -25, -32}
, {2, -3, 11, -18, 0, 8, 18, 8, -8, 3, -7, -22, 17, 2, 2, 15, 0, 16, 11, 4, -1, 15, 22, 6, -6, -2, -8, -4, 8, 11, -27, -2}
}
, {{-7, 4, -4, 9, 8, 20, 21, 16, -10, -3, -14, 1, 28, 3, 4, -16, -17, -6, -2, 24, 2, 9, -11, -2, 11, 1, -14, -1, 0, 10, -17, 12}
, {-33, -22, 3, -4, 9, -1, 1, -1, 9, -18, 4, -1, -18, 14, -12, -29, 3, 4, -16, 6, 20, 14, 12, 14, -12, -1, 20, 22, 11, 7, 11, -25}
, {-24, -10, 11, -7, 7, 7, -18, -21, -5, 0, -2, -8, -51, -1, -13, -13, 14, 11, -2, -26, 8, 12, 5, -5, -11, 16, 13, 3, 12, -2, 10, -32}
}
}
, {{{11, -10, 9, 14, -2, 8, 7, 2, -7, 6, 11, -3, 4, -9, -6, 12, 11, 2, 5, 6, 8, 0, -11, -3, 9, 8, 2, -3, 9, 12, 6, 6}
, {-37, -4, 7, 12, 4, -36, -8, 6, 11, -7, 0, -4, -7, 17, -5, 1, 3, -1, 3, -21, -6, -40, -7, 8, 4, -2, -11, -3, -18, -1, 6, -2}
, {13, -4, 1, 15, -21, 18, 4, 8, -4, 4, 7, -33, -25, -6, -37, 3, -8, -17, 8, -19, 3, 10, -26, -12, 3, -6, -7, -17, -2, -15, 2, -16}
}
, {{-17, -12, 12, 7, 30, 14, -7, 12, 2, -10, -6, -16, 5, 0, 8, -20, -9, 11, -16, 12, 1, -6, 2, 15, -17, -9, 0, 9, 5, 3, -8, -25}
, {-3, -30, 3, -15, 8, 4, -13, -2, -12, 5, 6, -10, 2, 7, -3, -12, 2, 3, -14, -1, 16, -5, 11, 8, -1, -23, -11, 15, 10, 6, 3, 14}
, {-10, -4, 11, -29, 2, 1, 17, -1, -12, 11, 8, -21, -4, 16, 3, 3, 11, 6, 15, 18, 22, 15, 11, 1, -6, -16, 5, 7, 19, 9, 11, -6}
}
, {{-4, 16, -40, -8, -30, -29, 10, 4, -6, -2, -3, 1, -1, -31, 5, -12, -2, -38, 6, 6, 1, 0, -42, -50, 14, -24, -28, -17, -16, -9, -26, -11}
, {-8, 15, -45, -12, -49, -1, 1, 0, -6, -4, 1, 13, -9, -46, 6, 8, -7, -20, 10, 1, -18, -1, -18, -26, 10, -17, -3, -8, -32, -24, -7, -3}
, {-6, 8, -24, -21, -35, -18, -5, -14, -8, 16, -7, 32, -19, -35, 5, 3, 9, -16, 11, -7, 3, -27, -35, -27, -6, 9, -8, -15, 1, -10, -9, 1}
}
}
, {{{9, 21, -28, -31, -24, 4, 5, 2, 11, -6, 2, 11, -15, -41, -11, 1, 20, 8, -15, 0, -9, -10, -43, -12, -15, 10, -22, -19, -12, -21, -23, -10}
, {-22, -4, -22, -8, 13, -6, -5, 12, -15, 18, 1, -1, 6, 2, 18, -17, 0, -16, -8, 11, 4, -6, -50, -21, 13, 5, 1, 8, -7, -4, 4, 8}
, {16, -6, 8, 9, 16, -21, -5, -10, -7, -5, -6, 1, 0, -10, 1, -8, -1, 1, 2, 7, 8, -8, -2, -5, -7, -14, -10, -1, 18, -3, 7, 19}
}
, {{-2, 31, -10, -20, -1, -7, -3, -36, 2, -12, -21, 14, -11, -13, 6, 11, -10, -3, -38, -10, -23, -23, 11, -10, -7, 1, 7, -28, -10, 2, -3, 16}
, {-84, 5, 0, -1, -20, -39, -38, 6, -71, 11, 12, 23, -53, 10, 12, -22, 17, -9, 6, -51, 11, -22, -25, -18, 6, 7, 13, 12, -7, 7, 15, -15}
, {-10, -11, 14, -17, 10, 17, 13, -23, -14, -4, 3, -50, -40, 4, -8, -15, 3, 6, -6, 25, 15, 12, 14, 23, -16, 1, -6, 6, 12, 1, 0, -61}
}
, {{4, 0, -2, -3, -13, -35, 14, 11, -12, 22, 4, 18, 12, 1, 9, -27, -3, 11, -24, 15, 2, -4, 9, 4, -11, 5, 1, 2, -7, 2, 0, 8}
, {-38, -3, -9, 36, -18, 2, -61, 11, -1, 12, 10, -10, -7, 3, 15, -8, -5, -1, 4, -15, 2, -23, -3, -3, 17, -57, -30, 2, 7, -7, 1, 4}
, {1, -12, 4, -30, 1, 13, 8, -29, -6, -38, -12, -40, -12, -12, -2, 1, 8, -28, -5, 16, 10, 11, 28, 1, -32, 8, 21, -20, 18, 10, 27, -19}
}
}
, {{{-39, -26, 16, 15, 6, 1, -5, 10, -27, 15, 9, 0, 12, 9, 1, -17, -10, -13, -1, -4, 1, -17, 18, 11, 11, -17, 17, 3, -5, 5, 6, 6}
, {-40, -7, 10, -1, 14, -26, -39, -3, -7, 9, 14, -6, -20, 10, 2, 1, 8, -14, 12, -43, -5, -27, 7, -12, 1, -1, 0, 6, 6, 1, 18, 9}
, {9, 18, 4, 1, -14, -11, -34, -38, 13, -23, -3, 0, -40, -17, -4, 2, -1, -14, 1, -22, 3, -15, 8, -11, -2, -16, -5, -19, -13, -12, -9, -11}
}
, {{-11, -1, -11, -39, -3, -10, -1, -14, -7, 10, -17, 1, 9, -19, 15, -15, 4, 3, -20, 19, 4, -13, -1, -11, -23, -25, 4, 2, -6, 14, -8, -6}
, {-37, 30, 2, -5, -10, -47, -103, 5, 7, 11, 17, 1, -11, 5, 1, -2, 2, -14, -3, -46, -1, -69, -34, -7, 14, -49, -42, 0, -1, 8, -25, 20}
, {-4, 19, 6, -8, -3, -7, -15, -27, 0, -3, -8, 14, 0, -17, -35, 4, -6, -16, 10, -5, -4, -15, 0, -23, -8, 0, -17, -29, -22, -25, -3, 12}
}
, {{0, -25, -11, -34, -58, -6, -15, 0, -39, 19, -6, -1, 4, -2, -3, 8, 18, 7, 21, 8, -1, -2, -7, 15, -15, -28, -27, 4, 10, 4, -25, -5}
, {0, -23, 11, 9, 18, 7, 0, 7, -8, -14, -11, -26, 3, 2, 5, -23, -24, 14, -24, 8, 8, 9, -12, 5, -17, -44, -28, 3, -3, 9, 9, -2}
, {-30, -17, 11, 16, 12, 2, -5, 0, 3, -17, -5, -38, -31, 11, -8, -28, 12, 31, 3, 1, 5, 2, 18, 3, -16, 17, 16, 1, 5, 9, 14, -26}
}
}
, {{{8, -9, -7, -13, -2, 9, 5, -10, -23, -2, -23, -8, 2, 6, -9, 6, -13, 4, 10, -9, -18, -4, -1, -2, -16, 12, -7, -11, -3, 2, -5, 0}
, {-13, 6, -15, 3, -24, 15, 10, 6, -11, 14, 3, 14, -3, -1, 11, 16, 17, -1, 33, 10, 0, -1, -8, -14, 20, 11, 1, -1, -4, 3, -1, 7}
, {21, 3, -1, 2, -3, 16, -2, -2, -6, 2, 6, 19, -21, 10, -3, 17, 10, 2, 15, -9, 5, -1, -15, 0, 21, -6, -3, 10, 11, -4, 3, -5}
}
, {{27, -6, -13, 0, -6, -14, 7, -3, -24, 3, 6, 9, 4, -22, 12, -1, 6, -11, -32, 6, -13, -22, -8, -8, -11, 8, 6, 5, -4, -9, -14, -6}
, {8, -13, -18, -14, -11, -1, -6, 10, -9, 14, 2, 15, 3, 1, 12, -3, -4, -7, 4, -4, 0, 9, -4, 1, 5, 4, -13, -1, -4, -1, -7, 14}
, {14, 2, -2, -20, -7, 14, -8, 4, -2, 18, 11, 0, 1, 5, 23, 12, 20, -2, 10, -9, 13, 16, -4, 1, 9, 21, 13, 17, -2, 10, 19, 10}
}
, {{-14, 18, -27, 13, 0, -24, -17, 6, 2, 13, 5, 25, -9, -21, 25, 26, -6, -48, 11, -35, -5, -32, -69, -30, 20, 8, 4, 11, -25, -25, -9, -12}
, {11, -8, -21, -24, -22, 25, 27, -52, -4, -58, -39, 11, -9, -51, -40, 15, -42, -9, -2, 9, -15, 39, -1, -37, -9, 30, 49, -60, -23, -61, 8, -18}
, {6, -26, -15, -6, -6, 5, 8, -48, -18, -23, -59, 19, -32, -26, -19, 11, -23, -21, -10, -9, -27, 7, -29, -55, -58, -2, 16, -16, -54, -39, 11, -37}
}
}
, {{{-17, 8, 16, 8, -3, -3, 3, 14, 1, 4, 17, -6, 14, 2, -31, 11, -1, -1, 2, -16, -16, -10, -20, 5, 5, 1, -6, -5, 5, -9, -9, -7}
, {22, 0, 4, -13, 5, 23, 10, -49, 13, -47, -23, -32, -19, -13, -37, 3, -25, 2, -10, 2, -12, 14, 6, -20, -57, 18, 9, -38, -5, -4, 1, -20}
, {11, -47, -11, -10, -18, 20, 16, -36, -9, -77, -43, -33, -20, -30, -52, -20, -35, 18, -8, 2, -14, 2, 19, -9, -76, 14, 20, -60, -12, -17, 2, -59}
}
, {{23, -49, 12, -3, 9, 6, 12, 4, 3, -4, -7, -10, 2, 3, -18, 17, -10, 10, 0, 9, -8, 17, 8, -4, 2, 2, 5, -1, -4, -25, 5, -24}
, {13, -23, 0, -4, 6, 1, 19, -15, 9, -6, -23, 7, 4, 0, 1, -13, -9, -7, -28, 15, -6, 13, 15, 2, -39, 21, 23, 4, -2, -5, 15, 3}
, {0, -6, 2, 1, 8, 3, 32, 7, 0, -10, -3, -34, -1, -9, -33, -44, -1, -3, -37, 5, -3, 18, 26, 8, -21, 24, 12, -6, 0, 4, 2, -44}
}
, {{-26, 21, -13, 1, -11, 7, -5, 13, 12, 16, 11, -22, -10, 2, 2, 17, -11, 0, 3, -10, -2, -3, 1, 8, 9, 7, 18, 9, 5, -4, 6, -35}
, {-14, -3, -10, -11, 6, 21, 7, -7, 5, 14, 2, 8, 0, 14, -6, -7, 0, -1, 4, -7, -8, 12, 1, -3, 6, 3, 0, 13, 2, -1, 5, 3}
, {-3, 10, 5, -7, -22, 9, 10, -9, 4, -3, 11, -14, 0, 15, 2, 9, -2, -11, 8, 1, 8, 7, 9, -18, 9, 13, 14, 8, 3, 2, 0, 8}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_2_H_
#define _FLATTEN_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 256

typedef int8_t flatten_2_output_type[OUTPUT_DIM];

#if 0
void flatten_2(
  const number_t input[2][2][64], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_2_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten_2.h"
#include "number.h"
#endif

#define OUTPUT_DIM 256

#define NUMBER_T int8_t
#define LONG_NUMBER_T int32_t

static inline void flatten_2(
  const NUMBER_T input[2][2][64], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_4_H_
#define _DENSE_4_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 256
#define FC_UNITS 128

typedef int8_t dense_4_output_type[FC_UNITS];

#if 0
void dense_4(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_4_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_4.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 256
#define FC_UNITS 128
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int8_t
#define LONG_NUMBER_T int32_t


static inline void dense_4(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q7(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q7(
#endif
                             (q7_t*)input,
                             (q7_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q7_t*)bias,
                             (q7_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 256
#define FC_UNITS 128


const int8_t dense_4_bias[FC_UNITS] = {6, -17, 5, 18, 6, -6, -9, 19, 0, 1, -7, 4, 6, 17, 6, -10, 4, 21, 19, 13, -3, 19, 3, 8, -2, 0, 5, 4, 0, 8, -4, 20, -4, 1, -13, 16, 6, 3, 20, 25, -1, 1, 16, -1, 0, 8, 5, -4, 1, 9, 1, 6, -1, 12, -3, 17, 7, 3, 1, -4, 5, 3, 11, -13, 24, 18, -1, 3, 13, -15, -11, -5, -2, 16, 1, -2, 11, 13, -1, -6, 10, 19, 8, -6, -8, 21, 0, 2, 12, 3, 20, 16, 16, 2, -9, 6, 11, 1, -4, -1, -4, 16, 19, 8, -1, -1, 17, -3, 1, -8, 9, 8, -5, 6, 8, 14, 9, -9, 17, 3, -3, 13, -14, 1, 11, -2, 6, 4}
;

const int8_t dense_4_kernel[FC_UNITS][INPUT_SAMPLES] = {{60, 10, 22, 16, 8, -15, 1, 32, 11, -18, -7, -21, 5, 20, -26, 13, 4, 17, 1, -6, 30, -35, 10, 0, -8, 0, -25, 8, 15, 7, 20, 16, -1, -29, -5, -6, 2, -41, -13, 5, -31, -8, -11, -25, -16, 12, 9, 46, 3, -22, -36, -4, -39, 6, -8, -15, 13, -2, -20, -35, -3, 36, -24, -14, 0, -6, 30, 10, -34, -42, 20, 2, -11, 23, 8, -10, 34, -15, 5, -23, -11, -21, -19, 7, 12, -22, 25, 12, -2, 5, -12, 3, -6, 25, -20, -2, 8, 0, -1, -11, -11, -12, -11, 27, -17, -33, -20, 6, 28, 41, -37, -3, 3, -12, -18, -21, -35, 18, 12, -16, -19, -10, -22, -9, 18, 0, -23, -9, -12, 12, 15, 42, 22, -20, -13, 29, 12, -19, -25, -6, 3, -15, -15, 17, 11, -18, 29, -32, 3, -12, -1, -4, 20, -3, 12, 2, 12, 16, 5, -2, 14, 15, 23, 7, 16, -16, -11, -6, -3, -6, 0, -10, 7, -5, -10, -35, 27, -10, 1, 1, 0, -20, 53, 4, -35, -6, -7, 7, 12, 5, 6, -44, 4, -3, 12, -14, 9, 8, -13, 11, 6, -24, -12, 0, -3, -32, 14, 2, -8, 23, -6, -38, 19, -2, 8, -9, -11, 2, -8, -13, -4, 14, -1, 4, -1, 30, -7, -15, 2, 0, -22, -2, 17, -15, -47, -13, 1, -5, 4, 11, -9, 3, -2, 23, 6, -9, 8, -15, -9, 9, -17, 11, 2, -8, 5, -40}
, {79, 9, -27, 12, -26, 6, 1, 11, 38, -11, 8, -3, -21, -1, -13, 11, -12, 14, 0, -17, 16, -33, -6, -2, -8, -15, -27, 7, 22, 14, -2, 1, -12, 10, -35, 1, -26, 14, -5, 17, -19, 3, 2, -17, -15, -9, -1, 48, 16, 5, 13, 6, -12, 4, -44, -3, 12, 0, -4, 10, -3, 34, 7, 1, 5, -4, 8, 13, -3, 17, -2, 22, 25, 28, 6, -7, 5, -45, 0, -25, -1, -13, -6, -33, -3, 16, 24, -3, 14, -11, 36, 17, -20, 1, -1, 1, 1, -3, 10, -4, -4, -22, -2, 29, -48, -5, -5, -36, 21, 14, 3, 2, 37, -12, 2, -6, -8, 22, 11, -7, 3, 27, -10, 2, 0, -4, -18, 4, 17, -7, -18, 24, 10, -31, 6, 28, 2, 14, -21, -12, -47, 11, -15, 7, -17, -22, 1, -35, 4, 27, 5, -24, -10, -13, 19, -29, 8, 2, -12, -20, -17, -19, -6, 2, -24, 9, 0, 14, 12, -21, -10, -30, -22, 12, -29, -22, -11, -2, -4, 8, 2, -3, 16, -19, -39, -15, -23, 11, -33, -11, -9, 12, -8, -40, 23, 0, -2, -8, -25, 7, 13, -1, -15, 7, 2, -21, 20, -20, -6, -9, 2, -33, 4, -3, 1, 22, -1, -4, -11, 8, -37, -2, -16, 4, 29, -8, 13, -10, 3, 22, -17, 10, -13, -42, -13, 15, -1, 36, -13, -18, 5, 22, -20, -1, -20, -11, 10, 7, -26, -23, -20, 9, 11, 6, 21, -22}
, {25, -5, 1, -1, -11, -35, 9, 22, -24, -31, 3, -17, -12, -9, -59, 7, -19, 21, 26, -20, 16, -37, -27, 12, -22, -20, -7, 19, -11, -6, -19, -4, -3, -5, -9, -22, -12, -37, -39, -9, -32, -6, -5, 4, -42, 3, 9, 17, 13, -30, -8, -16, -55, -9, -28, -14, -15, 19, -28, -25, 9, 4, -29, 0, -7, -4, 15, 1, -34, -22, -1, -1, 1, -4, -22, 4, 7, -16, -14, -43, -3, -5, -15, -8, 37, -8, 16, -1, 3, 22, -19, 2, 12, 11, -10, 10, -1, 7, 9, -14, 1, -26, -15, -13, 10, -6, 6, -25, 18, 37, -28, -2, 19, -22, 9, -7, -12, -19, -2, 21, 8, 15, 6, -16, 9, -15, -15, 13, -19, -16, 4, -7, 9, -32, -22, 26, -19, 4, -10, -4, -21, -18, 8, 14, -12, -22, -33, 2, -1, -4, 13, -26, 8, 1, -34, -3, -5, -6, -25, -26, 18, 34, -1, -2, 10, 11, -25, 16, -26, -33, 11, 12, 2, -1, -7, -49, -11, -4, -18, 28, 1, 7, -1, -2, -33, -1, -16, -7, 21, -19, -19, -9, 10, 10, 13, -10, 6, -12, -31, 11, 12, -11, -18, 23, -30, -14, 11, 10, 14, 0, -11, -31, -2, 5, -1, -49, -1, 14, 23, -19, -4, 5, -2, 0, -14, 10, -21, 9, -15, 8, -28, -4, -9, -49, 8, -19, -24, 14, -27, -10, 0, 13, -2, 31, 17, -14, 10, -28, -26, 18, -52, -12, -13, 12, 8, -33}
, {-7, 12, 1, -9, -22, -5, 5, 3, -1, -13, 9, 12, 14, 7, 6, -11, -12, 12, 5, 2, 3, -10, -8, 4, -1, 4, -34, 28, 15, -11, -8, 8, 18, 4, 2, -9, 8, -4, 4, 7, -6, 3, 12, 10, -6, -17, 5, 39, 12, -11, 5, 18, -2, -6, -33, -7, -4, -12, 7, 7, -7, -5, -9, 8, -23, -8, -33, 21, -12, -2, 21, -25, -14, 12, -18, -14, -29, -3, 17, 0, -17, -19, -33, 11, -17, 2, 13, 2, -26, 4, -17, -9, -6, 1, 5, -1, 6, 11, -12, 1, 18, 15, 19, 11, -1, -7, -4, 4, -11, 5, -12, 0, 33, -27, 15, 6, -18, 1, 2, 5, -4, -10, -8, 13, -5, -1, -3, 14, -4, 71, -4, 9, 4, -1, 5, 2, 1, -25, 23, 52, -14, -34, -4, 11, 6, -9, 1, -36, 21, 31, 2, -4, -11, 1, -30, -29, -1, 22, 14, -28, -14, -6, 2, 25, -14, 19, -14, -1, 40, -25, 7, -4, 10, -12, 6, 5, -6, 44, -8, -26, 9, -24, -4, -19, -9, 5, -9, 26, -34, -23, 9, -1, 8, 25, -4, 32, 19, -2, 6, 11, -4, -39, -34, 12, -12, -24, -13, 25, -4, 4, -6, -13, -8, 9, -5, 4, -19, 13, -12, -20, 23, -11, 3, -27, -2, 25, 14, -23, 6, 8, -21, 11, 20, -9, -18, -7, -36, -14, -8, 17, 3, -8, -34, -2, 19, -41, 2, -7, -17, -33, -4, 24, 22, -22, -18, -25}
, {33, -16, 10, -14, 2, -21, 11, 21, 8, 14, -2, -33, 7, -17, -20, -7, -17, 2, 4, -19, 0, -12, 12, 20, 16, -17, 1, -16, 7, 4, 10, -13, -3, 3, -11, -7, 7, -22, -5, 9, -25, -18, 0, 0, -12, 11, -12, 6, 10, -6, -32, -12, -46, 5, 9, -4, 15, -6, -18, -15, 5, 15, -19, -7, 3, -8, 26, -7, 11, -1, 24, 14, 8, 0, -43, -9, 14, 7, -12, -16, -6, 3, 0, -6, 1, 9, 2, 16, 28, 0, 12, 4, 3, 9, 23, -10, 10, 8, 9, 5, 19, 14, -6, -19, -1, -6, -3, 2, 12, -13, -10, -1, -13, -5, 11, -2, -11, -6, 5, -1, 24, 10, -1, 5, 32, -6, -5, 16, 27, -34, 28, 20, 15, 15, -6, -4, 6, -1, -40, -33, 9, -1, -1, -30, 16, 20, 30, 19, -11, -38, -16, -11, 1, 1, 19, 5, 4, 13, 8, -3, 10, 28, -21, -45, 31, -54, -34, 16, -24, -13, 1, 5, -1, 15, -4, -4, 22, -19, -40, 3, -10, 23, 16, -11, 0, -18, 11, -33, 18, 12, -44, -31, -23, 0, 15, 5, 0, -4, -2, -8, -22, -16, -4, 17, 16, 2, -10, 9, -31, 11, -33, 20, 1, -3, 23, 1, 5, 31, -4, -17, 12, 21, -5, -26, 1, 15, -10, -9, -55, -8, -16, -17, 12, -11, 29, 1, -8, -6, -30, 11, -18, 6, 20, 7, 1, -28, 17, -12, -10, -6, -6, -11, 9, -29, -12, 10}
, {18, 4, -19, -6, -5, 15, 1, -8, 34, -7, 7, -2, -35, -10, -13, -11, -19, 9, -26, -2, 19, -56, -9, -4, -8, -18, 2, -5, 22, 6, 4, -27, -9, 9, -2, 5, -17, 13, -7, -12, -25, 16, 4, -15, -15, 1, 6, -20, 2, 0, 14, 4, -44, 7, -40, -8, 10, 10, -1, 0, -13, -8, -1, 7, -5, -4, 14, 22, 11, 0, -17, 28, 8, 34, 33, -26, -2, -14, 15, -34, 0, -21, -11, -18, -34, 3, 7, 1, 16, 19, 13, 7, -32, -7, -5, 7, -13, 11, -25, -3, 11, -3, 20, 15, -34, -4, -15, -2, -9, 8, -9, -5, 37, -16, 18, -2, -7, -3, 24, 17, -6, 40, -39, 11, 16, 5, -18, -5, -12, 6, -11, 28, 1, -3, -3, 12, -4, 18, -13, 6, -25, -1, -14, 14, 15, -25, 10, -22, 22, 27, -3, 1, -1, -17, 5, -19, 9, 9, 5, -43, -9, -23, 6, -27, -8, 11, 14, 7, -4, -25, 23, -52, -17, 23, -31, -26, -3, -21, -6, 7, -8, -5, -9, -37, -25, -4, -43, -1, -48, -16, -4, -5, 18, -20, 6, -1, 13, 10, -32, 18, 17, -11, -5, -21, 3, -28, 14, -14, 6, -9, -13, -38, 16, 5, 32, -1, -1, -16, -6, 24, -62, 17, -18, -12, -6, -3, 13, -18, -16, 9, -3, 15, -5, -35, -22, -26, -14, -1, -10, -2, 12, 6, -11, -22, -18, 9, -17, -44, -10, 6, -38, -20, -7, 2, 19, -18}
, {-18, 10, 1, -4, -16, 17, -8, 14, -25, -11, 3, 12, 5, -6, 27, 29, -13, -21, -3, -7, -13, 28, 7, -19, -13, -10, 35, -26, -28, -20, 0, -15, 2, -6, 17, 7, -27, -10, 5, 2, 1, -3, 2, -13, 21, -21, -22, -19, -8, 0, 8, -3, 11, -29, 3, -11, 17, 4, -1, -15, 14, -24, 0, -13, 33, 2, -8, 5, -8, 0, -15, 3, -4, -32, -4, 0, -5, 8, -4, 27, -2, 5, 12, 11, 27, 29, -22, -27, 2, 7, -20, 17, 4, 5, 19, -15, -9, -14, 23, 20, -3, 2, -21, -22, 12, -13, -10, -32, -8, -31, -2, 8, -2, -9, -15, -8, 8, -15, -13, -5, 18, -12, -5, -18, -34, -22, 16, 3, -21, -37, -2, -22, -8, -30, 15, -10, -23, 9, -24, -23, 15, 12, 4, -6, -1, -30, -30, 13, -32, -14, 25, 21, -1, -7, -10, 7, 2, 11, 14, 5, 13, 17, 3, -31, 24, -16, 15, 5, 0, 14, 14, -1, -18, 1, -21, -13, -21, -39, 11, 27, -35, -15, 2, 10, -25, -38, -20, -8, 10, -21, -51, -2, -6, -47, -14, 0, -12, 6, -3, -35, 0, 8, -19, -14, 28, 38, -4, -31, -15, -13, 12, 5, -12, -33, -11, 20, 13, -24, 1, -16, 9, -1, -27, -18, 7, -15, -1, 4, 20, -28, -3, -11, -16, 21, 9, 13, 6, 19, -11, -11, -32, -1, 17, 12, -2, 25, -19, -9, 1, 4, 7, -16, 9, 27, -10, 3}
, {-3, -23, -8, 9, -12, -8, -15, 3, -34, -13, -27, -27, 20, 3, 17, 6, -4, 10, 8, 25, -4, 4, -4, 12, 11, 17, -12, -2, -12, -10, -17, 31, -17, -8, 2, -34, 27, -42, 11, 5, -1, -17, 7, 6, -1, 6, 0, 1, -9, -26, -8, 29, -13, -14, 17, -1, -30, -18, 18, -13, 2, 27, -44, 5, -39, 1, -4, -22, 15, 4, -4, -24, -16, 6, -24, 7, -12, -11, 9, -19, -3, -8, -32, 12, -24, -45, 10, 9, -12, -1, -4, -27, -17, 27, -4, -9, 4, 13, -17, -23, -18, 18, -7, -25, 4, -6, 4, -8, 16, -11, -7, -17, -18, 5, 26, 6, 4, -17, -7, -5, 0, -9, 7, -11, 5, -22, 26, 9, -19, 41, 12, 0, -6, -3, -11, -19, -12, -54, 9, 22, -2, -42, 17, 15, 4, -23, -21, -18, -1, 21, -6, 0, 17, -17, -17, -6, -26, 33, -32, -21, -4, 16, 17, -2, 7, 1, -28, 2, 18, 2, -1, 8, 0, -4, 16, -13, -5, 19, 15, -7, 11, -23, 17, -7, 5, -1, 18, 27, -4, -72, 2, -22, 13, 10, -45, 11, -8, -1, -10, 6, -7, -12, -1, -16, -34, 5, -37, -16, -7, 5, 6, -13, 6, -6, -16, -7, -5, 24, 11, 1, 23, -17, 4, -5, -18, 3, 5, 14, -4, -17, -31, -15, -2, 2, -7, -12, -17, -22, -19, 18, 21, 15, -12, -1, 23, -20, 23, -21, -8, 0, -12, 6, -27, 9, -6, -5}
, {45, 15, -3, 34, 58, -9, 12, 5, -18, -9, 24, 24, -4, -4, -23, 35, 24, -7, -14, -4, 9, -7, 3, -53, -14, -14, 40, -10, -5, -5, 18, -4, 1, -26, 5, 4, -14, 8, -29, -13, 56, -4, -33, -47, 3, -9, -9, 30, -16, -24, -29, -29, 1, 4, -21, -25, 3, -9, -10, -21, -31, -4, 5, -14, 25, -5, -5, 31, 5, 16, -8, 5, -4, -10, -11, -19, 10, 12, 8, 8, 22, 19, 3, -27, 21, 25, 21, -9, 14, -2, 6, 23, -13, 28, 18, 8, -13, 1, 19, -3, 2, -1, -5, -4, -22, -14, -25, -12, 6, 12, 1, 7, 17, -34, -22, -9, 13, 16, 8, -16, 0, 21, -6, -35, -11, 21, 9, -28, 5, 0, 7, 14, 12, 0, 8, -16, -3, -14, -16, 20, 17, 1, -18, 16, -3, 1, -10, 0, 3, -13, 4, 20, 15, 1, 10, -14, 4, -3, 3, 13, 4, 0, 12, 14, 22, 14, -4, 3, 1, 35, -4, 3, -21, -2, 13, 4, -11, 11, 7, 4, 9, 8, 6, -6, -14, -32, -11, 4, 17, 6, 2, -5, -25, -7, -1, 2, -20, -35, -4, -16, -6, -27, 20, -8, 10, 2, 30, -3, -22, -15, -9, 12, -5, -2, -4, 8, 14, 17, -15, -15, 13, -3, -6, -20, 0, 1, 21, 16, 1, 1, -11, -2, -11, 11, 9, 18, -3, -11, 25, 4, 9, -5, 25, -8, -4, -46, 4, 34, -9, -8, 36, 10, -4, -9, 1, 9}
, {33, 6, 3, 39, 20, -13, 14, -8, 49, 6, 9, -19, 7, -15, -7, -10, 1, 6, -14, -10, -1, 22, -2, -6, 2, 9, -5, 3, 22, 4, 27, -5, 0, 18, 11, -17, 2, -14, 10, 15, -27, 17, -4, -3, 4, 14, -5, 56, 21, 5, -7, -6, -4, 31, 14, -3, 19, -20, -12, -2, 20, 27, -12, 12, -12, -21, 6, -19, -3, 13, 4, -16, -3, 20, 5, -7, 57, -4, 10, -38, -1, 8, 7, -23, 12, -2, 13, -2, 10, -13, 4, 24, -4, 26, -27, -11, -23, 3, 20, 16, -9, -2, 5, -1, -13, -1, -9, -2, 4, 33, -7, -4, 20, -7, -7, -8, -14, 11, 18, -1, -7, 18, -8, -13, -10, -1, 11, 18, 43, -8, -5, 39, 23, -3, -11, 22, -8, 3, -62, -12, -8, -15, -26, 1, 17, 20, 10, 10, 8, 16, -7, 5, -30, 10, 8, 12, -7, -24, 10, -6, -4, 4, -31, -2, 8, -23, -11, 2, -7, 18, -15, 1, -20, -8, -1, 8, -4, -12, -8, -38, -16, 14, 8, -4, -9, 10, -1, 6, 29, 20, -4, -26, -38, -13, 5, -4, -26, -21, 5, -5, 9, -4, 8, -4, -5, -3, 28, 3, -20, 0, -36, 5, 5, 23, 16, 7, 8, 8, -26, -24, 0, 20, -43, -30, 7, 41, 13, -8, -10, 8, 1, 15, -3, -7, 8, -4, -2, -3, -15, -40, 9, 1, -2, -16, -13, -1, 0, 12, 3, -14, 21, 22, 13, -19, -9, -14}
, {-7, -1, 17, -5, -1, -1, -16, -2, -1, 32, 4, -9, 8, -9, 3, -2, 4, -26, -16, 7, -6, 18, 10, -1, 5, 0, 14, -9, -10, 12, 30, -21, -3, -12, -7, 10, -16, 11, 18, -12, 10, 3, -3, 8, 2, 2, -12, -9, -5, 6, 5, -11, 11, 1, -2, -4, 4, -13, 2, 7, -1, -24, 11, 3, 32, 10, 32, 1, 6, -1, 6, 32, 18, -6, -15, -4, 19, 22, -20, 9, 8, 3, 42, -4, 30, 38, 18, -10, 4, -15, 5, -26, 9, 13, 8, -19, -13, -2, 20, 17, -10, 6, -8, -23, 3, 0, -11, 4, -38, -22, -13, 0, -16, -22, -19, -20, 11, 0, 5, 6, 16, -25, -12, -10, -2, -12, 11, -16, -9, -31, 11, -16, 2, -3, 16, -6, 14, 14, -26, -25, 18, 9, -8, -9, 4, -17, -13, 32, -26, -26, 16, 2, 19, 4, 12, 0, -23, -1, 6, 19, 8, 20, -2, -33, 22, -14, -5, 12, -6, 15, 5, 14, 2, 3, 5, -3, -5, -27, -28, 15, -7, 21, -4, -10, -8, -7, 12, -28, 23, 26, -27, 7, -24, -42, 20, -4, -11, -2, -11, -40, -1, 38, -8, -28, 10, 23, 5, -34, -14, -23, 20, -3, -3, -43, 23, 15, 15, -27, -7, -6, 9, -2, -4, -9, 8, -2, -18, -7, 15, -36, 11, 0, -32, 16, 9, 15, 10, 42, 5, -12, -8, -31, 7, -19, -32, 17, -18, 12, 15, 23, -8, -31, 0, 28, -17, -5}
, {22, 25, -23, 5, 43, 10, -9, -2, 12, -31, 24, 25, -7, -6, 5, -1, -24, -12, 12, 13, 13, -13, -10, -5, -30, 4, 1, 10, 17, -17, -31, -16, -17, -44, -36, 57, 3, 19, 12, -18, 15, 41, -3, -28, -12, -17, 39, 23, -20, -4, 11, -5, 20, -23, -16, 19, 36, 4, 5, 17, -33, 6, 22, 6, -7, -10, -47, 32, -25, -14, 1, -27, -19, -12, 18, -13, 5, 24, -14, 15, -1, -3, -19, 8, 42, -24, 12, -12, -22, 4, -31, 30, -9, 2, -14, 16, 9, -10, -8, 2, -4, -19, -4, 10, -6, -6, 6, 3, 21, 19, 6, 5, 25, -18, -16, -6, -30, -3, -2, -9, -2, 13, 7, -6, -8, 0, -17, -11, -11, -10, 9, 14, 2, 2, 38, -17, -13, -7, -14, 9, 0, 6, 13, -12, -30, -37, 7, -21, -6, 22, -6, 9, 12, 19, 8, 6, -20, -5, -8, -1, 8, 16, 16, -8, 12, 20, 18, -9, 4, -9, 22, 30, -2, -29, -16, -7, -11, -1, 3, 36, 7, -10, 25, 4, 1, -11, 13, -11, 20, 14, 5, 0, 4, 7, -19, 16, 5, -6, 13, -2, 10, -46, 12, 36, -15, 3, -5, -9, 2, 12, 3, -14, -10, 3, -24, 8, 16, 16, -4, -13, 15, -3, 19, -16, 12, 21, -2, -2, 2, -7, -21, -9, 4, -8, -10, 11, -22, -11, -3, 14, -6, 36, 4, 6, 11, -32, 24, 8, 7, -11, 4, 27, 12, -12, -12, 17}
, {19, -25, 24, 10, 1, -10, -5, 44, 16, -4, 25, -13, 14, -10, 2, 12, -14, 9, 4, -6, -1, -3, 11, -4, 14, -1, 37, -9, -21, 2, 5, -21, -19, 4, 17, -36, 2, -6, 9, 1, 13, 11, -11, 0, -2, 11, -20, -36, 6, 0, -22, -5, 19, 5, 5, -21, -13, -3, -13, -38, 4, -11, -3, -9, 19, -9, 26, -2, 2, -8, -6, 24, 9, 27, 5, 5, 23, -16, -18, 5, 7, -14, -1, 4, 12, 36, 38, -19, 24, 11, 18, 20, -2, 17, 14, 13, -10, -7, 14, -7, 3, -12, -34, -13, -4, -28, -12, -33, 23, 10, -22, 9, 11, -16, -21, -20, -5, 19, -3, -18, 13, -22, 0, -19, 8, 10, -7, -22, -27, 25, 10, -5, -2, -28, 20, -2, -18, -10, -9, 20, -11, 5, 16, 6, -5, 7, -17, 15, -2, -27, 25, 18, 28, -33, -1, -14, 15, 12, 3, -17, 32, 6, 5, -3, 11, -9, -17, 15, 0, 13, -17, 16, 4, -6, -2, -27, 15, -50, -3, 10, -20, 10, 10, 1, -28, -26, -18, 21, 37, -10, -30, 2, -4, 1, -15, -4, 12, 20, -5, -25, 5, 10, -16, 5, 6, 10, -5, 0, -12, 22, -20, 29, -40, -28, -4, 21, -14, 18, -4, 8, 11, -23, -1, -9, -17, 12, -25, -4, -5, -42, 10, -11, -2, 1, -3, 17, -25, -15, 13, 38, -27, -4, 11, 19, 16, 13, 3, 16, 21, 15, 17, -15, -1, 19, -22, -29}
, {16, -29, 11, -4, 1, -3, 0, 12, 21, 3, -13, -34, 13, -6, 7, 0, 1, 0, -1, -13, 17, -24, 0, 18, 5, 5, 16, 10, -2, -13, 31, 21, -8, -17, 8, -22, 14, -33, -4, 14, -22, -13, -6, -2, 5, 24, 2, -12, 14, -13, -18, 2, -17, -7, 0, 7, 2, 4, 6, -13, 18, 26, -13, -13, -11, -7, 37, -8, 11, 9, 0, 1, 11, 10, 1, 3, 24, 3, 28, -12, -16, 3, 0, -1, -25, -18, -4, 5, 20, 19, 11, -12, -12, -3, -5, -10, -2, 0, 10, 12, -9, 24, 1, -3, 11, 7, 12, 0, -2, -6, -13, -2, -32, 23, -7, -8, 11, 7, 16, 10, 9, 4, 11, 19, 7, -8, 20, 11, 6, -4, 11, 34, 4, 4, 8, -29, -8, -24, -26, -4, -8, -13, -13, 18, 8, 2, 23, -20, 14, -32, -2, -5, 15, -23, 3, -15, 8, 24, -6, 15, -1, -1, 26, -11, 19, -44, -48, 27, -5, -8, -39, -3, 6, 22, 13, -22, 33, -22, -38, -8, -7, 2, 5, -6, -16, -11, 5, -12, -27, -2, -32, 6, 8, 10, -5, -15, -7, -20, 7, -11, -4, -36, 13, 8, 6, -19, -39, 0, -22, 13, -34, 17, 23, -56, 17, -15, 3, 13, -12, -3, 15, 34, 19, -18, -32, 18, -15, -29, -30, -43, -38, -31, 13, -17, 11, -13, -1, -11, 14, 15, -24, -12, 1, 12, 1, -19, -3, -21, 7, -3, -1, -23, 2, 3, -49, 23}
, {-31, -14, 11, -18, -36, 4, -3, -23, 10, -3, -1, -12, -1, -9, -4, 7, -26, -7, 3, -26, 5, -31, -4, 10, 6, -5, 1, 2, 10, 7, 21, -15, 6, 18, 5, -17, -3, -22, 5, 13, -18, 11, 8, 13, -14, -5, -8, -15, 11, -10, -11, 9, -4, -1, -14, 6, -14, 15, -6, -5, 12, -31, -2, 18, -2, 15, -10, -7, -7, 3, 25, 26, 1, -46, 16, 10, -11, -11, -3, -14, 1, -2, 8, 1, -33, -12, -56, 1, -11, 4, -24, -8, 2, -27, 8, -2, 5, 12, -11, 16, 16, -28, 2, 3, 18, -3, 8, -13, -49, -34, 2, 7, -11, -19, 0, 5, 5, -44, 6, 7, 10, -18, -2, -1, 21, -5, -23, 5, 5, 3, 15, 6, 11, -17, -7, 29, 18, -23, -48, -2, -8, 7, -7, 17, 17, 8, 1, -15, 9, 1, 16, -29, 9, 11, -17, -5, 2, 14, 2, -7, 6, -1, 1, 5, 7, -8, -21, 16, 10, -26, -6, -2, -11, 17, 10, -5, 2, 0, -13, 8, 0, 20, -1, 2, -32, -26, -12, -4, 6, 4, -7, -14, -6, -32, 31, -3, 8, -7, -54, 12, -2, -16, -23, -2, -5, -6, 13, -21, -3, -3, 3, -28, 8, 11, 7, -20, 7, -8, -1, -20, -21, 10, -25, 20, 0, -4, -19, -20, -13, 8, -20, 18, -31, -38, -28, -7, 3, 43, -4, -25, 5, 1, 34, -29, -1, -6, 2, -23, -33, 4, -46, -11, -36, 9, 25, -60}
, {16, 1, 3, -3, 10, -10, 7, 18, -4, -15, -20, -1, -12, -3, 3, -8, -3, 0, 22, 5, -6, 4, 0, 6, 10, -19, 45, -8, -16, 3, 25, -25, 13, -10, 5, 2, -16, 8, -11, -26, 31, -3, 1, -5, -13, -12, 16, -25, 0, 1, 0, -2, 9, -8, 22, 3, -26, -15, -1, 15, -6, -25, 0, -20, 28, -8, 12, 14, 3, -5, -32, 15, 9, -3, -1, 14, 5, 9, -39, -2, -5, -11, -2, 7, 32, 11, 8, -20, 6, 5, -2, 10, -7, -15, 4, 19, -4, -22, -25, 2, -2, 6, -16, -22, 5, -20, -19, -25, 6, 15, -3, 2, 13, -3, -13, -43, 8, -11, -6, -15, 18, 3, 4, -3, 3, -5, -28, -18, -34, -5, 12, -14, -8, -44, 1, 32, -41, 31, 2, -9, -11, -21, -18, 18, 6, -24, -28, -22, -1, -5, 0, 17, 28, -45, 3, 27, -1, 19, 17, -12, 12, 5, 19, -47, 24, -32, -8, 11, 2, 10, 5, 6, -11, -2, -15, -16, 27, -38, -24, 31, 11, 15, -20, -27, -26, 16, -24, -19, 43, -50, -34, 24, 12, -22, 16, -13, 0, 26, -22, -6, -2, 10, -8, -8, 0, 11, 4, -15, -6, -3, -7, 15, -8, -43, 0, -7, -5, -5, 15, 29, -27, -9, -23, 8, -13, -28, -26, -29, 18, -10, 38, 13, -17, -10, -6, 14, -2, 10, -16, -8, -1, 10, 23, 3, 4, 10, -2, 3, 4, 33, -15, -35, 14, 1, 2, -8}
, {20, -25, 7, 37, 10, -12, 2, 22, 30, 29, -20, -15, -4, -30, -19, 15, 23, -22, -33, -7, -6, 44, 12, -6, 0, 12, -6, -14, -16, 10, 7, -9, 19, -7, 7, -26, 15, 43, -35, 18, -31, -24, -30, 16, 3, 10, -37, -20, 4, 30, -23, -9, -1, 23, 14, -13, -13, -54, -21, -25, 15, 18, -14, -6, 1, -2, 44, -29, 9, 13, -13, 0, -1, 12, -3, -7, 16, 19, 14, -4, -9, -7, 8, -7, -24, 0, -16, -22, 29, -1, 4, 19, 3, -6, -12, 3, -14, 12, 25, 10, 7, 18, -23, -13, 1, -27, -21, -9, 4, -13, -9, -11, -33, 4, -7, -24, 6, 23, -3, -18, 10, 8, 8, 5, 9, 6, 13, 1, 15, -44, 7, -7, 13, -7, 16, -11, 0, -1, -64, -51, 2, 3, 8, -35, 6, -2, 9, 31, 2, -29, -1, -8, -3, -32, 27, -6, 7, -34, -5, 29, -5, 2, -53, -31, -2, -9, -12, 10, -38, -27, -42, 1, -34, 13, -1, 3, -1, -40, -9, -42, -49, 21, 6, 14, -13, -19, -14, -30, 1, 24, -25, -21, -18, -4, 40, -2, -37, -33, -2, -8, -6, -14, -7, -4, 17, 8, -2, -11, -7, -2, -9, 16, 4, -28, 8, 2, 2, 2, -35, -1, 3, -7, -8, -31, 6, 22, -6, 2, -21, -7, -10, 14, 4, 8, 15, 19, 6, 12, 14, 5, 23, -34, 15, 21, -25, -13, -11, -7, -14, -5, -1, -10, 12, -12, -44, 11}
, {-24, -13, 8, -5, -1, -9, -7, -8, -30, 16, -11, -32, 29, -28, 26, 21, -16, -16, 0, -15, -32, -29, 21, 12, 10, 5, -21, 5, -27, 11, 7, -5, 0, 7, 29, -60, 22, -21, 5, -13, 1, 11, 11, 10, -14, 2, -23, 9, 2, 0, -16, 6, -22, 5, 11, -5, -5, -29, 6, 5, 0, 7, -23, 3, 7, 21, -10, -2, 0, -29, 13, -10, -21, -20, -17, -12, 18, 4, 10, -2, 1, 22, 18, -20, 26, 0, 1, -19, -8, -2, 6, -4, 5, 28, -5, -11, -17, 0, -8, -7, -6, -28, -1, -8, -7, -4, 1, -6, -6, 16, 17, 3, 10, -57, -8, -8, -25, -20, 6, -5, 8, -34, -18, -11, -16, -10, -20, -4, -25, 27, 0, -33, -14, 4, 0, -29, 14, -61, 8, 7, -16, 2, 5, 4, 6, 19, -1, -7, 16, -30, -30, -16, -12, -16, -44, -36, 2, 12, 41, 9, -3, 8, -9, 13, -29, 0, -43, -4, 11, 23, -48, 15, 4, -31, 14, 2, 3, -1, 18, -59, -6, -30, -2, 3, -13, -4, 12, 14, 0, -5, -9, 19, 5, 4, -49, 8, -2, 15, -12, -8, -10, 3, -27, 0, -10, 9, -13, 5, -2, 30, 17, -39, -8, -21, -12, -3, -10, 11, -2, -40, 6, -21, -11, -5, -12, 25, -31, -14, 31, 6, 3, 19, 1, -32, -19, -4, -40, -25, -4, -3, 13, -3, -7, -1, 16, -12, 16, -4, -2, 0, -8, -1, -44, 11, -8, -67}
, {15, -42, 4, -5, -27, -5, -7, 33, -27, 10, -19, -38, 1, -7, 3, 25, 4, -20, 4, -1, -10, -15, 19, 7, 2, -2, -5, -12, -22, 4, -4, -33, 2, -2, 32, -24, 20, -15, -21, 20, -27, -6, -28, 18, -11, 8, -25, -40, -13, -40, -21, -11, -26, -12, 5, -9, -9, -2, -26, -21, 5, -6, -20, -15, 12, 10, 36, 2, 17, 11, 4, 16, -31, 4, -5, -9, 8, -22, -27, -7, 14, -15, -14, -4, 21, 25, 2, -42, 5, 4, 3, 0, 5, -8, 10, 4, 0, 1, 34, -12, -2, -21, -25, 9, -8, 15, 0, -17, 23, -14, -1, -2, 4, -10, -6, -10, -4, -11, -5, -5, -2, 8, 17, -24, 17, -2, 16, -2, -10, -14, 14, -41, -20, -27, -4, -5, 40, -39, 12, 1, -4, -1, 3, 8, 0, 13, 36, -31, 22, -17, 23, -6, -7, -15, -49, -29, 17, 6, 4, 17, -6, 2, 7, 9, 0, -5, -23, 5, -22, -29, -31, 0, 6, -1, 17, -21, 5, 4, -20, -5, -27, -30, -29, 1, -36, 1, -22, 7, 4, 23, -10, -20, -5, 12, -28, -6, 3, 11, -16, 18, -3, -13, -7, 14, 10, 0, -50, 8, 17, 14, -5, -14, -17, -1, 3, -33, -38, 27, 12, -20, 15, -21, 1, -6, 6, 27, -30, -14, 18, 11, -52, -7, 16, -19, -5, -29, -46, -7, -9, 7, 8, -12, -2, -5, 7, -17, 24, -30, -7, -21, -20, -3, -13, 4, -11, -26}
, {4, 7, 12, 9, 15, 18, 0, -30, -11, 7, 17, 19, 4, -11, -2, -15, -30, -10, 1, 4, -5, -8, 3, 10, 4, 11, -23, 16, -8, 10, -3, -6, 0, -14, -3, -3, 20, 3, -9, 5, 13, 7, 10, 7, 13, -4, -9, 10, -10, 12, 5, 0, 17, 4, -7, -7, 10, 18, 10, -6, -5, 2, 21, -1, -8, 14, -17, 37, -6, -21, 11, -7, -23, -39, -17, -9, 6, -3, 4, 25, -11, -9, -4, 6, 24, -26, -15, -19, -37, -8, -31, -4, 17, 4, 20, 5, 9, 10, 9, -5, 0, -34, -1, 20, 0, 16, 2, -23, 10, -2, 19, 6, 4, -45, -9, 14, -15, -4, 10, -13, -10, -41, 6, 2, -13, -15, -20, 16, -17, 15, 9, -23, -13, -11, 33, -20, -22, 2, 2, -5, -26, -16, 17, 31, -22, 3, 12, -4, -2, -38, -10, 2, 8, -15, -28, -9, -5, 18, -7, -13, 19, 4, 28, -15, -20, -5, -3, -19, 6, -3, 7, 1, 20, -38, -1, 3, -4, -9, 1, 5, -19, -25, -13, -1, -3, 9, 0, 12, -22, 2, -9, 4, -20, 1, 8, -2, 17, 10, -20, -11, 21, -33, -20, -8, 16, 6, 10, 12, 9, 12, 6, -5, -43, -5, -32, -6, -13, 28, -15, -33, 4, -31, -14, -28, 16, 11, -10, -2, 28, 7, 8, 5, -4, -15, -24, 38, -40, -44, -3, -13, 10, 22, -17, -1, 28, -24, 40, 12, -7, -31, 8, 33, 19, 8, -2, -49}
, {-14, -8, -25, -7, -29, 8, 17, 12, -2, -3, -7, -5, 6, 6, -20, -7, 6, 11, -10, -2, 12, 18, 11, 8, 13, -1, 18, 10, 8, -11, 3, -10, -9, 16, 6, -2, -1, 2, -19, -9, -13, -11, 5, 6, -3, 23, -4, -3, 17, 2, -3, 2, 13, -4, 9, 1, -20, 4, -7, -9, -5, 10, -1, -12, -23, -13, 13, 13, -13, 40, -20, 3, 7, -1, -2, 3, -3, 0, 16, -21, -2, 0, -7, -6, -21, -10, -16, 21, -33, 31, 27, -9, 6, -13, -29, 13, -16, -5, -7, -7, -10, -27, 24, 12, 12, -10, 8, 10, -28, -21, 0, -22, 10, 11, 6, -8, 18, -9, 1, 7, 2, 20, -2, 11, -4, -3, -2, 9, -26, 9, 17, -22, -10, 9, -18, 4, -19, 11, 29, 25, -34, -31, 31, 18, -27, -30, -31, -25, 0, 11, 16, 6, 16, -17, 1, -5, -4, 11, -4, -46, 19, 19, 46, -18, -18, -17, 36, -16, 10, 39, 20, -4, -19, 18, -17, -20, -11, -17, -6, 70, 10, -15, -26, -11, -5, 5, -13, 1, 11, -42, -14, 10, 13, -16, 0, 3, -3, 18, -12, 10, 18, 3, 7, -2, 22, -17, 0, -22, 8, -15, 14, -9, -7, -19, -7, 8, -7, -27, 10, 33, -53, 6, -2, 5, -21, -18, -11, -6, 9, -15, -2, -3, -5, 5, -17, 9, -7, -21, 15, -5, -8, 27, -9, 11, -16, 28, -14, -11, 6, -2, -24, -15, 11, 2, 27, -23}
, {1, -18, 24, 7, 7, -24, -6, -5, -39, 11, -8, -2, -44, -11, 2, -5, -31, 16, 0, -14, -14, -31, -3, -15, -12, 22, -22, -26, 4, -2, 3, -22, 15, 0, 19, -15, 13, -27, -7, -12, 4, -9, -8, -13, -7, -1, 4, -31, 6, -31, 3, 0, -22, 3, 3, -20, -1, 12, -6, -11, -10, -13, -31, -2, -3, 6, 17, 17, 12, 12, 8, -8, 2, -9, -2, -20, -14, 18, -3, -15, 9, -3, -6, -7, 23, -20, -13, 2, -11, 4, -18, 7, -5, 0, -3, 13, 18, 10, -23, -10, -1, -12, -20, -17, -14, -1, 0, 17, 7, 3, -10, -6, 10, -20, 4, -13, -17, 6, 7, -6, -6, -14, -11, 2, 15, 29, 6, -20, 16, -9, 10, 40, 7, 2, -10, -10, -7, -8, 2, -29, 3, -28, -2, -7, 15, 18, -13, 2, 4, -47, -8, -6, 9, -5, -9, -18, -20, 11, 2, -1, 11, 17, 4, -31, 23, -42, -11, 11, -18, 5, -1, -3, -15, -7, 5, 11, 13, -16, -30, 18, -24, 20, 5, -6, -4, -20, 9, -26, 3, 6, -23, -17, 13, 18, 11, 11, -4, -23, 12, 12, -22, -41, 23, 25, -2, -54, 11, 34, 3, -4, -35, 40, 10, 8, 15, -36, 5, 2, 7, -11, 18, 34, 23, -3, 3, 23, 3, -3, -35, -6, -16, -25, 27, 7, 38, 0, 5, 17, -3, 8, -14, 1, 19, -3, 17, -28, -14, -26, -11, -20, -1, -14, 3, -13, -24, 27}
, {-43, -28, -8, 2, -7, 6, -22, 2, -40, 0, -14, -3, 13, 17, 23, 20, 0, -8, 5, 22, -15, 25, 4, 15, -2, 18, 29, -2, -27, -9, -7, -6, -15, -8, -5, -21, 0, 4, -10, 2, 26, 8, 0, 14, -7, 4, 3, -31, -16, 0, -7, 12, -9, -10, 26, 1, -32, 1, 0, 22, -7, -24, 2, 12, -6, 22, -17, -18, 4, 1, -2, -37, -53, -40, 23, -3, -31, 14, 4, 13, -10, 20, -5, 31, -18, -11, -55, -3, -33, -1, -33, -28, 15, -48, 15, -47, -9, 0, -15, -4, -8, 10, 20, -19, 11, 9, 18, 17, -53, -62, 2, 3, -53, -2, 0, 14, 7, -58, -7, 5, 1, -31, 14, -2, -9, -15, 2, 7, -17, -16, 8, -37, -58, -6, -6, -22, -4, -2, 8, 3, -12, 1, 11, -10, -46, -2, -21, 28, 2, -6, 11, 24, -7, -15, -34, 2, -3, -3, 10, -18, -7, 9, 8, 0, -26, 6, 16, -21, -7, 18, 9, 9, -31, 0, -5, 24, -42, 2, 13, 17, 6, -18, -23, 7, 5, 9, 12, -3, 21, -23, -1, -1, 11, 1, -15, 26, -12, 15, -5, -8, 8, 21, -26, -38, 16, 33, -18, -6, 6, -5, 31, -14, -3, 5, -16, -10, 33, -18, 17, -11, 11, -22, -13, -6, 10, -22, -16, -20, 12, -1, 12, 2, -25, 10, 2, 6, -7, -7, -12, -7, -3, 5, -6, -15, 16, 4, 6, -18, -3, -24, -22, -7, -14, 4, 7, -4}
, {-18, -10, 21, -14, -24, -12, 1, 10, -20, 16, -8, -15, 14, 6, 12, 8, -15, -7, 16, -8, 5, 16, 0, 13, 21, -10, 32, -3, -27, 5, -9, -5, -9, 19, -9, -11, -3, -13, 5, 6, 4, 4, -12, 10, 1, 11, -10, -44, 6, -18, 10, -1, 6, -5, 7, -13, -18, 10, -14, -13, 3, -10, 0, -8, 14, 9, 15, 17, -17, -11, 9, 15, 9, -22, 15, -2, 4, 4, -34, 19, 11, 21, -10, -6, 10, -7, -14, -4, -9, 21, -12, 1, 8, 5, 1, 8, 7, -2, -32, 1, -28, -19, -4, -29, 7, -9, 7, -24, 6, -5, -17, 12, -7, -28, -6, -32, -25, -17, 3, 0, 7, -20, 13, -6, 4, 7, -41, -13, -49, 16, 2, -10, -30, -44, -5, 5, -25, -1, -15, 17, -12, -13, -11, -2, 9, -5, 5, 20, 7, -24, 5, 11, 18, -14, -15, 10, 4, 30, 25, -14, 31, -2, 29, 5, 20, -23, -15, 3, 11, 25, -7, 8, -21, -7, -1, -32, 13, -53, -2, 38, -18, 7, -9, -1, -18, -24, -22, 10, 26, -38, -22, 1, 5, 5, 39, -12, 18, 15, -25, -8, 0, 8, -13, 2, 14, 3, -16, -36, -4, 24, -13, 0, -13, -49, 5, -20, -34, 7, -2, 1, -5, 6, 6, -15, -33, 7, -43, -8, -3, -36, 22, -14, 6, -9, -23, -4, -36, 0, -1, 5, -41, -7, -8, -19, -12, 5, -8, -17, 19, 10, 5, -5, -5, 14, -27, -51}
, {-10, 1, 7, 17, -5, -16, 12, 3, -4, 8, 12, -43, -8, 11, 1, 17, -6, 4, 15, -3, 14, 4, 12, 8, 5, 3, -16, 17, -10, 5, 18, 16, -6, -4, 21, -53, 4, -15, -1, 4, -54, 11, 11, -1, 17, 18, -1, 22, -6, -1, -5, 3, -33, 17, 19, -2, 12, 7, -6, 0, 15, 0, -37, 7, 2, 3, 27, -21, 14, 7, -19, 0, 3, 0, -1, -8, 24, -5, 8, -39, -28, 12, -3, -1, -14, -23, 11, 3, -16, 7, 10, 8, -7, -3, -9, 11, -35, -4, -6, -13, 21, 6, -4, 13, 3, -1, 1, -5, -6, -30, -4, 2, -11, 11, 18, 15, 21, -5, -18, -4, 6, -11, -7, 4, -3, -31, 17, 15, -16, 2, -21, -8, 0, 0, -26, 26, 6, -9, 20, 14, -16, 14, -15, -3, -19, 2, -13, -7, -9, 48, -12, 3, -54, -15, -8, -25, -6, -35, 18, -12, -25, -24, -13, -9, -43, 16, -6, -14, 10, -14, -36, -3, 2, 11, -11, -12, -19, 5, 4, -33, 22, -34, -18, 22, 1, 20, 1, 21, -19, -9, 2, -19, 39, 1, 12, 12, -16, 6, -16, 7, 30, 5, -11, 0, 5, 6, 1, -9, 16, -12, 27, -45, 6, 16, -1, 8, 5, -18, 5, 18, 0, -12, -10, 19, -12, -38, 13, -8, 27, 8, 4, 32, 2, -11, -17, -19, 6, -14, -7, -23, 32, -14, -28, 2, 6, -2, 1, -25, -50, -13, -20, -12, -28, -2, 3, 5}
, {19, -4, 3, 22, -3, -11, 2, -2, 2, 4, 18, -9, 1, -10, 3, 3, -1, -9, -9, 2, -16, -9, 1, 10, 5, -19, -20, 7, -6, 8, 10, 4, -5, -22, 13, -3, 17, -20, -13, -2, -21, 6, -3, -15, 7, 6, 2, 6, -1, 5, 1, 16, -21, 3, -8, 8, 12, 3, -1, -1, -7, -12, 14, 8, 11, 3, 0, 0, 14, -9, -13, -6, -12, 2, -10, -16, 24, -38, -32, 17, -7, -2, -33, -22, 7, -22, 9, -11, -2, -20, 8, 3, -15, 20, -6, -27, -19, 18, -12, -13, 4, -27, -31, 19, -12, 5, -1, 7, -3, 15, -5, 1, -8, -20, -15, 11, 2, 17, -10, -12, 3, -20, -4, -25, -2, -13, 3, -7, -11, 7, -28, -4, -13, -4, 1, -21, 14, -39, -39, 9, -11, 12, 3, 4, -6, 5, 4, 19, 19, -4, -22, -18, -24, -11, -34, -17, 16, -13, 35, -34, -2, -31, -28, 18, -28, 21, -12, -21, 7, 15, -11, -5, -35, 1, 2, 6, -13, 5, 11, -26, -7, -15, 17, 8, -34, -6, 2, 1, -13, -23, 0, -6, -18, -29, -42, 17, 4, 3, -56, -5, 19, -30, -38, -12, -12, 28, 15, -10, 8, 8, 25, -23, -36, 10, -49, 40, 7, 7, -1, -44, -29, -39, -36, -1, 11, -16, -19, -1, 3, 29, 10, 24, -4, -16, -47, 18, -29, 9, -18, -36, 31, 19, -3, -5, 22, -43, 19, 23, -44, -29, -12, 23, 6, -5, 21, -20}
, {-37, 28, -5, -8, 18, -5, 11, -3, 3, 0, 9, 6, -7, -15, 16, -4, 0, -3, 26, -18, 13, 29, -9, 4, -15, 19, 4, 0, -2, -15, 0, -3, -6, 3, -22, 14, -8, 10, 12, 2, 19, 22, 11, 2, 2, 3, 10, 33, 8, -6, 9, 2, 17, -6, 2, -17, 31, -8, 9, 18, -2, -8, 10, 3, 6, -17, -57, 7, -20, -8, 4, -23, -19, 0, 42, 9, -33, -2, -3, 22, -29, 24, 23, -4, 32, -18, -10, 3, -28, -12, -38, -15, 4, -36, 0, 1, -5, 8, -11, 13, 6, -10, 25, 5, 6, 3, -2, -3, -21, -4, -5, 22, -19, 13, 6, 14, -13, -2, -8, 14, 14, 0, -4, 22, -4, 4, -22, 15, 4, -8, 9, 29, 26, -1, 2, 19, -27, 6, -35, -18, -3, -35, 6, 16, 2, -5, 2, -7, -22, 21, -3, -23, 33, 7, -1, 8, -42, 0, -14, -12, 24, 30, 13, -23, 10, -14, -4, -5, -7, -23, -6, -15, -7, 0, -32, -21, 4, -17, -40, 45, -22, 1, 37, -48, 4, -18, -14, -30, 39, -11, -14, -17, -14, -17, 22, -8, -21, 12, -25, 0, 16, 11, -10, 17, 18, 4, 25, -18, -20, -14, -25, 0, 0, -12, 11, 2, 8, 16, 4, -22, 13, 16, -4, -21, -21, -8, 3, -21, -34, -1, 4, 11, 14, -34, -11, 12, 3, 11, -41, -8, -6, 8, 37, -27, 23, 13, 2, 1, 11, -4, -16, -5, 4, 5, -19, 12}
, {21, 20, 11, 28, 36, 18, 17, -8, -11, -12, 10, 8, -8, -10, -22, 20, -3, -33, -37, -6, 1, -51, -14, -35, -14, 17, 7, -25, -5, -17, 5, -19, -15, -20, -8, -13, -5, -19, -22, -40, 10, 19, -6, -32, 0, -16, 2, 12, -21, -44, -8, -14, -12, -3, -33, -14, 18, 16, 11, -7, -27, -29, -1, -6, 18, 3, 13, 27, -5, -26, -1, -1, 3, -15, 2, -24, 4, -16, -18, 37, 27, 25, 16, 22, 56, 4, -15, -24, 9, -27, -23, 8, 8, 35, -15, 2, -7, 6, 15, -2, -25, -1, -34, 5, -22, -18, -12, 6, -7, 13, -16, -8, 21, -46, -3, -30, -29, -12, -11, 12, -43, -3, -22, -10, -28, 6, -25, -30, -12, 9, 24, 0, 1, 3, 11, -21, 7, -14, -4, 3, -13, 10, -26, -23, -21, -14, -53, 25, 36, -16, -11, 11, -16, 8, 4, -4, 4, -23, 5, 1, -9, -4, -19, 16, 6, 19, 3, -6, 15, 11, 9, -10, -16, -5, 17, 10, 0, 21, 6, -37, 5, -2, -6, 1, 11, 7, 4, 20, 10, 13, 11, -32, -8, -17, 15, 9, -13, -36, 5, 0, -27, -26, 22, 1, -6, 1, 38, -11, -17, 16, -20, -15, -12, -15, 11, -11, 5, -16, -21, -28, 17, 16, 10, -9, 0, -13, 7, 15, -9, 6, 9, 12, 5, -22, -12, 14, 26, 17, 10, 3, 15, -5, -4, -6, 9, -19, -3, 6, -6, 2, 11, 17, -17, -1, -30, 14}
, {13, -21, 6, -17, 6, 9, -27, 26, 7, 33, -29, -35, 21, -13, -18, -10, 6, 22, -6, 1, -3, 8, -2, 3, 5, 3, 15, 7, -22, 10, 17, 2, -18, -8, 4, -36, 6, 3, -21, 6, -26, -36, -11, 4, 10, 12, -1, -4, -2, 1, -23, -18, 2, 14, 8, -37, -16, 2, -20, -6, -5, 11, -15, -18, -28, -43, 47, -19, 16, 31, -23, 0, -15, 4, 5, 6, 12, 7, 22, -35, -13, -30, -21, -17, -55, -8, 5, 2, 34, 22, 4, -17, -44, -3, -20, -19, -25, -8, 16, -12, 0, 30, 16, -4, 26, -14, -3, 11, -2, -18, -57, -45, -35, 19, -7, 8, 24, -3, 6, -11, 9, 11, 10, 2, 22, -14, 31, 28, 4, 6, -18, 12, -6, -7, -21, -44, 9, -17, -1, -12, 2, 3, -7, -18, 1, 2, -2, -13, -17, -22, -22, 4, -4, -3, -16, 18, 10, 0, 8, 11, -12, -36, -21, -18, 2, -17, 8, -12, -11, -19, -29, -11, 6, 24, 7, 10, 0, -15, -5, -11, 11, 4, -23, 12, 7, 8, -7, -16, -18, -2, 9, 9, 16, 5, 18, 5, -26, -37, 20, -2, -17, -15, -13, -14, 18, -20, -4, -10, 12, -24, -16, 18, 1, -8, 9, 0, 5, 10, -17, 20, -6, 15, 31, -3, -7, -9, -10, 14, -12, -39, -14, -27, 16, 7, 29, 16, -7, -23, 12, 20, -3, -29, 2, 33, -15, 1, -26, 0, -6, -10, 17, -31, 22, -26, -16, 15}
, {-53, -33, 6, -24, -28, -3, -23, -10, -24, 8, -22, -26, 37, -6, 3, -8, 1, 4, 2, -25, -26, 4, 14, -4, -9, 2, -23, 4, -15, 20, -15, 12, -13, 17, 6, -44, 6, -17, -1, 3, -14, -43, 4, 23, -4, 13, -3, -26, 15, 5, -23, 8, -1, -7, 13, -4, -7, -14, 7, -8, 13, 17, -27, -8, -4, -14, 11, -25, 27, 39, -3, 0, -15, 3, -17, 17, 28, -4, 4, -7, -18, -17, -33, 0, -33, 8, -26, -16, 11, 8, 46, -28, -20, -37, 8, 1, -15, 10, 19, -10, -3, 31, 4, -18, 17, -5, 8, 30, -10, -49, 10, -7, -38, 42, 3, 11, 32, -44, -8, -9, 20, -8, 28, 8, 20, -21, 47, 18, -20, 5, -9, -16, -30, 2, -26, -25, 2, -38, -2, 4, -5, 4, -5, 2, 2, 23, 3, -33, -11, -1, -20, -13, 9, -20, -21, -56, 9, 10, -3, -2, 12, -19, -4, -1, -11, -21, -15, 12, 2, -16, -45, 7, 6, 4, 15, 11, 5, 6, 4, -21, 1, -12, -25, -3, 10, -21, -6, 12, 0, -12, -12, 0, 3, -1, -26, 15, 7, -10, 19, 1, -10, -18, -10, -1, -17, 9, -30, 3, 15, -16, 3, 16, -23, 5, 6, -7, 4, 13, -6, 21, -4, -6, 14, -10, -2, 11, -22, -11, 7, 3, -22, -26, 8, -1, 14, -3, -6, -44, -6, 15, -4, -21, 1, 10, 7, -23, 0, -9, -1, -11, 3, 1, -14, 1, 14, 14}
, {59, -1, -9, 30, 26, 0, -4, 14, 10, -9, -10, -4, -2, -2, -18, 7, 4, -12, -5, -20, 1, 0, 6, -16, -17, -33, 7, -21, 6, 8, 1, 22, -26, -21, 10, -2, -8, 12, -46, -4, -23, -17, -36, -21, 4, 0, -12, 19, -1, 2, -8, -12, 10, 17, 12, -11, -24, 3, -23, -19, -37, 4, 20, -24, 13, 1, 26, 15, -2, 17, -7, 19, 18, 29, -15, -1, 19, -18, 5, 18, 3, 8, -22, -21, 11, 34, 25, -15, 23, 8, 27, 6, -8, 7, -22, -11, -2, -7, 27, -23, -19, 4, -19, -5, -16, -31, -23, -16, 29, 5, -30, -6, 16, -1, -29, -22, 14, 7, -12, -35, 6, -4, -29, -7, -2, 24, 1, -23, -15, -2, -2, -1, 6, -1, 10, 23, 3, 0, -12, 12, 0, 11, -20, 6, 2, -5, -24, 19, -16, 21, 15, 4, 13, 8, 23, 5, 15, 5, -2, -4, 16, -2, 0, 12, 1, 10, 3, 10, 11, -8, 9, 8, -6, 6, 2, -18, -11, -4, 15, -8, 8, 7, -16, 0, -15, -2, 8, 0, 1, -7, 4, -10, -16, -34, 1, -17, -4, 15, -17, -21, 20, 22, -8, -13, 4, 29, -1, -7, -9, -8, 2, 29, -29, -8, 5, 11, -5, 13, -3, 22, -15, -22, -19, 0, 2, -12, -12, -2, -5, 14, 19, 14, -16, 5, -15, 9, -2, 5, 15, -2, 6, -3, 22, 1, -8, 18, -6, 27, 10, 28, 18, 1, -1, 17, 17, -16}
, {2, -50, -4, -35, -26, -11, -17, 36, -20, 6, -11, -49, -4, -9, 13, 17, -9, 13, 29, -26, -12, -26, 8, 5, 1, -6, -11, -4, 1, 12, 7, 4, 6, -13, 5, -13, 10, -32, -1, -9, 5, -20, -7, 17, -26, -10, -15, -24, -7, -32, -12, -2, -22, -3, -22, -29, -36, 3, -39, -16, -5, -14, -35, 3, -35, 17, -6, -45, 26, -9, 12, -25, -24, -24, -30, -5, -24, -6, -1, -13, 15, 6, -5, 22, -14, -18, -38, -11, -34, 12, 10, -16, 1, -47, 4, 9, 13, 10, -39, -34, 19, -10, 2, -7, 20, 11, 13, -23, -61, -56, 8, -11, -11, -16, -7, 6, 2, -41, -22, -7, -15, -27, 20, -38, 6, -14, 9, -8, -34, 12, 17, -29, -36, -13, -24, -28, 29, -27, 7, -10, -5, 6, 22, -4, -31, 9, 1, -16, 3, -24, 6, 5, -9, -8, -37, -22, 10, 10, 6, -21, 10, -23, -6, 2, 0, 3, -3, 10, -14, -29, -3, 15, 8, -5, -4, -26, -3, -28, -4, 8, -35, -22, 5, 19, -4, -11, -3, 2, 17, -2, -2, -33, 10, -3, -32, -8, 11, 4, -15, 4, 2, -13, -37, 19, 14, 1, -36, 10, 7, 13, -30, 1, -45, -1, -1, -62, -15, 26, 12, -54, 22, -33, 1, 2, 4, 12, -16, -21, -1, 3, -57, -4, 9, -22, -7, -5, -41, -10, -40, 11, 7, 8, -28, -7, 14, -4, 17, -39, 10, -36, -22, -4, -12, -5, -15, -50}
, {-11, -20, 9, 16, -1, 16, -8, 15, 43, 14, 4, -25, 3, -3, 4, 5, 3, 25, 4, -8, 10, -9, 7, 7, 9, 2, 7, -1, -11, 18, 25, 4, -14, -2, -9, -27, -10, 5, 9, 14, -22, -3, 5, 14, -8, 17, -5, 19, 18, 24, -14, -11, 9, 19, 14, -4, 6, -12, 7, 7, 6, 13, -5, 4, -6, -28, 29, -34, 24, 22, -27, -11, 13, 0, 19, 6, 5, -6, 43, -46, -41, 8, 3, 5, -35, -6, -18, 2, 15, 18, 22, -9, -26, -9, -21, -35, -50, -18, 30, -1, 9, 29, 21, 10, 21, 11, 3, -17, -12, -31, -7, 2, -28, 7, 18, 25, 22, 2, -19, 9, 29, 7, 8, -10, -13, -20, 29, 23, 23, -14, -29, 15, 11, 13, -28, 3, 6, -1, -12, -18, -5, 14, -9, -26, 9, 1, 6, -12, 0, 28, -3, -9, -26, 7, 14, -8, -7, -5, 3, 18, -31, -18, -10, 14, 15, -1, -9, -3, 4, -19, -30, 2, -22, 6, 1, 17, -3, 14, -7, -2, 5, 18, -12, 9, 3, 14, -8, -9, -14, 5, -2, 15, -2, -4, 1, -2, -35, -42, -2, 15, -3, 4, 13, -2, 15, -10, 20, -8, -8, -23, -10, 9, 25, 27, 35, -13, 2, -33, -27, -9, -6, 10, -7, -28, 2, -5, 22, -15, -30, 27, -12, -3, -6, 6, -4, -27, 8, -5, -24, -2, 4, -29, -28, -8, -32, -20, -16, -8, -24, 4, 3, 3, 1, -16, -5, 7}
, {33, -3, -13, 7, 13, -11, 27, -10, 25, 16, -24, 10, -21, -12, -5, 3, 29, -15, -2, 10, -1, -9, -17, -18, -28, 1, 4, -32, -4, -26, 13, -25, 15, 14, -6, -17, -3, -4, -19, -25, -22, -2, -13, -27, -4, -10, -8, -31, -1, -23, -10, -20, -4, 5, -32, -16, -18, 13, -2, -11, -11, -12, 8, -11, -2, -1, 30, -3, -1, 22, -7, -1, 19, 21, -9, 6, 12, -8, -3, -13, 18, -32, -33, -12, -11, 4, 14, -18, 30, 10, 23, 13, -36, 2, -23, 11, 23, -5, 37, -40, -3, 5, -15, 6, -9, -28, 10, 23, 30, 0, 2, -7, 16, 11, -21, -9, 9, 19, 12, 4, 12, 24, -14, -29, 12, 22, 15, -8, 37, 5, -6, -19, -4, 17, 9, -28, -3, -5, 14, -11, -19, 3, -3, -28, 8, -4, -23, 4, 15, -42, -1, 4, 8, 6, 5, 13, 10, 2, 9, -7, -9, 6, -5, -12, -9, -39, -1, -12, -35, 20, 3, -32, 0, -5, 20, 18, -4, -41, -15, -7, 12, 16, -38, 12, 16, 7, 10, -35, -31, 10, -3, 4, 4, 8, 20, 12, -19, -26, 16, 19, -11, -12, 30, -36, -8, 15, -17, 0, 13, -44, -19, 30, -5, 12, 8, 0, 13, -5, 7, 20, 14, 9, 1, 0, 25, 4, 23, -2, -36, -5, -11, -33, -16, 19, 9, -15, -8, 7, 21, 31, 0, 7, -1, -6, -27, 1, -31, -17, -12, 8, -13, -47, -6, 7, 9, 11}
, {-21, 0, -9, 9, -19, -13, -47, -26, 9, 21, 7, 23, -24, 3, 11, -58, -48, -65, -65, 9, -17, -6, -19, -5, -28, -11, 39, -37, -38, -9, -2, -30, 8, -38, -26, 30, -38, 26, 5, -16, 22, 19, 3, -14, 20, -39, 5, -26, -22, 2, -6, -6, 1, -21, -4, -7, 4, 11, 7, 10, -18, -20, 14, 14, 3, 14, 19, 20, 1, -5, -8, 21, -12, 30, 20, -11, 7, 7, -16, 8, 22, -16, -13, 12, -17, 62, -1, -30, 20, -28, -5, 4, -10, 2, -15, -29, 1, -10, 10, 18, -24, 1, -6, 19, 2, 1, -2, 3, 21, -13, -12, 16, 7, -22, -17, -25, 15, 25, -18, -6, -6, -15, 2, 4, -12, 1, -2, 7, -31, -10, -5, -7, -18, 10, 26, -34, -4, 5, 25, 0, -11, 13, 21, -22, -19, -45, -36, 20, 2, 1, -3, 8, -11, -1, 14, 17, -2, -4, -6, -2, 1, -8, 1, -3, -18, 10, 28, -4, -27, 23, 29, -13, 22, -4, -17, 23, -9, 5, 5, -7, -7, -19, -43, 14, 23, 33, 8, -11, -7, 5, 41, 7, -12, -3, -31, -16, 8, 11, 3, -11, -11, 28, 11, -5, 21, 32, -25, 2, -18, -4, 5, 27, -31, 2, 0, -1, 15, 7, 11, 6, 18, -3, -6, 5, -2, -5, 7, 25, -9, 9, -16, -36, 1, 37, 2, 11, -7, 13, 27, 1, -33, -4, 2, -15, -4, 8, -11, 15, 22, 13, 27, -1, -35, 13, 10, 5}
, {38, -5, 9, 28, 21, -20, 14, 51, -17, 6, 0, -47, 11, 1, 11, 25, -3, 12, -3, -14, 3, 11, -6, -7, 7, -1, 3, 8, -4, 11, 7, 3, -12, 8, 9, -17, 30, -36, -31, 5, -30, -28, -20, -9, -11, 1, -1, -12, -12, -20, -29, -12, -56, 4, 23, -13, -24, -14, -32, -23, 16, 24, -31, -18, 14, 2, 35, -9, 9, -16, 11, 17, -7, -4, -22, -9, 30, -9, -28, -17, -8, 1, -7, 15, 36, -15, 19, -30, 9, 3, 0, 7, -12, 18, 8, 6, -3, 15, 13, -16, -19, -23, -24, -7, -7, -11, -16, -9, 26, -1, -7, 9, -5, -25, -18, -25, -4, -14, -17, -33, 11, -4, -10, -37, 0, 4, -22, -19, -17, 27, 10, -19, -11, -14, -17, -11, 12, -29, 15, 3, 11, 14, -24, 11, -1, 19, 0, -8, 16, 5, 5, 7, 29, -7, -21, -17, 20, 29, -8, -5, 15, -5, 2, 15, 20, -11, -32, 31, 6, -2, -22, 14, -6, -2, 28, -36, 1, -4, 0, -10, 12, -3, 11, 3, -23, -5, -8, 15, 32, -8, -3, 1, 2, 4, -24, 6, 8, 28, -22, 9, 0, -7, -4, -2, -15, 8, -9, 7, 1, 19, 3, -33, -8, -10, -9, -8, -11, 14, 7, -8, 16, -25, -7, 11, -1, 16, -14, -4, 21, -5, -18, 8, 10, -38, -17, 1, -11, -21, -6, -7, 8, 9, -16, 8, 6, -18, 4, -14, 6, -23, -28, -2, -9, 12, 5, -49}
, {7, 21, 7, 3, -5, -4, -9, -37, 6, 1, 21, 10, 10, -21, -3, 1, 4, -11, -15, 6, -29, -21, 10, -7, -5, 2, -41, 10, -5, 19, 6, -2, 14, 7, -7, 3, 19, 11, 10, 16, 9, 5, 0, 10, 0, -14, 7, 23, 18, 13, 4, 4, 3, -2, 0, 18, 13, -17, 1, 10, -6, -9, -14, 15, 0, -17, -14, 8, -13, -3, 7, -15, -9, 11, 9, -8, 5, -17, 18, 4, -25, -9, 1, -19, 12, -12, 22, -2, -18, -14, 7, -12, 1, 33, -20, -21, 2, -19, -35, -9, -9, -26, -1, 13, -6, 0, -3, -2, 23, 28, -20, -2, 6, -16, -3, 1, -7, 24, 15, -4, -8, 15, 10, 14, 3, -41, -21, 8, 0, 9, -4, 8, 2, -34, 3, 14, -7, 6, -25, 15, -20, -13, 11, 38, -5, -7, 27, -6, 6, 27, -13, -16, 7, 0, -10, -11, -11, 9, 7, -17, -2, -9, 11, 0, -2, -5, 14, -8, 17, 2, -10, -14, -24, -23, 7, -2, 13, -8, -8, 8, -15, -2, -12, -1, -15, -19, -19, 19, -19, -22, -4, -2, -20, -30, 2, 13, -22, 10, -39, -12, 27, -18, -35, 7, 6, -11, 27, -14, 9, 23, -11, -30, -22, -5, -6, 42, -21, 23, -25, 2, -23, -17, -11, 2, 12, 34, -7, -20, 7, 11, -16, 20, -11, -13, -14, 37, -26, -9, -10, -2, 7, 31, 9, 0, 3, -26, 27, 22, -15, -38, -39, 24, 18, 7, 9, -34}
, {-31, 16, -2, -14, -25, 30, -4, -24, 4, 0, 13, 7, 2, -19, 26, 11, -20, -13, 7, 16, 0, 17, 7, 11, -8, -7, 7, -5, -1, -16, 2, -21, -3, 0, -14, 18, -7, -8, 0, -2, 19, 38, 27, 12, -2, -7, 5, -10, -13, -3, 0, 36, -7, 3, -2, 10, -11, 11, -5, 12, -4, -52, 12, 19, 12, 17, -25, 4, 17, -2, 18, -7, -13, -41, 14, -3, -32, 20, 1, 26, -14, 5, 10, -11, 7, 15, -41, -1, -14, 12, -36, -14, 20, -32, 25, -9, -2, 1, -8, 27, 15, 9, 4, -15, 4, 42, 20, -9, -42, -39, 12, 30, -29, 6, 11, 36, 5, -24, -13, 12, 2, -19, 4, 16, -6, -1, 5, 29, -1, -12, -2, -15, -14, 2, 9, -26, -12, -6, -3, -10, -7, 16, 14, -22, -21, 4, 3, 4, 0, 7, 4, -2, 4, 10, 8, -5, -9, -3, 3, -4, 8, -7, -6, -1, 0, -8, 15, -8, -5, -3, 27, 3, -12, -3, -8, -2, -26, -5, 14, 20, -4, -13, 8, -7, 6, 12, 1, -22, 13, 2, 8, -11, -17, -18, 14, 23, -4, 5, -4, -8, 6, 12, -19, -2, 22, 2, 16, 4, 11, -3, 9, -5, 2, 11, 0, -7, 25, 2, -20, -27, 3, 15, -10, -14, 12, -2, -7, -9, 3, 18, -10, 5, -12, -9, -7, -2, 12, 31, -25, -26, 5, 14, 8, -29, 12, -17, 11, 4, -4, 7, 3, -11, -27, -11, -9, 11}
, {-3, 15, -6, -11, -30, 15, -12, -14, 27, -5, 10, -22, 10, 9, -3, -10, -12, 26, 11, -26, 6, -23, -12, 14, 10, 25, -22, 17, 20, -21, -10, 20, -13, 13, -14, -7, 4, -18, 14, 3, -41, -6, 4, 0, -12, -19, 2, 7, 14, -18, -5, 24, -17, -5, -6, -7, 36, -6, 5, 19, 17, 19, -22, 23, -45, -11, -14, 0, -1, 8, -4, -32, -19, 24, 16, -6, -30, 9, 24, -5, -17, -11, -12, -2, -4, -29, 2, -9, -47, -4, -4, -29, -17, -4, -18, -17, -2, 17, -18, -10, 14, 18, 4, 21, 2, -11, -6, 11, 13, -12, -5, -33, 8, 5, 18, 3, 16, -2, 18, 12, -25, -2, 16, 22, -14, -25, 10, 3, 5, 16, 0, 34, 13, -12, -7, 17, 13, -21, -8, -3, -6, -27, 20, 14, 7, -5, 8, -9, -4, 23, 4, -4, 14, -28, -4, -36, 3, 22, 6, -13, 15, 1, 23, -7, -6, 2, -27, 20, -1, -21, -17, 12, 0, 10, 0, -19, 23, -12, -25, -7, -7, -21, 13, -8, -7, 1, -18, -2, -16, -29, -8, -20, 19, 10, 9, -15, 1, -14, 0, 1, 17, -36, -1, 8, 13, -23, 14, -5, -14, 22, -1, -28, 24, -28, 14, -34, -18, 11, -2, 11, 19, -6, 7, -5, -23, 6, 1, 13, -7, -25, -49, 8, 18, -18, -17, -6, -16, -27, -15, 18, -25, 13, -10, 24, 27, -1, -1, -10, -10, -23, -16, 0, 3, -3, -28, 14}
, {-17, 20, -7, -12, -25, 8, 3, 14, -18, -10, -4, 13, -21, -3, -6, -4, -29, 4, 3, 15, 24, -26, 0, 7, 1, -13, 4, 3, 14, 0, -18, -27, -11, 1, -1, 16, 1, -3, -9, 6, -3, 5, 10, -12, 5, 0, 23, -11, 3, -6, 4, 11, -13, 8, -7, 10, 10, 8, 16, 11, -13, -27, -8, 13, -23, -2, 15, -8, 4, -4, 2, -9, 3, -9, 19, 1, -24, 6, 11, 3, -2, 4, -19, 17, -15, -41, 14, 14, -1, 16, -11, -14, 3, -21, 6, 16, 2, 4, -21, -9, -8, -18, 6, -9, 1, 5, 21, 13, -9, -10, 0, -11, -23, 26, 8, 4, 14, 0, 8, 9, -11, -16, -8, 27, -1, 17, 14, 6, -5, 9, 16, 3, -20, -25, -1, -26, -21, -25, 29, 3, -4, -19, -2, 19, -9, -27, 0, -14, 1, -15, -21, 3, 0, 16, -21, -20, 2, -1, -15, -7, 11, 2, 14, 1, -10, -4, 6, 0, -5, -11, -9, -6, 11, -18, -20, 23, -3, -11, 7, 9, -11, -10, 0, -7, -6, -6, -10, 8, -16, -35, 9, -2, 4, 13, -17, -16, 6, 0, 18, -16, -18, -53, 2, 23, -14, -35, -51, 26, -4, 21, 0, 3, 1, -18, -27, -38, -23, -5, 1, 10, 20, 7, 45, -9, -4, 20, 2, 42, 3, -55, -30, -38, 4, 4, 4, -17, -23, -32, 11, 30, -35, -14, -41, -12, 10, -14, -17, -33, 30, -27, 1, -24, -21, 0, -56, -7}
, {40, 14, -7, 19, 10, 11, -4, 10, -23, 13, 22, 9, 11, 2, -9, 19, -4, -9, 7, -17, 14, 26, 7, -52, -18, -8, -1, 1, -21, -4, 8, 6, -8, -55, 4, 1, 14, -5, -8, -8, -4, -11, -43, -52, 27, -9, -12, 36, -3, 5, -26, -27, 1, 9, -25, -43, 18, 9, -1, -38, -31, 12, 4, 1, 17, 9, 14, 15, 17, -28, 0, -7, 4, -8, -34, 13, 15, -25, -16, 18, 18, 23, 18, -24, 20, -23, -16, -9, -13, -9, 21, 24, -1, 35, 0, -1, -21, 5, 3, -11, -3, -21, -35, 12, 6, -27, -19, 2, -2, -8, 0, 10, 32, -28, -41, -4, 14, 11, 7, -23, -3, -6, -30, -33, -7, -7, -16, -34, -16, 7, 10, -3, 49, -5, 5, -19, -9, -6, -22, 6, 1, 12, 13, 13, -23, -5, -15, 10, -25, 0, 20, 17, 16, -1, -23, 24, -6, 5, -4, -19, 8, 7, -23, 17, -16, 8, 7, 15, 17, 34, 11, -4, -20, -14, -1, -11, -14, 2, 17, 4, 0, -24, 6, 3, -21, -28, 10, 2, 19, 11, 4, -4, -10, -47, -17, -11, -11, -1, -33, -49, 13, -39, -22, -9, -39, 20, 7, -37, -27, 0, 17, -23, -1, -14, -35, 23, 4, -19, -23, -35, 20, -30, -8, 0, -24, -5, -4, 21, 12, 14, 10, 11, -8, -17, -47, 17, -26, -6, 11, 1, 4, 17, -9, -11, -17, -43, 9, 17, 13, -34, 6, 10, -6, 1, -8, -28}
, {-5, 28, -23, 2, -4, 6, -9, -50, 29, 13, -2, 3, -25, -6, -11, -1, -8, -24, 3, -14, 14, -27, -9, 2, 10, -13, -33, 19, 12, -3, 14, 8, -14, -5, 5, 12, -5, 0, -5, 6, -8, -3, -12, -14, 0, 9, -3, 42, 1, 14, 5, -2, -5, 22, -34, 11, -2, 18, 17, -8, -24, 1, 8, 18, -11, 10, -31, 7, -8, -1, -12, -5, 12, 10, 13, -29, 4, -23, 12, -17, -40, -21, -20, -8, -2, -6, -9, -8, 0, 0, -10, 25, -7, 10, 2, -11, 12, 7, -19, -14, 12, -5, -10, 2, -15, 11, 5, -15, 15, 2, 2, -2, 28, -22, 4, 12, -18, 4, 15, -12, 10, -5, 7, -3, -15, -7, 16, 6, 12, 1, -36, 11, -3, 11, 5, 14, 0, 0, -20, 2, -36, 0, -8, 10, -19, -3, -1, -48, 20, 29, -10, 8, -39, -10, 3, -25, -10, -7, 27, -59, -31, -36, -19, 15, -13, 25, 1, -9, 3, -17, -11, 2, -12, -5, -37, 0, -12, 3, 4, 20, 18, -6, 0, -15, -26, 13, -7, 34, -42, -32, 19, -2, 0, -8, -2, 18, -3, 5, -39, 4, 21, -7, -35, 11, -18, -16, 27, 9, 4, 1, 13, -52, -9, 20, 10, 34, 1, 13, -8, -10, -32, -5, -11, -2, 11, -1, 19, -24, 5, 16, 5, 21, 23, -22, -24, 9, -21, -25, -20, -21, 38, 21, -31, -17, 12, -25, 5, 6, -29, -37, -20, 16, 19, -9, 29, -22}
, {37, -12, 19, 11, 29, -26, -1, 42, -10, -4, -5, -41, 15, -9, -13, 16, 10, 13, -2, -15, -1, -18, 1, -17, 7, -26, 3, -1, 13, 7, 12, 15, -15, -22, 12, -22, 27, -46, -13, 9, -46, -38, -23, -8, -2, 11, -8, -9, 14, -27, -42, -3, -36, 8, 10, -41, -1, -31, -44, -37, 18, 3, -42, -31, -3, 6, 34, -8, -2, -6, 15, 6, -1, 3, -15, -22, 42, -4, -31, -17, 5, 6, 6, 1, 21, -11, 7, -18, 27, -2, 3, 21, 7, 20, 9, 1, 20, -5, 21, 5, -16, -18, -28, -15, -23, -25, -21, 1, 37, 21, -9, 2, 5, -20, -23, -21, -12, 6, -2, -12, 6, 6, -12, -27, 0, 8, -9, -25, -16, 6, 15, -1, 25, -21, -16, -10, -6, -32, -34, 5, 24, 4, -8, 5, 19, 23, 21, -15, 11, -3, -2, -15, 19, 14, -11, -20, 7, 5, 3, 9, 11, -10, 3, 5, 16, -14, -22, 11, 8, 8, -33, 3, 7, 16, 12, -6, 17, -3, -8, -32, -12, 6, 13, -2, -27, -12, -9, -1, 6, 25, -11, -26, -4, 7, -7, -5, 13, 1, -9, -5, -9, -31, -2, 3, -1, -14, -8, 15, -11, 25, 0, 0, 16, -2, 2, -11, 2, 26, 13, 0, 1, 5, 1, 9, 9, 18, -21, -22, -4, -10, -37, 11, 7, -29, -18, 7, 5, 6, 6, -7, 20, -4, 11, 17, -5, -28, 14, 5, -4, -2, -24, 10, -15, 1, -6, -18}
, {-21, 40, -13, -1, -4, 2, 14, -29, 20, -1, 2, 24, -24, -10, 10, -18, -21, -3, 5, -8, 22, -1, 8, 2, -2, 10, -4, -1, 7, -14, -8, -36, 5, -20, -7, 43, -17, 22, -5, -19, 23, 36, 14, 6, -21, -13, 28, -8, 2, -2, 6, 10, 24, -9, -10, 4, 18, 4, 2, 8, -20, -24, 33, 28, 10, 10, -6, 20, -3, -18, 1, -10, -4, -32, 27, 4, -11, -3, 9, 19, 27, -5, 10, -11, 32, -1, -18, 2, -14, -12, -43, 25, 15, -19, 9, 16, 14, 8, -2, 30, -7, 2, 24, 13, 3, 11, 1, -9, -12, 8, -6, 23, 33, -5, 6, 3, -25, -17, 15, -4, 8, -4, -9, 29, -2, 6, -34, 1, -21, -25, 18, 1, 23, 1, 47, 9, -21, 11, -1, -16, -3, 0, 11, -21, -12, -16, -4, -8, 2, -30, -13, 11, 27, 12, -3, -6, -9, 15, -36, 1, 10, 10, 5, -42, 8, -13, -5, 17, -8, -5, 18, 11, -2, 5, -33, -2, 3, -8, -3, 20, 1, 4, 11, -16, 16, 8, 3, -28, 16, -4, 2, -1, -30, 5, 25, 9, -1, 3, -5, 10, -11, 10, -1, 10, 19, -13, 11, 4, -16, -28, -34, 9, 2, -6, 10, 27, 8, 3, 11, -12, 3, 30, -40, -13, -3, 21, -14, -4, -28, 7, -14, -3, -14, -24, -17, 0, -9, 19, -8, -6, -10, 1, 35, -19, -20, 10, -2, 9, 1, 16, 11, -1, 11, -3, -5, 32}
, {-32, 17, -10, -13, -46, 13, 18, -18, 11, 9, 0, 14, -12, -9, 0, -12, -16, -4, -16, -6, 0, -18, 7, 13, 12, -28, -48, 22, 5, 2, 12, 7, 3, 3, 0, 11, -5, -16, 12, -4, -20, 6, -14, -6, 1, -1, -11, 13, -16, -3, -1, -2, -6, 12, -53, -3, 8, 11, 0, 2, -6, 5, 12, 0, 11, 0, 24, 12, -9, 10, -5, 24, -9, -17, 0, -12, 26, -20, -14, -18, -2, -2, -6, 3, -33, 15, 2, -9, -1, -12, -18, 7, -23, 3, 10, -7, 9, 3, -24, -5, 4, -17, -2, 14, -22, 17, 5, -11, 12, 6, 6, 18, 14, -23, 4, -1, -11, -8, -4, -11, 0, -17, -4, 5, -7, -17, -16, 0, 12, 2, -2, -14, -11, -16, -6, 27, 12, -8, -26, 1, -23, -11, -10, 15, -16, -10, -6, -14, 12, 25, 10, -12, -16, -16, 7, -23, 3, -8, 20, -30, -27, -21, 4, 3, -34, 13, -8, -22, 20, -10, -7, -27, -7, 19, -2, -38, 0, 9, 5, -18, 2, -24, -30, 5, -50, 0, -19, 25, -65, 26, -6, 9, -4, -39, 31, -10, 0, -4, -56, -6, 15, 26, -37, -10, 8, -7, 18, -13, -5, -13, 26, -39, 3, 32, 11, 4, 11, -4, 3, -18, -44, -10, -30, 15, 28, -22, 21, -38, 5, 27, -20, 37, -18, -27, -20, 9, -15, 12, -11, -44, 16, 8, -18, -7, -10, -24, 34, -10, -62, -19, -39, 22, 13, 7, 29, -37}
, {-42, -11, -6, -34, -21, 10, 1, -14, -12, -14, 1, -26, 4, -20, 12, -14, -18, 13, 18, -16, -11, -19, -2, 26, -14, -16, -25, -11, -2, 0, 5, 11, 1, 15, 9, -52, 5, -47, 22, 0, -20, 14, 9, 10, -14, -7, -28, -30, 12, 11, -33, 15, -9, -5, 0, -20, -13, -5, -1, -9, 13, -21, -21, 10, -13, 0, -21, -18, 10, 11, -2, -40, -3, -48, 22, 9, -22, 23, 15, -17, -24, 14, -13, -22, -35, -47, -53, 2, -28, -7, -19, -13, 10, -35, 12, -8, 4, 0, -7, -5, 23, -7, 39, -23, 20, 14, 18, 26, -40, -57, 10, 1, -27, 22, 26, 4, -4, -55, -19, 19, -2, -30, -5, 18, -4, -1, 9, 16, 19, -4, 6, -34, -26, 6, -50, -7, 9, -27, -12, -3, 16, 9, -2, -22, -8, 14, -3, -10, -1, -12, -23, -6, 25, 10, -6, -12, -6, -6, -17, 8, -2, -27, -42, -2, -2, -19, -13, -1, -11, -34, -51, 7, 6, -25, 2, 22, -23, -12, 8, -19, 2, 3, 3, -8, 7, 3, 6, 6, 11, -8, 3, -13, 12, -9, -1, 18, -2, -7, -10, 13, -18, -2, -2, -21, 9, -7, -25, 11, -4, -6, 21, 15, 12, -5, -2, -38, -8, -15, -3, 0, -11, 23, 2, -3, -27, -12, -5, -12, -21, 4, -11, 2, -10, -9, -20, -13, 5, 3, -16, -12, 9, -1, 14, -18, 17, 11, 5, -10, -15, 20, -14, -7, -26, -3, -5, 15}
, {-32, 3, 4, -4, -11, -4, -3, 15, -10, -7, -23, 13, -16, 2, 9, -3, -24, -7, -1, 0, 1, 5, -12, -7, -5, 17, -9, -13, 4, -11, 8, -8, 26, 1, 10, 18, 5, 22, -10, -4, 19, -10, -1, 0, -17, -15, -15, -19, -2, -5, -8, 20, 9, -34, 8, 1, -28, 13, 5, -1, -4, -19, 0, 4, -7, 4, -8, 11, 0, 8, 4, -15, 25, -28, -11, -4, -24, 5, 5, -6, 17, -31, -26, 1, -38, -15, -43, 18, 2, 1, -15, -15, -13, -25, 5, 3, 24, 11, -20, -14, -21, 2, 15, -18, 3, 7, 8, 25, -30, -43, 4, -7, -25, 28, 22, 6, 4, -22, 5, -5, 1, -22, 4, 5, -2, 9, 13, 2, 31, -1, -11, -32, 0, 9, 2, -8, -1, -32, 10, 4, -32, -18, -13, -16, -36, -3, -28, 1, 29, -7, -46, 4, -28, 4, 0, 9, -26, -28, -13, -20, -13, -25, -24, -7, -25, -21, -13, -41, 2, 4, -14, -30, 8, -19, 7, 30, -38, 11, 7, -4, 5, 3, -39, 5, 24, 15, 9, 3, -38, -18, 21, -15, 11, 21, -33, 5, -8, 3, 30, 21, -31, -2, 31, -1, -45, -24, -31, 33, -7, 7, 14, 17, 18, 13, -2, -48, -1, 8, -4, -15, 13, -13, 9, -4, -5, -31, 15, 18, -3, 3, -14, -30, -10, 11, -5, -48, -22, 7, -9, 18, -21, -24, -14, -4, -3, -17, -26, -34, -18, 3, -1, -8, -14, -6, -4, 22}
, {7, -15, -14, -12, -59, -20, 19, -24, 7, 14, 6, -10, -41, -29, -8, 3, -40, -3, -6, 1, 5, -32, 16, -4, -2, -23, -12, -13, 30, 2, 16, -3, -2, 35, -7, -10, -10, -7, -29, 0, -37, -4, -2, -16, -9, 8, -3, -19, 13, 8, -10, -1, -12, -1, -55, 1, -10, 26, -24, 11, -9, -5, 15, 6, 6, 12, 11, -2, -5, -15, -5, -6, 14, -42, -2, -12, -5, -8, 19, -13, 4, -8, 13, -31, -28, -2, -16, 21, 12, -8, -7, 15, 8, -9, 6, -7, 0, -1, -4, 23, 14, -5, 5, 21, -3, -1, -11, 0, -23, -13, 4, 12, 31, 25, 22, 7, -38, -59, 13, 15, 13, -8, -45, 10, 9, -4, -6, 21, 21, -20, 5, 29, 9, 10, -6, 15, -3, 12, -16, -6, 19, 4, 7, 1, -3, -4, 3, -19, 30, 10, -2, -12, 3, 7, 23, -6, -1, 3, -6, 20, 7, 13, 4, -24, 3, -18, -19, 6, -19, -12, 0, -13, 3, 8, -37, -11, 4, -15, 1, 18, 7, 5, 30, -33, -25, -24, -27, -5, 11, -1, -1, -6, 12, -22, 24, 0, 0, 1, -28, 23, 7, -5, 9, -41, 15, -17, 14, -6, 9, -10, -9, -14, 47, 5, 15, -9, -13, -30, 3, 8, -32, 20, -14, 1, -10, -4, 13, -17, -27, -6, -19, 2, -27, -14, -34, -34, -2, 20, -26, -14, -3, -5, -1, -20, -41, 0, -17, -28, -21, 21, -52, -37, -18, -4, 17, -30}
, {-33, 20, -6, -17, 5, 3, -7, 24, -22, -21, -5, 4, -8, 7, 19, 21, 9, 6, 12, 15, 26, 7, 1, -9, 1, 5, -27, -17, 6, -11, -9, 14, 0, 9, -10, -9, -14, 8, 1, -14, 9, 4, 3, 2, -5, 5, -20, -33, -3, -4, 4, 7, 14, -11, -17, 17, -19, 10, 18, 23, 7, -15, 27, 4, -5, -5, 20, 14, 11, 11, -8, 16, 8, -24, -10, 8, -23, -3, -4, -6, 12, -19, -16, -24, -9, 4, -2, 1, -15, -5, 13, -10, -15, -25, 2, 2, -12, 5, -20, -3, -24, 16, 4, -23, 8, -21, -9, 10, 7, -21, -14, -9, -18, 25, -7, 17, 22, -11, -5, 12, -5, 1, 11, -1, -13, 10, 14, 7, -14, 3, -9, -42, -12, 2, -16, 5, -7, -14, 75, 6, -37, -16, 4, -29, -29, -12, -47, 16, -5, -8, -31, 1, -30, 21, -2, -15, -16, -20, -18, -49, -25, -29, -22, -6, -25, 24, 7, -18, 1, -3, 1, -14, 30, -52, -12, 7, -19, 9, 8, -5, 23, 1, -41, 7, 18, 24, 10, -2, -32, -59, 27, 1, 21, 16, -19, -7, 2, 5, 7, 5, -18, 9, 3, -11, -26, 18, -16, 7, -13, 1, 35, -9, 18, 9, -34, -23, 29, -16, 9, 21, 10, 23, 8, 9, -16, -30, 1, 26, 9, -19, -2, -31, -10, 12, -1, -43, 5, -28, 0, 6, -11, -25, 12, 9, -5, -8, -53, 2, 9, -13, -5, -14, -16, -11, -28, 9}
, {-24, 14, 6, -13, -26, -7, 6, 11, 4, 0, -2, -1, -8, 12, -7, -11, -31, -4, 17, -13, 10, -9, 1, -3, -18, -6, -29, 15, 0, -3, -4, 4, 9, -2, -5, -6, -6, -1, -9, 9, -10, -10, 4, 13, 8, -4, 1, 12, -5, -2, 1, 7, 2, 9, -25, 7, 16, -6, -7, -7, -2, -3, 11, 19, 6, -18, -16, 12, -8, 4, -15, 26, -13, 15, -7, 0, -1, -9, 17, -16, -7, 1, -20, -13, -24, -6, 23, 17, 3, 4, -2, -3, -15, -29, -6, 17, 6, 23, -9, 2, -3, -12, 9, 24, 5, 16, 6, 6, -13, -11, -3, 10, 24, -8, 6, 14, -12, 5, 3, 8, -14, -2, 3, 8, 1, 0, -5, 1, -15, 9, -9, -16, -10, 0, 10, -1, -2, -2, 44, 36, -24, -6, -1, 24, -30, -5, -17, -21, 31, 36, 13, 10, -38, 3, -9, -16, 11, -26, -10, -14, -49, -29, -4, 15, -47, 26, -2, -42, 31, -37, -6, -22, 33, -7, 4, -20, -29, 21, 1, -21, 40, -18, -38, 9, -21, 29, 8, 5, -34, -20, 16, -20, 18, 18, 0, 18, 21, 7, 1, 15, -12, -11, -8, 2, -19, -27, 3, 3, 10, -10, 31, -19, -5, 29, -3, 0, -31, -9, 22, -15, -8, -16, 15, 21, 6, -12, 24, -21, 27, 24, -36, 15, -7, -15, -15, -24, -35, 8, 18, 9, 31, -16, -57, 5, 2, -10, 1, -32, -8, -13, -9, 7, 16, -17, -25, -5}
, {-19, -11, -14, 9, 23, 14, -1, 15, -23, 1, -1, -4, -50, -6, -19, 22, -14, -44, -17, 8, 0, -27, 1, 6, 1, -11, 33, -10, -15, -11, 15, -37, 0, 8, 3, -8, -4, 17, 1, -5, 8, -14, 5, -9, 15, -36, 7, -19, -21, -2, 7, 6, -4, 0, -6, -13, 12, 20, -5, 5, -16, -18, 14, 1, 15, 8, 3, -18, 25, 8, -4, 51, 19, -5, 2, -1, -14, 0, 12, 21, 30, 5, 24, 8, 15, 25, -36, 7, 8, -5, -15, 6, 1, -40, -7, 3, -9, 2, -3, 13, -12, 7, 6, -31, -10, 11, 7, 7, -17, 4, 6, 3, -30, 11, 3, -8, 28, -1, 2, 15, 23, 16, -8, -4, -28, 4, 8, 3, 15, -26, 6, -4, 18, 2, 0, 0, -7, -6, 4, -11, -3, 0, 8, -43, -16, -7, -32, -2, 23, -35, -24, 21, -14, 19, 5, 16, -4, -22, -8, -1, 9, -4, -41, -27, 3, -3, 13, -9, -12, 8, 5, -8, 5, 3, 4, 22, -14, 12, 3, 8, 0, -10, -23, -3, 23, 6, 18, -29, 17, -11, -1, 7, -10, 16, -4, 4, 18, -13, 30, -11, -22, -8, 30, -14, -11, 13, -4, -10, -14, -3, -17, 36, 23, -10, 10, -5, -1, 5, 0, -36, 9, 19, 6, -9, -4, 6, -2, 20, -20, -23, -7, -30, 7, -2, 9, -8, 26, 24, -10, 33, -15, -3, -4, -24, 0, 4, -24, 22, 25, 22, 40, -16, -16, -3, -40, 29}
, {42, 0, -12, 45, 15, -40, -4, 4, -5, -16, -10, 17, -1, -4, -30, 32, 21, -5, -25, 5, 4, -36, 10, -24, 0, -9, -22, -8, 15, 20, -5, 20, 8, -11, 12, -11, 13, -32, -18, -3, 5, -20, -27, -44, -12, -1, -8, 3, -4, -50, -27, -17, -30, 10, 9, -20, -8, -21, 2, -9, -16, -11, -55, 0, 0, -16, 10, 5, 1, 22, -1, -5, 8, 32, -30, -24, 37, -18, 3, -27, 20, 15, -33, 8, -11, -14, 37, -24, 1, 7, 30, 1, -18, 23, -4, -12, 3, -6, 14, -16, -17, 16, -20, 10, -16, -32, -42, 0, 5, 15, -13, -2, 18, -6, -52, -10, -5, 26, -6, -26, 4, -4, -11, -26, -4, 15, 8, -25, 31, 19, 1, -21, 19, 3, 5, -8, -6, -3, -6, 1, -7, 12, -3, 4, 8, -7, -8, -1, 8, -8, -9, 9, -9, 17, -3, -5, 7, -23, 9, -9, 2, 10, 3, 18, -6, -2, -11, -2, 0, 22, -13, 2, 5, 0, 18, 22, -13, -1, -5, -27, 12, -3, -15, 3, 23, -16, 14, 11, -6, 9, 4, -18, -18, 4, 1, 6, -18, -19, 16, -11, 7, -19, 0, -11, 3, 9, 17, 9, 4, -1, 0, 22, -20, 8, -12, 17, 0, 14, -12, 12, -13, -4, 5, -1, 16, 13, 4, 2, 18, -5, 6, 4, 1, 18, 10, 22, -13, -34, 9, -33, 10, -18, -18, 15, 16, -33, 0, 0, -25, -35, 0, 17, 32, 2, 23, -6}
, {-17, -3, -6, 4, -7, -9, 18, -7, 16, -7, 7, 23, -7, -2, -5, -6, -1, 0, 7, -1, -8, -20, 8, -6, 5, -5, -47, 4, -4, -1, 30, -2, 12, 16, 12, -8, 16, 11, -1, -3, -20, 8, 9, 10, 11, 3, 10, 23, -7, -4, -6, -3, -5, 9, -1, 18, 22, -7, 8, 12, -1, -1, 21, 6, 1, -7, -16, -1, 25, 6, -7, -10, 12, -11, -11, -10, 3, 9, -3, -18, -21, 2, 2, -8, -22, -12, -5, 14, -21, 17, -18, 2, -8, 9, 3, -17, -7, 1, -29, -3, 10, -5, 0, 4, 4, 18, -4, -7, 2, 3, 4, -1, -8, 4, 7, 16, 0, -3, -9, -5, 11, 2, -4, -1, 6, 0, 17, 14, 25, 7, -30, -12, 2, 17, -6, -14, 21, 0, -2, 1, -29, 17, -11, -27, -18, -10, -10, 0, -7, 9, -17, 1, -39, 13, 1, -1, -10, -30, -2, -17, -38, -39, -14, 7, -41, 26, 20, -16, 5, -22, -17, -24, -12, -5, 3, 34, -26, 7, 7, -6, 7, 12, -27, 16, -2, 13, 0, 1, -38, 6, 25, 8, -7, 13, -29, 35, -12, -26, 4, 17, -5, -3, 26, 0, -36, -25, 33, 25, 13, -13, 11, -14, 31, 25, -5, 13, -9, -4, 6, -32, -21, -14, -23, -13, 25, -39, 35, -27, 9, 20, -15, 18, 10, -8, -35, -5, 2, 25, 0, -38, 24, -17, -42, -28, 4, -24, -1, -4, -55, 0, -15, 6, -8, -12, -3, 1}
, {0, -4, 23, -3, 8, -20, -12, 12, -7, -6, -29, 5, -16, 13, 12, 17, -4, -8, 4, 8, -6, 17, -4, 1, 0, 14, -8, -33, -16, 34, 11, -3, 6, 16, 9, -7, 21, 10, -11, 1, -4, -27, 8, 10, -6, -1, -2, -29, 1, -9, -17, 11, -1, 2, 17, 1, -14, 9, 1, 13, 1, 4, 0, -18, -7, 1, 26, -1, 17, 33, -10, 12, -35, 17, -28, -11, 15, -2, -7, -8, -2, -37, -39, -4, -28, 18, 2, -20, 33, -1, 13, -21, -2, -26, -8, -17, 6, 10, 19, 1, 4, 25, -12, 0, 14, 17, -1, 12, -25, -16, -9, -23, 8, 8, -1, 3, 16, 15, 11, -12, 2, -17, 14, 8, 19, -1, 22, -5, 10, -17, -11, -16, -33, 9, 9, -57, 18, -9, 15, 10, -1, -11, 5, -42, 3, 13, -22, -5, 22, 10, 1, 2, -12, -2, -31, -9, -6, -10, -8, 4, -8, -32, -27, -14, -3, 1, -21, -4, -22, -27, -11, -1, 33, 8, 11, 10, 3, 14, -5, -17, 4, -7, -23, 2, 6, 27, -3, 8, -20, 3, 7, -13, 2, 43, -23, 19, 18, -21, 22, 16, -18, -44, -10, 17, 8, -16, -28, 37, 15, 9, -6, 17, -17, 27, -21, -15, 8, 27, -15, -7, 24, -16, 30, -18, -11, -4, -1, 16, 10, -7, -41, -14, 6, 26, 7, -30, -20, -38, -1, 10, -6, -48, 1, 1, -23, -16, -29, -21, -2, -42, -9, -16, 4, -16, -18, 9}
, {35, -8, 13, 46, -11, -21, 11, 33, 45, 10, -4, -2, -5, -11, -10, 4, 18, 1, -10, -7, 4, -6, 6, -4, -9, 2, -18, 10, -17, -2, 11, 9, -4, 16, -6, 6, -1, -9, 12, 6, -33, 14, 0, -1, 2, 20, 5, 42, 0, -2, -4, 7, -29, 2, -2, 11, 10, -42, -21, 10, 7, -7, -6, 9, 1, -15, -10, -3, 9, -8, -6, -4, 7, 29, -23, -8, 26, -15, 5, -20, -7, -4, 14, -30, -5, -26, -5, 6, 15, -29, 12, 4, 1, 21, -6, -22, -14, 10, -12, -5, 3, -6, -7, 30, -5, -3, -21, -10, 8, 22, -8, 0, 8, -26, 13, -6, -23, 24, 5, 0, 6, 19, -11, -6, 3, -25, -24, 7, 21, -11, -11, -1, 13, -6, -19, -22, 34, 4, -47, 9, 19, 0, -5, 6, 12, -6, 17, 7, 8, 7, -23, -15, -35, 13, 6, 2, 13, -10, 24, 35, -31, -26, -29, 5, -7, 4, 7, -5, 11, -12, -13, 1, -9, -8, 23, -3, -4, 12, 0, -17, 0, 12, 13, 9, -8, -2, 1, 5, -7, 28, 6, -6, -28, -47, 16, 3, 3, -11, -38, -3, 3, 3, -4, -17, -2, -13, 31, -25, 2, -1, -14, -3, 9, 23, 11, 25, -7, 17, -10, -29, -35, -8, -37, 2, 23, -14, 4, -33, 7, 8, 3, 20, -15, -34, 2, 22, 13, 22, -25, -39, 27, -13, -13, -3, 1, -23, 37, -3, -31, 2, -9, 22, 16, 4, 0, -31}
, {-7, -18, -8, -10, -2, 3, -11, -4, 1, 3, 3, -45, 5, 5, -3, -3, 0, 20, 11, -6, -3, 14, -4, 10, -4, 6, 10, -9, 13, -5, 11, 23, -16, -3, 4, -25, -2, -17, 10, 2, -34, 0, -6, 8, -3, -5, -23, -18, 4, -31, -18, 8, -32, -4, -5, -1, -8, 25, 18, -15, 4, 42, -22, 5, -36, -14, -1, -9, -7, -4, -7, -12, -13, -7, 15, -4, -14, -6, 12, 0, -16, -11, 8, 0, 11, -5, -2, 19, 2, 17, 14, -21, -13, -7, -15, 6, -14, 7, -1, -5, 4, 13, 21, 0, 15, -10, 1, 6, -3, -24, 2, -35, -25, 33, 13, -4, 33, -19, -10, 5, 9, -3, 23, -6, 13, -10, 30, -11, -37, 20, 7, 45, -2, 5, 18, -24, -10, -7, 5, -7, 19, -3, 15, -2, 13, 4, 2, 1, -19, -2, 4, -15, 26, -3, -4, -6, -9, 11, -1, 3, 20, 8, 22, -17, 6, -23, 23, 10, -10, 35, 22, 6, -16, -18, -2, -5, 13, -12, -10, 21, -12, 0, 9, 9, -4, -31, -7, -15, 16, -23, -31, -14, 12, -1, -12, 13, -4, -17, 8, -3, -13, -28, -5, 23, 11, -8, -18, -2, -13, 7, -14, -20, -2, -32, -10, 5, -23, 0, -5, -1, 30, -24, 13, -11, -45, 24, -1, -9, 1, -64, -11, -30, 12, -7, 18, 9, -18, -51, 4, 25, -39, 19, -9, 16, 32, -5, -12, -17, 26, -18, 5, -22, 14, -16, -32, 17}
, {-22, 7, -22, 8, -8, 14, 0, 11, -7, -2, 13, -21, -3, -22, 4, 5, -21, -12, 12, -6, 8, -12, 13, 16, 12, -7, -3, 3, 16, -19, -5, -12, -8, 6, 28, -33, 3, -26, 10, -8, -35, -6, 10, 4, 0, 2, -3, -24, -16, -22, -9, 15, -21, 2, 4, 1, -19, -3, 5, -4, 13, 1, -3, 21, -34, -15, 0, -11, -13, -37, -10, -28, 15, -4, 9, -6, -6, 12, 10, 17, -15, 16, 24, -12, 39, -33, 8, 42, -2, -8, -18, -1, 10, 29, -14, -16, -21, -6, 2, 16, -2, -11, 16, 11, -6, -6, -8, -17, 7, 41, -12, -9, -10, -2, 15, 5, -32, -22, -3, 3, -1, 6, -2, 7, 5, -4, -4, -3, 24, 17, -6, -22, 28, 12, -8, -7, -12, -13, 8, -8, 19, 26, 5, -8, -1, 5, -15, 14, -28, -37, -45, 16, 17, 28, -11, -20, 18, 0, 3, -2, 3, 23, -6, 14, -24, 13, -5, 9, -9, 22, 12, 26, -20, -25, 11, 11, -16, -2, 16, -5, -18, 3, 21, 14, 12, -8, 10, 12, 19, -15, 7, -16, 22, -11, -30, 4, -7, 5, -25, -17, -9, 0, 17, 2, 4, 1, -13, -3, -34, 2, 7, -8, 13, -23, -13, 10, -42, -18, -7, -2, 9, -18, -2, -11, -22, -11, -9, 11, -4, -18, -4, -3, -4, -23, -13, 5, -4, -13, 5, 6, -15, 23, 10, -4, 28, 18, -18, 12, -6, 16, 1, 9, -1, 4, -10, 0}
, {1, -4, -18, 8, 8, 15, 3, 1, 41, 5, -19, -24, -13, 2, -12, 6, -11, 26, -2, -20, 19, -12, -1, 12, 7, -23, 22, 16, 17, 14, 6, 4, -14, 14, 6, -47, 2, 0, -8, 9, -55, 1, -13, 7, 7, 6, 4, 25, 15, 2, -6, -5, -9, 9, -9, -16, 23, 19, -14, 5, 11, 10, 6, 0, -27, -51, 15, -38, -26, 17, -13, -29, 10, 11, 37, 7, 12, 6, 38, -28, -32, 18, 7, -1, -21, -24, -9, 16, 2, 37, 17, -13, -23, -6, -19, -7, -31, 6, 5, -2, 0, 14, 17, 4, 16, -13, -3, 1, -28, -6, -7, -12, -11, 23, 12, 7, 28, -17, -26, 13, 16, 25, 18, 5, 11, -1, -10, 20, 20, -9, -13, 28, 13, 9, -32, 5, -9, 11, -5, -6, -4, -5, -21, 7, 9, -1, -4, -1, -24, 15, 3, -5, -22, 0, 18, 17, 11, -17, -18, -8, -10, -16, -19, 0, -20, 0, 2, -8, -5, -24, 5, 8, -4, -11, -7, -1, -4, 1, -4, 15, -6, -6, 4, -7, 0, 2, 12, 0, -2, 1, 8, -30, 4, -2, 9, 10, -13, -28, -11, -5, -11, -1, 7, -10, 9, -50, 6, 4, -23, -36, -6, 23, 33, 10, 23, -9, 10, -51, -15, -3, -2, 32, 3, -37, -22, -13, 15, 2, -43, -17, -10, -9, -1, 0, -1, -33, 0, 12, 0, 2, -3, -15, -20, -1, -3, 8, -44, -27, -1, 18, 25, -19, -19, -48, 12, 13}
, {-6, 4, 7, 20, -18, 6, -14, -18, 17, -6, -2, -1, 2, 12, -11, -5, -18, 19, -2, 10, -13, -5, 5, -2, 1, 10, -27, 18, 8, 3, -6, 12, -14, 7, 16, -10, 21, 15, 3, 1, -30, -16, 6, -9, 11, 29, -8, 25, 6, 21, 9, -7, 3, 18, 7, 8, 28, -12, 9, 17, 3, 5, 8, -10, -22, 1, -1, -25, 29, 15, -8, -30, -17, 3, 18, -3, -10, -16, 15, -23, -44, 6, -26, 10, -25, -20, 7, 21, 8, 24, -21, -21, -26, 14, -23, -37, -9, 2, -21, -18, 24, 4, 18, 14, 9, -10, -1, -6, -15, -10, -3, -6, 7, 24, 22, 17, -5, -1, 1, -2, 12, -9, 8, 13, 3, -30, 9, 15, -4, -1, -33, 4, -15, 17, -12, 0, 20, 3, -3, 10, -20, 32, -4, -6, -31, -22, -17, -12, 1, 43, -20, -9, -34, -11, 1, -7, -11, -30, 4, -17, -35, -35, 2, 6, -12, 11, 21, -32, 11, -4, -15, -14, -11, 4, -4, 18, -17, 22, 11, 4, 4, -4, -28, 17, 9, 8, 4, 13, -54, -14, 11, 9, 6, -8, -17, 17, -13, -13, -2, 19, 11, 15, 2, 2, -2, -19, 25, -12, 3, -20, 8, -27, 39, 25, -10, 15, -6, -9, 5, -6, -9, -8, -25, -14, 8, -37, 21, -34, 1, 30, 3, 25, 16, -1, -9, -7, -18, 0, -7, -16, 19, 11, -32, -24, -6, -5, 1, -1, -54, 11, -32, 10, 13, -10, 23, -6}
, {7, 2, -18, -1, 7, 4, -20, 14, 13, -5, -5, 17, 24, -14, -21, -14, 10, -19, 8, -10, 1, 4, -22, -20, -14, -12, 9, -38, -14, -3, -5, 3, -27, -2, -8, 10, -12, 6, -9, 11, 0, -4, -20, -14, -20, -8, -5, -23, -2, 2, 9, -20, 20, 2, 2, -5, 14, 23, 17, -13, 2, -15, 26, -1, 20, -10, -3, 17, 10, 13, -1, 20, 17, 23, -20, 2, 3, -9, 18, 4, 0, -30, -22, -31, 6, 36, 3, -21, 32, 6, 32, 8, -16, -2, 24, 28, 5, -38, 22, -1, -2, 16, -10, 0, -8, -6, -31, -17, 17, 1, 5, -5, 5, 16, -24, -27, 16, 17, 18, -10, 26, -1, -8, 2, 7, 0, 15, 5, -11, -25, -4, -5, -17, -36, 5, 1, -8, -4, -15, 6, -1, 8, -1, -6, 13, -10, 7, 8, 1, 18, -1, 14, 19, -19, 19, 12, -14, 32, -9, 10, 15, 20, 17, -9, 16, -17, 3, 18, 13, 0, 22, 16, -12, 26, -23, -19, 18, -25, 8, 25, 6, 7, -8, -4, -25, -12, -14, -16, -5, 7, -24, 18, 12, -23, -7, -13, -17, 3, 1, -11, 4, 19, -16, -7, 4, 47, -21, -37, 11, -31, -14, 33, -16, -10, 17, 26, -26, 5, 4, 34, -13, -1, -3, 4, -24, -8, -2, 6, 5, 18, 18, -23, -17, 20, -1, 2, -11, 2, -2, -9, -38, 19, 24, 18, -15, 2, -3, 15, 10, 22, -23, -9, -2, 24, 20, 8}
, {-25, 24, -15, -2, 31, 9, -2, 3, -15, -13, 8, 23, -7, 4, -4, 2, 5, -5, -3, 4, 3, -11, -9, -8, -12, -2, -10, -13, -2, -6, -31, -23, 5, -14, -8, 38, 8, -1, -15, -18, 17, 22, 1, -15, -15, -24, 6, -1, -27, -15, -1, 11, 10, -30, 6, -8, 13, 17, 3, 23, -20, -20, 24, 10, 16, 9, -54, 22, -11, -14, 3, -10, -22, -2, 1, -22, -37, 1, -3, 33, 13, -23, 13, 6, 21, 20, -1, -2, -25, -8, -36, -8, 16, -38, 16, 3, 11, -3, -6, 14, -6, -16, -5, -33, -4, 12, -14, -12, -18, -1, 0, 12, 2, -30, -3, -3, -37, -12, -8, 10, -3, -21, 6, 1, -17, 6, -18, 16, -6, -19, 11, 3, 29, 5, 9, -38, -20, -18, -5, -29, 17, -6, 24, -23, -3, -10, 13, 4, -22, -40, 5, 10, 10, 40, 1, -13, -25, -6, -24, 11, 28, 29, 5, -27, 9, -21, 4, 22, -23, -13, 32, 12, 9, 5, -23, -24, 5, 9, 23, 40, -6, -7, 16, -18, 9, -3, 17, -17, 9, 0, 0, -9, -10, 9, 3, 2, 3, 10, 8, -6, 2, -16, -7, 8, 3, 15, 13, 10, -20, 3, -30, -2, -21, -18, -8, -3, 4, 28, -1, -47, 2, -9, 6, -24, 1, -8, -11, 17, -7, -17, -28, -28, 18, -21, -5, 13, 3, 22, -4, -3, -19, 15, 49, 9, 18, -7, 22, 11, 16, 10, 15, 9, 14, -4, -27, 7}
, {-61, -36, 13, -23, -21, 9, -13, -26, -16, 7, -17, 0, -4, -16, 13, -29, 1, -17, -13, -16, -24, 20, -7, -1, -10, -1, 12, -8, -13, -4, -16, -8, -7, 1, -7, 22, -9, 33, -4, 4, 16, 19, 6, 8, 15, -7, -19, -54, -1, -5, 13, 15, 13, -9, 28, -9, 4, 24, -4, 23, 3, -15, 4, 11, 23, 37, -55, 16, 16, -22, 4, -8, -29, -31, -2, -3, -28, -5, 0, 23, -20, -8, -7, -2, -15, 4, -43, -15, -35, -40, -39, -23, 11, -46, 29, -28, 8, 2, -18, 5, 25, -19, -12, 6, 13, 9, -2, -27, -59, -43, 15, 11, -24, -38, -26, 15, -9, -26, -14, -11, 21, -42, -8, 7, 8, -27, -42, 12, -3, -13, -26, -13, -25, 2, -6, 6, -11, 19, 6, -7, -35, -8, -8, -15, -48, -9, -8, 16, 15, -4, 6, 25, -23, -4, 1, 7, -8, -22, -27, -27, -26, -16, -21, -13, -24, 8, -2, -15, 0, 5, 24, -5, 3, 12, -21, -14, -32, -3, 5, 21, 5, -18, -31, -7, -3, 17, 2, 1, -27, 7, 7, 18, -6, 6, 11, 7, 5, -15, 4, 13, 19, 9, -31, -21, 34, -1, 6, 16, 18, 8, 16, 1, -28, 23, -20, -12, 21, 12, 3, -17, -13, -26, -8, 3, 18, -17, -22, 0, 4, 17, 5, 14, -12, 6, -20, 7, -10, 5, 2, -10, 1, -16, -22, -44, 2, -17, 4, -11, -25, -19, 5, 5, -7, -8, 19, -45}
, {-2, 11, -6, -3, -11, 15, -7, -10, 19, -13, -5, 4, -41, 17, 10, -4, -28, -18, -11, 8, -1, -23, -5, -15, -13, 2, -27, -31, -3, 4, -7, -47, 15, -8, -2, -12, -28, -7, -8, -10, -16, -4, 3, 1, -7, -24, -4, -10, -23, -1, 12, 5, -2, -2, -29, 11, -4, 11, -1, 2, -19, -4, 10, 0, 8, 5, -6, 19, 0, -24, 13, 23, -1, -10, -39, -16, -2, -17, 2, -5, -1, 1, -17, 0, -7, 32, 2, -5, -18, -24, -16, 20, 1, 3, 12, 0, -1, 0, 1, 4, 23, -28, 6, 44, -26, 1, -8, -16, -6, 13, 3, 18, 32, -58, -14, -2, -31, -11, -3, -23, -8, -46, -37, 4, -4, 33, -25, 8, -3, 26, -30, 9, -11, 5, 19, 2, 15, -6, 13, 17, -33, -12, -14, 6, -26, 30, -3, -39, 30, 1, -14, -20, -34, -6, 3, -25, 11, -15, 13, -23, -46, -31, -8, 2, -52, 16, -19, -27, -1, -39, 1, -5, 26, -10, -15, -19, -25, 10, 4, -9, 18, -17, -32, -28, -15, 21, 3, 15, -42, -32, 14, 20, 5, 15, 19, 9, 24, 2, -5, 21, 7, 1, -21, 9, -7, -36, -8, 15, -1, 9, 2, -12, -12, 16, 3, 17, -13, 13, 11, -15, -13, -27, -11, 13, 20, -2, -6, -27, 3, 19, -13, 14, -6, -9, 22, -4, -28, -14, -11, 7, 24, -13, -1, 11, 8, -17, 21, 0, -14, -49, 16, 12, 15, -5, 0, -27}
, {-22, 19, -27, -13, -34, 2, 9, -15, 27, 17, -9, -4, -21, -13, 9, -20, -17, -17, -38, -6, -8, 11, 11, -1, -14, 2, 1, -31, 6, -16, 8, -28, 13, 20, -9, 13, -23, 25, -21, -2, 18, 19, -14, -7, -7, 1, 6, -25, 6, 17, -1, 4, 25, 23, -6, -6, -10, -6, -10, 16, -8, -18, 11, 0, 2, 2, 17, 9, 10, 0, -11, 12, 20, -7, 12, 27, 3, -5, 3, 9, 0, -3, 22, -30, -11, 36, -2, 1, 4, 16, 13, -13, 7, -2, -10, 4, -3, -23, -3, 31, 9, 19, 6, 6, 11, 0, -14, 6, -2, -2, 1, 4, 30, -1, -7, 5, 10, -9, 21, 9, -2, 0, -10, 15, 1, -12, 1, 6, -10, 3, 1, 49, 6, 19, 28, 21, -18, 26, -4, -5, 8, -7, 0, 5, 1, -33, -10, 0, 2, 21, 0, 2, 6, 15, 14, 25, -9, 3, -12, -12, -4, -8, 3, -22, 4, 15, 23, 4, -13, -15, 31, 3, -6, 17, -36, -12, 1, -33, -14, 50, 17, 11, 1, -27, -3, -1, -11, -1, -8, -38, -1, 16, 4, -39, 27, 4, -18, 15, -21, 2, 15, 3, 10, -25, 15, 7, 22, -23, 5, -41, -19, 16, 38, 4, 12, 15, 17, -24, 1, 33, -39, 22, -17, 9, -13, -22, 24, 4, -19, 5, 18, 8, -13, -9, -6, -18, 17, 52, -34, -42, 9, -14, 37, -35, -20, 23, -16, -19, -1, 35, -12, -12, 2, -4, 19, 6}
, {-17, -7, -5, 11, 9, 16, -37, -21, 1, -5, 13, 6, -13, 9, 6, -19, -4, -18, 10, 13, -1, 1, -13, -8, -17, 0, 7, -4, -6, 0, -48, -26, -8, -24, -18, 27, -2, 21, -3, -10, 20, 25, 6, -2, -9, -13, 9, 16, 0, 1, 19, 2, 17, -8, 6, -6, 26, 13, 6, 7, -4, -14, 17, -4, -13, 23, -37, 18, -7, -34, 13, 0, -27, -23, 7, -19, -2, 4, -20, 26, -1, -3, -33, 18, 5, -14, -3, -14, -36, -2, -13, -17, 5, 9, 5, 25, 29, -6, -9, 8, -5, -29, -23, 9, -17, 13, -22, -13, -3, -4, 0, -8, 9, -25, -23, 11, -25, 16, -1, -1, -20, -37, -6, 10, -17, 11, -44, -13, 3, 15, 13, -6, 4, -9, 21, -13, -35, -22, 5, -9, 17, -28, 9, 6, -1, -7, 8, -32, 16, -36, -6, -2, 4, 14, -8, -18, -27, 3, -35, 4, 8, 7, 17, -10, 7, 7, 20, -3, -9, -7, 29, 4, -2, -5, -17, -11, 12, -25, 10, 42, -20, -13, 16, -37, -5, 3, -9, -2, -25, -2, -6, 5, -7, 18, -28, 2, 20, -8, 3, -9, 0, -60, -22, 7, -23, 3, -11, 22, -10, 32, -14, -2, -27, -20, -17, -13, -8, 19, -20, -30, 23, -5, 7, -27, 7, 28, -7, 26, -1, -6, -27, -20, 15, -18, -30, 23, -11, -10, -3, 12, -14, 13, 15, 5, -8, -27, 4, 15, 5, -34, 27, 21, 21, -4, -45, -8}
, {-18, -11, -2, 25, -1, 8, -8, 11, -23, 14, 17, -13, 8, 1, 16, 2, 14, -20, 34, 7, 8, 21, 1, 6, -3, 7, 34, 2, 4, -5, -37, -6, 5, 10, -12, 1, 19, -4, 11, -11, 7, -7, 4, 13, -8, -4, 1, 5, -12, -6, -14, 5, 7, -9, 23, -16, -4, 0, -10, -3, 0, -7, -10, -21, -6, -2, -35, 14, -12, -26, 13, -16, -19, -24, 7, 12, -16, 16, -37, 16, -13, 36, 24, 10, 16, -10, -24, -5, -11, -3, -25, -10, 12, 5, 3, 2, -11, 0, 9, -11, -26, 10, 18, -24, 4, -18, 14, 9, -13, 12, -4, -5, -21, -7, -4, -28, -10, -33, 8, -1, -16, -30, -12, 8, -21, 10, -16, 3, 6, 17, 33, 19, 18, -8, 2, 20, -33, -9, -17, -7, 7, -34, 9, 17, 0, 6, -4, 9, 3, 2, 2, -17, 31, 6, 2, 1, -23, 9, -1, 2, 12, 28, 21, -6, 13, -34, -12, -3, 0, 15, -20, -6, 6, -4, -12, -12, 10, -13, -37, 13, -29, 3, 34, -35, 6, -34, 4, -8, 40, -10, -48, -2, -7, -19, 0, 9, 7, 12, 6, -20, -12, 0, -5, -3, -1, 8, -5, -19, -16, 37, -22, -6, 5, -44, 0, -24, 9, -2, -3, -25, 16, 26, 16, -52, -18, 11, 7, -10, -7, -40, 12, -3, 3, -32, -7, 2, 4, 6, -33, 4, 1, 6, -4, -9, -13, 14, 11, 7, 8, -8, -4, 12, 20, -2, -26, -14}
, {22, -21, -7, 9, -8, -33, 8, 31, 18, 16, -22, -70, -12, -4, -2, 5, 13, 22, -18, -9, -14, -24, 5, 16, 12, -39, 15, 15, 3, 3, 10, 20, 16, -6, 8, -46, -12, 5, 9, 12, -46, -3, -2, 9, -1, 15, -10, -10, 3, 0, -31, -5, -31, 22, 4, 3, -24, -12, -53, -11, 11, 6, -23, -11, -12, -35, 49, -29, 23, 23, -20, -4, 41, -1, -1, 19, 30, -12, 0, -36, -27, -7, -8, 1, -7, -13, -8, 6, 48, 24, 36, -14, -6, -26, -35, 0, -34, 3, 18, 10, 1, 16, 4, -23, 10, -21, 6, 26, 6, -13, -10, -20, -25, 18, 22, 0, 6, 8, 22, 0, 25, -1, 15, -4, 10, 2, 17, 10, 15, -32, -13, -19, 2, 8, -12, -4, 4, 5, -6, -4, -2, 17, -15, -34, 1, 5, -1, 20, 7, -28, -10, -12, -8, 14, 12, -4, -14, 11, -2, 1, -1, -15, -45, -19, -13, -28, -2, 8, -16, -1, -10, -7, -4, 39, 2, 7, 26, 16, -5, -5, -1, 5, -20, 13, 16, 5, 7, -9, 12, 7, -7, -27, 25, -7, 5, -11, -13, -16, -1, 22, -28, 1, 14, -15, -25, -26, -6, -2, -8, -31, -22, 16, 30, -29, 17, 4, -12, -28, 13, -12, 1, 10, 21, 23, -18, 0, -5, 4, -60, -1, -1, -51, 3, 15, 14, -31, -1, 13, 6, 8, -26, -9, 22, 0, -11, -5, -58, 3, 15, 13, 9, -54, 4, -25, 26, 2}
, {-51, 1, 17, -12, -12, 17, -8, 2, -29, -6, 5, 22, 19, 7, 7, -5, 10, 6, 2, -14, -3, 1, -13, 4, -5, 13, 17, 7, -26, -1, -36, -12, -1, 7, -12, 35, 6, 11, -2, -3, 26, 25, 11, 2, -7, -18, -11, -55, 17, 16, -8, 11, 10, -27, 7, -8, 12, 3, 16, 17, 0, -30, 14, 8, 28, 43, -40, 14, -3, -38, 5, -10, -20, -43, -4, -14, -45, -7, -28, 25, 0, -3, 11, 11, -1, 5, -32, -17, -35, -26, -48, -2, 37, -43, 19, -3, 24, -16, -22, 4, 32, -8, -11, -14, -8, 41, -11, -19, -41, -26, 19, 30, -26, -44, -21, -1, -58, -27, -3, -7, -5, -38, -19, 5, -33, -12, -35, 7, -26, -13, 19, -7, -4, 8, -4, 10, -18, -26, 3, -15, 11, -2, -3, 8, 6, -1, 9, -15, 9, -39, -16, 18, 12, 12, -10, -3, 6, 3, -16, -2, 27, 3, -9, -9, 12, -4, -7, 6, -4, -2, 19, 10, -5, 1, -3, 10, -11, -3, 1, 5, -8, -5, 10, -10, 10, -4, -1, -1, 4, 21, -12, -4, -8, 1, 8, 14, 25, -17, 2, -17, -16, -6, -4, 0, 31, 6, 5, -3, -8, 16, -22, -9, -11, 12, 20, -5, 20, -2, -12, -34, 6, 11, -10, -23, 9, 4, 0, -4, -18, 4, 0, 4, 4, -24, -12, 3, -1, 14, -6, -11, -8, -3, 23, -29, 6, 6, 5, -9, 8, 9, 7, -5, -22, -5, -1, -19}
, {-24, 7, 1, 2, -7, 4, -6, 4, -11, 1, -9, -6, 6, -13, 19, 9, 3, -9, 7, 2, 8, 6, 6, -2, 10, -3, 1, 6, -10, 14, -12, 23, 11, 28, 12, -15, 4, -24, 17, 1, -8, 13, -8, -2, 0, 10, 0, -1, -7, -18, -9, 8, -4, 2, 13, -14, -17, 7, 4, -22, 8, 21, 6, 8, 6, 16, -39, 5, 14, -51, 25, -16, -14, -46, -3, -7, -8, 15, -10, 10, -2, 26, 54, -1, 53, 3, -4, -19, 0, -20, -22, 6, 6, 0, 19, -4, -18, 6, 9, 4, 3, -24, -7, -6, 18, 7, -5, -19, -24, 3, 2, 3, -20, -45, -12, -1, -26, -28, -22, -5, 4, -46, -11, -10, -23, -17, -25, -5, -27, 13, 33, -25, 17, 4, 0, -43, -3, -24, -10, 9, 23, -9, 11, -1, 3, 17, -15, 35, -18, -16, 11, -1, 3, 8, -25, -3, 29, -17, 4, 13, 12, 12, 1, 19, -14, 0, 2, -6, 8, 32, 3, 16, 9, -9, 24, -10, 0, 4, 14, -16, -22, 0, 10, 12, -5, -29, 3, 3, 29, 12, 11, 0, -11, -25, -16, 8, -3, 8, -30, -16, 8, 12, -8, -25, 3, 27, -21, -9, 0, 0, 18, -34, 10, -20, -18, 3, -16, -9, -1, -22, -4, -27, -38, -6, -4, -10, -11, -1, 10, -2, 24, 0, -10, -37, -17, 9, 7, 20, 5, -5, 0, 4, 16, -7, 27, 7, 8, 24, 6, 6, 7, 10, -24, 19, -30, -32}
, {4, 19, -21, -23, 6, 6, -5, -12, 20, 2, 1, 19, -37, -22, 0, 5, 4, 7, -6, 3, 15, 28, -35, -9, -24, -3, -17, -10, 10, -18, 7, -22, 8, 0, -16, 38, -11, 35, -2, -9, 3, 19, 4, -5, -27, -36, 17, 32, 20, 8, -2, -4, 25, 3, -6, -7, 4, 0, -3, 47, -3, -19, 35, -4, 5, 9, -15, 26, -4, 8, -9, -1, 0, 11, 23, -12, -23, -12, 10, 9, -7, -12, -18, -11, 5, 27, 4, -16, -18, -4, 7, 2, -6, -14, 7, 5, -4, -11, -11, 7, 11, 11, 14, 15, -12, 8, -12, -31, -3, 15, -6, -3, 33, 39, -10, 7, 15, 24, 26, -6, -6, 23, -10, 8, -21, 6, -9, 4, -18, 0, -31, 25, -2, 3, 15, 14, -25, 38, 24, -16, -25, -18, 1, 0, -36, -37, -26, -16, 11, 8, -6, 0, -5, 14, 16, -8, -5, -27, -38, -51, -16, -39, -1, -9, -21, 20, 16, -26, -3, -23, 33, -23, 5, 17, -37, -5, -24, -12, 4, 11, 7, 1, -13, -36, -8, 25, -6, -20, -34, -41, 14, -8, 9, -6, 33, -13, -4, 5, 12, 8, 12, 13, -29, 16, 42, -8, -1, -10, 2, -18, -18, 10, 2, 6, 20, 15, 14, 8, 5, 9, 2, 1, 5, -11, 5, -12, 22, 27, -7, 1, -8, 5, 1, 11, -5, -9, 0, 12, 16, 2, -1, 9, -1, -10, -14, -1, -5, 7, 5, 5, -1, 10, 17, -7, 0, 30}
, {34, 16, -54, 14, 14, 3, 2, -7, 5, -5, 12, -4, -17, -1, 1, -8, -1, -1, -17, 11, 12, 32, -12, -10, -31, -7, 2, -17, 0, 4, 12, -29, -8, -19, -23, 22, -29, 30, 4, -6, 9, 22, -4, -26, -15, -28, 11, 29, -13, 14, 13, -18, 9, -11, -39, 1, 49, -10, 5, 16, -33, -32, 25, 8, 15, 16, -7, 29, 17, -9, -19, 27, 16, 8, -15, -21, 4, -8, -5, 1, 17, -13, -15, -3, 15, 56, 8, -27, -15, -3, -25, 18, -6, -4, 11, -25, 12, -13, -13, 5, 9, 7, 11, 8, -23, -13, -11, -5, 12, 30, 15, 11, 45, -23, -21, -4, 9, 46, 14, -13, -9, -10, -23, 3, -13, 12, -5, -12, -32, -6, -7, -12, -29, 2, 47, 19, -11, 18, 0, 5, -22, 20, 4, -11, -10, -31, -17, -2, -12, 12, -7, 8, -5, 8, 0, 9, 11, -7, -12, -10, -17, -12, 21, -5, -15, 8, 5, 0, 1, -5, 21, -1, -7, 11, -22, -15, -30, -5, 18, 9, 9, 8, -24, -24, 10, 11, 7, -9, -16, -7, 10, 13, -3, 0, 24, -19, -2, 7, 8, -4, 15, 12, -10, -17, 11, 32, 18, -5, 3, -23, -11, 11, -14, 8, 2, 47, 1, 14, 2, 10, -9, -22, -7, 14, 6, -6, 20, 14, 12, 1, 17, 2, 1, 5, -13, 11, -2, -3, 24, -1, 14, 21, -5, -9, -17, 22, 18, 33, 2, 7, 22, 8, 49, 11, 29, 4}
, {53, -8, 0, 43, 11, 0, -3, 23, 8, -7, -1, -2, 20, -6, -17, 15, -11, -2, -19, -20, 4, 26, 10, -3, -18, -28, 22, -6, 9, -5, 23, -11, -18, 13, 11, -21, -6, 8, -33, -9, -18, 0, -17, -22, -1, 15, -3, 12, -6, 5, -4, -11, -2, 11, -1, -13, -17, -14, -26, -11, -10, -15, 17, -13, 20, -2, 34, -3, 11, -1, 6, 8, 9, 9, -3, 13, 20, 18, -6, 15, 13, -7, 46, -15, 28, 16, 19, -8, 8, 1, 33, 11, 16, 7, -12, 4, -10, -11, 26, 20, -47, -2, -16, -10, -2, -15, -16, -4, 4, 8, -26, 12, -5, -13, -17, -26, 14, 35, -14, -21, -7, 12, -30, -20, 11, -9, 9, -14, 4, -49, -1, 21, 16, -17, 18, -3, 20, 13, -3, -14, 8, 22, 10, -15, 6, -4, -5, 51, -15, -5, 10, -4, 11, -6, 8, 11, 14, -4, 2, 16, 5, 15, 2, -13, 0, -11, 18, 15, -28, 28, 4, -4, 6, 10, 5, -14, -7, -6, 4, -27, 4, 21, 13, -6, 4, -3, -1, 2, 25, 23, 0, -3, -13, -47, 9, -22, -9, -7, -26, 0, -1, 2, 17, -21, 8, 26, 16, -16, -6, -9, 4, 33, 2, -23, 7, 32, 22, -34, 0, 4, 4, -8, -14, 31, -13, 10, -31, 3, -2, 4, 7, -2, -15, 10, 6, 16, 10, 48, 8, -9, 8, -16, 19, 5, -35, 5, -18, 45, 7, 13, 1, -12, -25, 20, 6, -14}
, {-26, -18, -10, -4, -10, 30, -10, 16, -7, -3, -1, 10, 26, -23, 6, 18, -3, -33, -14, 9, -3, 26, 1, -6, -8, -22, 17, -39, -9, 5, 7, 7, 7, -26, 14, -5, 7, 14, 0, 11, 13, 9, -11, -25, 9, 10, -28, -46, 6, -14, 4, 0, 18, -6, -12, -9, 16, 10, -7, -12, -13, -42, 12, 28, 0, 3, 9, 7, -8, 12, -3, 14, -14, -4, -1, 2, 1, -12, -11, 16, 17, 0, 20, -3, 7, 31, -3, -28, 2, 14, -4, -19, -1, 5, 14, -23, -28, -9, -5, 5, 11, 21, -28, -8, -3, -1, -7, -14, -8, -21, 1, 15, -13, -2, -17, 2, 8, 13, -8, -20, 10, 17, -4, -18, 13, 1, 7, -8, -13, -61, 6, -8, 8, 3, 23, -14, -9, -28, -64, -44, 11, 14, 13, -33, 4, 13, -11, 12, -30, -16, 13, 6, 18, 6, 1, 19, 0, -12, -29, 18, 10, 4, -15, -38, 15, 16, 6, 8, -18, 9, -8, 6, -11, 22, -16, -13, 3, -33, -7, 16, -21, -29, 10, -8, -27, -32, -14, -26, 23, 9, -7, 14, 2, -12, -26, -18, -25, -22, 14, -39, 24, 6, -42, -7, -3, 31, -3, -25, -17, -6, 20, 8, -32, -3, -10, 7, -8, -8, -31, -18, 16, -20, 0, -9, 7, -30, -22, 25, 5, 4, 1, -22, -7, 33, 13, -3, 3, -22, 2, 19, -30, -16, 0, 17, -31, 7, -13, 2, 20, 23, 16, -1, -29, 18, -20, 29}
, {-32, -8, -7, -9, 11, 12, 13, 24, -31, -29, -8, -3, 6, 22, 11, -13, 8, 24, 19, -12, 14, -6, -21, 3, 4, -5, -27, -12, -14, -12, -25, 10, 7, -8, 0, 2, -6, -2, 4, -8, -3, -24, -4, 5, -9, 7, 0, -36, 11, -12, -2, 6, -15, -30, 13, -8, -22, 13, 4, -5, 11, -29, -1, -3, -6, -4, -15, -8, 10, 9, 1, -13, 8, -1, -4, 6, -47, 2, -20, 1, -6, -2, -43, 1, -14, -6, 9, 6, -21, 17, 15, -15, 7, -24, 5, 13, 3, 19, -16, -3, -28, 0, -11, -39, 12, 1, 19, 7, -8, -21, -3, 4, -41, 10, 15, -6, -10, -18, -1, -2, -16, -7, 6, -4, -15, 3, 5, 4, -14, 31, 13, -26, -34, 4, -7, -12, -7, -20, 44, 17, 6, -32, -11, -11, -20, -22, -19, -26, 13, -21, -20, -20, 5, -10, -7, -18, -15, -5, -4, -20, -15, -26, -3, -3, -13, -7, -31, 3, 9, -41, -20, -1, 21, -11, 1, 6, 6, 11, -8, -17, 30, 9, -25, -35, 3, 29, 18, 15, -7, -53, 16, -7, 12, 21, -27, -11, 16, 1, 2, 16, -22, -28, 2, -6, -13, -4, -45, 24, 2, 15, 0, -8, 13, -14, -25, -24, -9, 14, 1, 5, 17, 0, 24, 5, -9, -10, 12, 33, 12, -11, -57, -37, 2, 4, -17, -34, -28, -18, 10, 35, -8, -38, -21, -5, -18, -24, -7, -10, 2, -25, 3, 3, -20, -13, -29, -8}
, {-5, -2, 4, 11, 4, -32, 4, 15, -34, -7, -19, -35, 4, -24, -4, 8, -20, -6, 4, -13, -17, -14, 18, -17, 4, -32, 0, 4, -2, 9, 10, 16, -5, 2, 15, -63, 15, -26, -6, -7, -39, -22, -21, -15, -6, 9, -20, -8, -8, -2, -30, -6, -26, 10, -7, -39, -5, 0, -42, -41, -5, -12, -28, 6, -10, -8, 14, -21, 14, -5, -15, 0, -4, -6, -1, 18, 19, -1, -10, -36, -11, 35, -14, -1, 4, -9, -3, 4, 3, -12, 13, 10, -12, 11, 0, 7, -22, 1, -27, -12, 10, 17, -10, -21, 3, -23, 3, 40, 15, 0, -7, -20, 15, 24, 24, 2, -25, 5, 3, 8, -14, 6, -8, -3, 12, 1, 12, 1, -17, 14, -23, -35, -9, -3, 0, -29, 17, -37, 20, 6, -19, 9, 4, 14, -24, -1, -8, 2, 17, -1, -16, -13, -41, 12, -26, -1, 7, -15, 9, -15, -14, -32, -23, 6, -33, 21, -11, -8, 13, 15, -25, -3, 5, 10, -5, 21, -22, 25, 30, -38, -6, -18, -19, 13, 17, 7, 10, 18, -2, -4, 7, 6, 31, 0, -41, 14, -2, 8, -25, 0, -9, 18, -3, 9, -36, 10, -31, 7, -8, 10, 30, -40, 8, -2, -7, 13, -33, -7, 22, 8, 1, -26, -1, 17, -7, -30, 1, -2, 33, 20, 1, -1, 14, -16, -32, -26, 4, 17, 1, -15, 14, 20, -38, 5, 13, -13, 3, -12, -37, 22, -41, 0, -11, -3, 23, 5}
, {-3, -30, 5, 11, -16, 3, -14, 7, -5, 16, 0, -10, -3, -34, 11, 12, 12, -28, -19, -8, -45, -2, 27, 0, 8, 14, -4, 0, -29, 13, -14, -6, 1, 9, 14, -22, 5, 6, 12, 5, -13, 11, 14, 16, -9, -1, -13, -19, 5, 3, -5, -1, 11, 4, 9, -18, -15, -1, 7, -13, 10, 5, 3, 8, 13, 15, -3, -22, 14, -11, 1, -29, 2, -31, 6, 4, -4, 0, -12, -1, -22, -10, 30, -10, -4, -22, -56, 6, -7, -11, -6, -11, 8, -2, 5, -11, -6, 2, 15, 2, 11, -16, 4, -7, -7, 4, -2, 9, -23, -6, 14, 14, -21, 3, 4, -12, -14, -27, -1, 2, 11, 25, -17, -6, 2, -26, 3, 2, 5, 3, 20, -49, -5, 4, 1, -37, 22, -12, -42, -9, 24, -7, -13, -37, 4, 7, -8, 24, 16, -36, -30, 4, -30, 6, -3, 0, -6, -23, 9, 22, -15, 19, -72, 19, -20, -7, -31, 1, -5, 0, -22, -5, -31, -21, 14, 3, -11, 16, 0, -54, -5, 29, 6, 6, 2, -22, 6, -4, 2, 29, -5, 9, -41, -35, -8, 4, -4, -19, -16, 13, -46, 29, 4, -25, -4, 8, -7, -6, 6, 11, 11, -10, 5, -7, 17, -9, -17, -25, 17, -28, -18, -17, -41, 19, 8, -21, -3, -31, 1, 12, 11, -1, -7, -13, 22, -1, 5, 30, -25, -46, 8, -30, 20, -13, 5, -12, 28, -8, -32, 18, -20, 0, -42, 8, -8, 9}
, {-17, 5, -16, -2, 6, 10, 0, 17, -11, -28, 5, -1, -12, -10, 19, 0, -26, -10, -3, 5, 8, 10, -28, -7, -20, 4, 37, -4, 16, -16, -24, -13, 1, -5, -2, -1, 22, 1, -1, -29, 19, 13, 3, -13, -14, -21, -1, -17, -1, -16, -15, 17, 10, -31, -16, -2, -13, 28, -2, 3, -1, -29, 11, 0, -7, 7, -8, -1, -10, -14, 18, -6, -21, -16, 14, -6, -30, 21, -10, 3, 4, 27, -6, 14, 15, -22, -14, -16, -20, 14, -27, -16, 9, -11, 1, 8, -9, 15, 8, -2, -7, -12, 0, -37, 12, 5, 3, -9, -16, -11, -2, -5, -34, -4, 6, 6, 4, -29, -15, 0, -12, -29, 8, -10, -13, 15, -34, 9, 3, 8, 6, 3, -4, -9, 16, -35, -31, -19, 3, -6, 15, 10, 19, -15, -29, -25, -12, 4, -17, -28, 0, 11, 8, 21, 1, 10, -55, -1, -17, -22, 2, 26, 6, -26, 7, -2, 15, 2, -15, 2, 11, 0, -2, -4, -26, 10, 1, 2, -6, 30, -3, -31, 2, -13, 12, -2, -1, -21, 19, -33, 6, -24, -1, 3, -21, -8, 14, 8, 3, -16, -13, -21, -8, 13, -14, 24, -26, 8, -12, 7, -3, 18, 7, -25, -18, 0, 2, 14, 18, -18, 21, 1, 12, -25, -9, 6, -2, 50, 14, -50, -16, -36, 13, 11, 2, 3, -13, -30, 9, 40, -32, 11, -2, -4, 32, 9, -12, 0, 34, -8, 14, 1, -33, -9, -36, 13}
, {-67, -41, 14, -43, -48, -25, -4, 3, 20, 1, -11, 0, -5, 5, 6, 18, -24, -16, -19, 2, 5, -50, 19, 11, -1, -30, -24, -16, -15, 8, -1, -28, 3, 15, 13, 1, 1, -3, -5, 4, -43, 6, 6, 3, -12, 0, -16, -50, 12, -10, -4, 13, -38, 7, -42, 0, -28, 37, -17, 7, 5, -18, -13, 18, 17, 16, -11, 26, 11, 20, 10, 34, 6, -57, -32, 2, -27, -28, 6, -5, -9, -14, -8, -14, -53, 6, -56, 4, -27, -25, 4, -6, -2, -40, 27, 10, 6, 16, -7, -4, 1, -7, -4, 17, -5, 16, 11, -13, -48, -77, 9, 0, 1, -12, 10, 14, -13, -50, -6, 7, 2, -28, 11, 1, 6, -6, -9, 9, -22, -12, 1, -24, -14, -22, -1, 23, 9, -2, -16, -9, -9, -2, 1, 12, -29, 5, -16, -19, 14, 2, 16, -7, -3, -16, 0, -25, 5, 9, 16, -21, 1, -24, 17, -3, -33, -3, 2, 18, 20, -24, -4, -11, -4, 6, -6, -9, -20, -13, -13, -22, -18, -1, -40, -4, -37, 18, -4, 7, -16, -8, -1, -2, -6, -12, 19, 4, 22, 6, -11, 25, 10, -4, -29, -21, 25, 9, -16, -5, 12, -7, -23, -12, -7, 13, -7, -44, -15, 3, 22, 1, -31, -19, -17, 22, 23, 10, -3, -29, 6, 17, -24, 3, -33, -3, 4, 0, -20, -23, -11, -7, 12, -18, -39, 0, -9, 1, 6, -41, -17, -39, -33, 8, -15, 4, 1, -42}
, {-10, 12, -8, -25, -15, 26, 31, -50, 27, 2, 12, -22, -24, -28, -9, 3, -43, -26, -14, -8, 18, -30, 5, 8, 4, -14, -20, 23, 5, 7, 36, -23, -15, 7, 5, -2, 8, -23, -3, 15, -16, 6, 1, -1, -5, 17, -9, 31, -1, 2, -15, -7, -20, 24, -19, 6, 18, -5, -19, 4, -3, -18, 7, -5, -44, -9, -4, -33, -26, -14, -13, -41, 2, 10, 45, 7, -52, 4, 18, -11, -43, 14, 16, -23, 2, -19, 14, 14, -25, 4, -42, 10, 2, -4, -28, -18, -24, -4, -54, 8, 6, 10, 30, 3, -4, 4, -12, -18, -1, 35, -13, -26, -10, 7, 17, 24, -13, -11, -13, 15, -8, 17, -1, 19, -27, -17, 3, 15, 20, 3, -12, -10, 19, 14, 6, 14, 10, 20, 3, -15, -1, 23, -8, 13, -3, -12, -13, 15, 7, 35, -6, -8, -18, 24, 1, -20, -13, -10, -2, -13, -17, -21, -1, -6, -33, 0, 8, 7, 9, -20, -9, 3, -9, 2, -6, 4, -31, 14, 14, -11, -3, -1, 19, 16, -3, 4, -2, -4, 22, -6, -8, -32, 3, -22, 8, 6, -24, -18, -19, 12, 15, 14, 5, -9, 2, -22, 17, 1, -31, -2, -9, 10, 13, 12, -5, -16, -7, -15, 2, -9, -16, -21, -1, -37, -4, -21, 0, -12, -23, -9, 18, 19, 4, -24, -30, -17, 15, 15, -6, 11, 7, -10, -18, -9, 3, 11, -31, -2, 6, 3, -10, -46, -11, -9, 22, 4}
, {33, 13, -27, 18, 0, 15, 5, 41, -20, -24, 2, -18, 4, 5, -22, 8, 15, 16, 15, -22, 10, -11, -18, -11, -24, -30, 26, -6, 13, -26, -4, 19, -31, -17, 2, -20, -16, 12, -5, -15, -21, -9, -23, -13, -3, 1, -18, -9, -3, -15, -16, -27, -11, -15, -26, -14, -7, 18, -17, -8, -15, -12, 6, -20, 19, -6, 33, 1, -2, -6, 4, 5, 35, 12, 12, 16, 20, -2, 13, 4, 13, -11, -4, -9, 34, -2, 16, 1, 34, 31, 16, 13, 6, -1, 4, 20, -16, -4, -14, -12, -24, -2, 2, 2, 10, -38, -14, 1, 29, 16, -18, -8, 20, 25, -1, -22, 6, -1, -16, -25, -5, 26, -16, -18, 4, 10, 18, 0, -54, -12, 14, 4, 5, -34, -2, 6, 9, -8, 5, -7, 5, 11, 0, -11, -8, -16, -11, 10, -16, -10, -5, 13, 13, -2, -25, 26, -2, 9, -13, -13, 15, 25, 0, -19, -1, 15, 25, 14, -8, 7, 26, 13, -4, -19, -2, -19, -2, -17, 22, 9, -16, -4, -4, 18, -10, -29, -6, 1, 30, -16, 1, -4, 11, -19, -23, 5, -2, 14, -5, -9, 13, 31, -8, -1, -11, 34, 1, -2, 0, 4, 0, -6, -31, -6, 10, 15, -21, 2, 8, 26, -25, -23, 3, 19, -34, -40, 2, 4, -2, 10, 17, -15, 7, -3, -41, -20, -13, -1, 8, 7, -6, 25, 11, 9, -5, 13, -6, 3, 9, 41, 12, -6, -27, 18, 6, -9}
, {-16, -20, -4, -23, -2, -8, 17, 11, 1, 13, -7, -58, 2, 5, 2, -23, -13, 14, 9, -23, 15, -10, -1, 15, 20, -42, 3, -1, 10, 5, 11, 5, -4, 9, 2, -42, 2, 0, 12, 8, -71, -24, 0, 11, -6, 12, -5, -22, 10, 2, -7, 4, -19, -13, 2, -12, -9, 21, -26, 6, -2, -18, -19, -4, -4, -17, 15, -39, 8, 9, -15, -6, 40, -27, 5, 31, -30, -12, 8, -24, -29, -6, -6, 18, -30, -34, -37, 22, -19, 27, -4, -19, 1, -49, -9, 13, -11, 19, -21, -2, 16, 8, 24, -5, 30, 8, 16, 8, -29, -30, 17, 2, -31, 42, 20, -1, 23, -37, -5, 16, 8, -6, 2, 2, 6, -4, 10, 8, 23, -7, 30, 18, 21, 2, -15, 2, 1, 8, -18, 2, -15, -1, -6, 0, 3, 11, 15, 16, 1, -7, 14, -7, -12, 6, 2, 0, 19, 27, -14, 14, -5, -3, -22, 2, 15, -8, -27, 24, -10, -9, -34, 6, -11, 11, 16, 2, 30, -12, 0, -8, 1, -10, -7, 2, -3, 7, 2, 12, 4, -7, 6, -10, 29, -5, 2, 5, 6, -4, -1, 33, 1, -7, 6, 7, -18, -33, -1, 0, -9, -23, 3, 4, 39, 3, -7, -3, -4, -20, 8, 10, -9, 30, 8, 13, -34, -19, 0, -17, -43, 7, -7, -24, -8, -5, 6, -27, 6, 4, -15, -13, -2, -1, -8, -13, -23, 11, 1, -43, -18, 9, -10, -14, -37, -8, -21, 15}
, {-42, 21, -36, -16, 17, 19, 16, 14, -31, -23, -20, -6, 11, 20, 7, 3, -6, 30, 27, 20, 12, 4, -26, 4, -10, 18, 9, -14, -2, -20, -14, -1, 0, 14, -29, 6, 2, 0, -3, -5, 23, 2, 4, 26, -18, -11, -15, -17, 0, -32, -3, 16, -9, -25, 1, -12, -7, 36, 9, 2, -14, -15, 2, -16, -16, -4, -10, -8, -7, 8, 7, 5, -2, -11, -7, -1, -28, -4, -19, 13, 1, 1, 1, 30, 2, -24, 14, -4, -27, 18, -9, -14, 9, -6, -10, 10, 16, 15, 22, -7, -30, -13, -13, -51, 19, -14, 9, 2, -30, -10, 4, -12, -16, -17, 3, -16, 1, -34, -4, 7, -8, -3, 8, -2, 3, 13, -24, -14, -1, 14, 21, -5, -17, 12, -2, -10, -42, -2, 18, 12, 6, -33, -8, 3, -10, -52, -27, -3, 10, -34, 0, -3, -3, 10, -4, -6, -21, -9, 2, -40, 16, 37, 23, -3, 1, -17, 24, -7, 9, 24, 23, -4, 13, -19, -27, 4, -16, -4, 4, 39, 14, -3, -3, 2, 13, 8, 5, 14, 13, -48, 6, -1, 17, 11, 2, 7, 11, 4, 11, -9, -4, -52, 6, 5, -5, -7, -13, 11, 2, 19, -3, -8, 13, 7, -23, -1, 18, 20, 10, 0, 22, 0, 39, -11, 2, 3, 13, 41, 23, -35, -27, -28, 8, 7, 9, -19, -7, -30, 14, 41, -13, -1, -16, 10, 21, -25, -9, -6, 22, -28, 1, 15, -6, -4, -43, 1}
, {-43, -14, 2, -19, 10, 8, -10, 25, -25, 4, 0, -8, 26, -5, 6, 9, -12, -13, -24, -2, -13, -6, 16, 3, -1, 11, 26, -11, -5, -2, -22, 12, -6, -6, 22, -10, 21, 0, 0, 0, -9, -1, 10, 14, 1, -14, 5, -44, 0, -13, -9, 3, -3, -14, 15, -3, -11, -7, 5, 0, 9, -15, 8, 14, -1, 4, -2, -6, 20, 6, 7, 3, -15, -38, -2, -10, -47, 16, 1, 27, 15, 7, -1, 29, -8, 14, -36, -11, -21, 23, 1, -21, -2, -52, 19, 6, 1, -9, -3, -1, 4, -1, 17, -1, 39, 14, 21, -1, -36, -46, -7, 7, -31, 38, 1, -3, 18, -35, -14, 3, 9, -28, 23, -1, -1, -14, 52, 6, -3, -2, -11, -10, -19, 18, -10, -35, 1, -26, 1, 1, 27, -8, -2, -15, -2, 18, -3, 10, -14, -26, -27, -9, 0, -5, -17, 40, -3, -20, -8, 12, -22, 8, -22, -9, 18, -7, -4, 4, -18, 7, -8, 11, -25, -11, 3, 6, -16, -6, 16, 0, -1, -11, -8, 10, -8, -8, 22, -7, 15, -10, 5, -5, -4, 14, -36, 6, 5, -12, 36, -5, -14, -7, -2, -4, -9, -11, -44, 7, -6, 9, 10, 23, -2, -2, -2, -8, -19, 5, 6, -29, 45, -3, 18, -12, -1, 11, -20, 34, -8, -12, -13, -26, 21, 38, 30, -8, 3, -3, -1, 37, -27, -17, 11, 18, -4, -7, -2, 6, 39, 23, 25, 2, -46, -1, -19, 52}
, {-29, 29, -6, 16, 3, 4, -11, -30, -11, 7, 22, 6, 4, 6, 0, -8, -12, 8, -1, 2, -14, 35, 1, 0, 9, 11, -19, -5, -3, 4, 40, 5, 9, 3, -3, 9, 12, 24, 2, -7, 15, 11, -7, 2, 19, -2, -6, 27, -12, 6, 10, 0, 6, 3, 4, -3, 12, -4, 10, 0, -8, 0, 22, 15, 18, 3, -4, 18, 20, -18, -8, 23, 17, -11, -6, -25, 16, -9, -15, 13, 16, -14, -20, -4, 1, 29, 4, -17, -12, -27, -5, 14, 1, 12, 8, 1, 17, -9, -17, -15, 12, -6, -9, 32, -12, 3, -4, -28, 35, -16, 4, 16, 21, -38, -25, 15, -17, 6, -16, -16, 5, -6, -14, -13, -12, -8, -21, 1, -21, -8, -16, -20, -25, 4, -7, 27, 0, 1, 15, 8, -8, 14, 12, 8, -10, -26, -19, 9, -16, 13, 13, 9, 6, -1, 1, 0, 7, -11, 6, -22, -3, -26, 3, 7, -19, 23, 4, -10, -6, -10, 1, 4, -12, -24, -13, -15, -19, -1, 12, -6, 26, 7, -1, -7, 1, 10, 3, 13, -9, -3, 5, 10, 2, -2, 1, -8, -2, 30, -34, -16, 33, 55, -25, -23, 22, 9, 7, -7, 13, -4, 36, -21, -9, 3, -26, 14, 3, -4, 7, 33, -16, -25, -34, 27, -3, -48, -4, -7, 7, 6, 46, 32, -14, -25, -27, 3, 0, 1, -5, -26, 16, 9, -7, -26, 7, 7, 13, 17, -22, -24, -15, 5, 3, 11, 39, -30}
, {-15, -2, 8, 11, -24, -19, 10, -26, -45, 4, -5, -22, -6, 2, 2, 8, -16, -6, 7, -27, -10, -7, -7, 7, -1, -14, -31, -4, 1, -2, 4, 5, 0, 4, 14, -14, -9, -5, 18, 6, -22, -9, 1, -11, -13, 1, -2, 12, -10, -3, 21, 17, -22, 16, 2, -1, 9, 1, -16, 0, -11, -2, 8, -2, 11, -8, 16, 3, 19, -10, 3, -3, 8, -28, 8, 2, 18, -5, -29, 1, 3, -5, -16, -6, 8, -30, -5, -9, -3, -11, -22, 19, -9, 18, -2, -17, -7, -1, -36, 0, 16, -10, -3, -2, -4, 8, 5, 3, -1, 5, -2, 11, 16, 19, 4, 18, 13, 3, -8, -12, -3, -19, -15, -21, 12, -16, -10, -5, -26, -4, -33, -26, -29, -8, -14, 0, 21, -34, -14, 9, -3, 15, -26, 12, -14, -14, 9, 23, 13, 19, -4, -4, -16, -22, -12, 12, 7, -12, 2, -20, -28, -26, -19, 27, -3, 28, -20, 18, 9, 11, -32, 0, -26, 11, -4, -9, 4, 6, 38, -11, 29, 6, 14, 9, -33, 11, -19, 9, -42, -3, 13, -6, -10, -18, -4, 5, -1, 2, -49, 6, 14, 15, -7, -9, -18, 24, 5, -2, 3, 6, 25, -17, 22, 21, -7, 27, 7, -12, 13, -2, -33, -2, -36, 39, -12, -27, -20, -16, 13, 34, 34, 29, -19, -7, -19, -8, 12, 16, -20, -33, 39, 16, -8, -18, -4, 17, 16, 22, -46, -5, -8, 15, 13, 14, 28, -18}
, {29, 13, 11, -8, 20, -1, -5, 25, -32, -31, -10, 0, 14, -1, 3, 17, 10, -1, 22, 15, 0, -7, -6, -5, 1, 23, -4, -1, 10, 13, -21, 30, 21, -21, 3, 0, 35, -9, -2, 1, 14, -13, 15, 24, 1, -9, 5, 12, -8, 3, -4, 18, 0, -18, 16, 2, -23, -15, 8, -14, 12, 19, -5, 0, -23, -2, -19, 0, 8, -3, 4, -2, -34, -10, -46, 4, 3, -4, -7, -19, -12, -1, -22, 11, -17, 2, -6, -14, -14, 7, -19, -3, 8, 20, -5, 11, 5, 17, -7, -7, -25, 7, -2, -24, -5, 1, 5, 7, 7, -5, -6, -4, -30, -19, 12, 1, 2, 1, -4, -6, -2, -27, 18, -2, 14, 19, 3, 4, -13, 36, 11, -12, -21, 22, -15, -36, 19, -37, 42, 34, -1, -22, 6, -2, -6, -16, -11, -24, 8, -7, -13, -20, 1, -5, -13, -50, -38, 7, -8, -37, 11, -8, -3, 12, -14, 23, -19, 2, 28, -1, -18, -7, 23, -21, -6, 9, 15, 37, -7, -23, 5, -33, -1, -17, 29, 13, 2, 29, -11, -55, 32, -5, -13, 22, -52, 0, 17, -12, 23, 2, -21, -37, 1, 5, -23, -19, -33, 29, -12, 23, 13, -19, 6, 8, -20, -25, -7, 14, 2, -53, 7, -9, 21, -28, -23, 5, 12, 7, 26, -11, -38, -12, 7, -10, -12, -21, -39, -20, -14, 16, 9, -34, -19, -8, 17, -43, 7, -26, -3, -25, -7, -10, -36, -3, -27, -2}
, {1, -14, 4, -1, 16, 22, -5, 11, 12, 9, -13, -10, 26, -29, -1, 18, 12, -5, -11, -24, -1, 34, 13, -13, 10, 0, 22, -14, -25, 12, 24, 0, -12, 6, 6, -16, 8, 25, -7, 14, -10, -15, 3, 14, 2, 7, -4, -3, -14, 12, 3, 6, 8, -11, -9, 0, -2, 11, 7, -19, 5, -13, 17, -3, -19, -7, 27, -1, 15, 20, -22, 5, -28, 33, -9, 1, 17, -12, 7, 23, -17, -2, 3, -13, -14, -8, 1, -20, 0, -6, 21, -16, -6, 21, -11, 5, -19, -2, 6, -7, 5, 9, -5, -1, 35, -17, -17, -22, -12, -8, -23, -8, 2, 21, -25, 12, 22, 14, 1, -5, 7, 18, -12, -15, 17, -12, 22, 11, -21, 15, -5, 1, -4, 8, 10, -2, -1, -8, -23, -14, 14, 15, 19, -10, 10, -3, 5, 0, -19, 16, -1, 0, 7, 5, 20, -9, -7, -5, -17, 19, 3, 5, -13, -24, 6, -2, -9, 14, 12, 4, 11, 9, -30, 13, -5, -4, -6, -28, 1, 25, -4, -19, -4, 10, -6, -34, -6, -19, 3, 0, -15, 9, 3, -22, -8, -1, -27, -28, 2, -22, 49, 14, -26, -2, 24, 50, -3, -28, 7, -30, 7, -2, -12, -10, -16, 33, 0, 9, -32, 1, 27, -15, 19, 0, 1, -14, 1, 8, 11, 11, 18, -21, -13, 4, -3, 25, -34, -41, -5, 30, 5, 23, 5, 0, -30, -15, 1, 13, 21, -22, -3, 9, 32, 6, 18, 11}
, {-61, -20, 9, -22, -20, 12, -8, -11, -25, 0, -2, -41, 14, 16, 22, -20, 3, -2, -7, -3, -13, 2, 3, 12, 16, 13, -27, -12, -14, -8, -25, 17, -15, 1, 6, -46, -2, -15, 18, -5, -25, -12, 14, 1, -24, 7, -9, -31, -17, 8, -9, 0, -8, -9, 11, -4, -29, 1, 14, -4, -4, -12, -4, 14, 2, 0, -8, -11, 6, 11, -6, -27, -18, -35, -15, 27, -53, 3, -11, -13, -40, 3, 0, 18, -36, -2, -53, -5, -44, -5, -1, -22, -4, -59, 5, -41, -18, 1, -5, 0, -1, 12, -10, -11, 6, 12, 10, 1, -44, -62, -4, -5, -22, 34, -1, 7, 31, -47, -18, -6, -2, -6, 18, -11, 4, -28, 62, -3, -21, 3, -12, -45, -26, 9, -44, -9, 4, -21, 6, -7, -13, 7, 1, -18, -28, 1, -2, 31, 3, -11, -17, -12, -57, -17, -4, -17, 15, -21, -3, -9, -29, -32, -18, -1, -20, 7, 11, -21, -6, 2, -36, -14, -37, -38, -1, 2, -20, -11, 17, -27, 6, -28, 0, 16, 16, 9, 2, 0, -6, -31, 4, 8, 20, 5, -37, 0, 11, 13, -5, 4, -12, 8, 10, -5, -5, 18, -20, 14, 2, -2, 19, -5, 8, 0, 5, 2, 8, -15, 5, -2, -9, -5, -19, 10, -18, -32, -28, 19, -11, 14, -3, -27, -1, 15, 3, -15, 22, -7, -14, -8, -6, -1, 38, -6, -4, -19, 3, -9, -21, 15, -14, 2, -40, 1, -8, 21}
, {-24, -4, -1, -35, 3, 16, -13, 6, 15, -9, -3, 22, -6, -5, -16, -27, -12, 9, -2, 13, 5, -5, -12, -7, -23, -3, -1, -13, -1, -16, -33, -38, 13, -15, -18, 50, -16, 34, -10, -4, 8, 18, -9, -2, -5, -4, 12, -5, -10, 16, 17, 8, 10, -20, -10, -1, -6, 21, 2, 17, 3, -3, 38, 1, 3, 6, -16, 33, -6, -7, 1, -11, -2, -25, 24, -3, -18, -4, -22, 40, 2, -2, -13, 9, 33, -3, -4, -27, -23, 24, -27, -4, 14, -20, 17, 34, 29, 5, -9, 12, 4, -21, 0, 0, 1, 8, -3, -27, -7, -5, 15, 13, 24, -14, -1, 2, -10, -6, 13, -11, -8, 2, -8, 7, 3, 3, -50, 5, -50, -8, 22, 5, -5, 2, 44, -3, -69, 12, 5, -20, -14, -3, 27, -7, -9, -39, -2, 3, -15, -40, 1, -15, 17, 5, 10, -5, 7, 9, -16, -8, 22, 19, 27, -20, 15, -13, 17, 18, 0, 14, 29, 12, 14, 16, -25, -30, 27, -22, -1, 55, -26, -2, -6, -21, -16, 9, -13, -12, 1, -25, 1, -5, 8, 1, 13, -11, 26, 9, -11, -3, 14, -13, -21, 1, 10, 18, -2, 9, 5, 16, -11, 7, -23, -18, -11, -3, -29, 15, -8, -23, 18, 4, 9, -13, 7, 34, -24, 23, 1, -8, 5, -12, 1, -24, -9, 21, -19, -5, 2, 7, -27, 22, 20, 12, -6, 6, 12, 9, 38, 3, 13, 12, 14, 6, -18, -18}
, {22, 22, -9, 14, -15, -18, 13, 23, -18, 0, -2, -41, -24, -15, -44, 3, -9, 2, 19, 1, 29, -24, 8, -16, 1, -36, -27, 16, 11, 0, 8, -12, -2, -20, 1, -23, 5, -7, -11, -6, -5, -9, -34, -33, 13, -5, -3, 28, 6, -30, -32, -17, -33, 6, -6, -29, -1, -22, -21, -30, -7, -13, -29, -11, -7, -33, 23, 9, -26, -34, 7, 3, 4, 6, -17, -32, 22, -20, -30, -31, 13, 11, 20, -10, 11, -13, 9, 4, -6, 6, -10, 19, -12, 7, -28, 13, -12, 0, -8, -14, -22, -28, -32, -6, -33, -32, -6, 10, 16, 22, -35, 10, 40, -43, -35, -24, -40, 10, 4, -46, -12, -12, -46, -38, -1, 15, -48, -19, -7, 14, 8, -3, 0, -15, -8, 6, 8, 3, -14, 6, -18, 7, -5, 29, 7, -1, -1, -3, 1, -3, 0, -6, 14, 12, -9, 12, 15, 6, -3, -30, 9, 20, 20, 16, -5, 18, -6, -4, 7, 27, -5, 2, -3, 16, -1, -23, 12, 6, 2, -27, 2, 2, 20, -2, -31, -9, -1, 12, 21, 1, 8, 0, 2, -15, 1, -8, 7, -8, -55, 1, 21, 14, 2, 12, 1, 6, 27, -1, -18, 6, 25, -34, 9, 6, -13, 8, -22, 4, 6, 8, -23, 6, -6, -10, 5, -2, -19, 5, 12, 14, 8, 31, 12, -22, -30, 11, 7, 34, 20, -25, -1, 12, -2, 4, 6, -19, 2, 17, -20, 21, -1, 4, 18, -4, 30, -56}
, {-39, -29, 6, -13, -27, 19, -12, 20, -2, -6, -28, -31, 8, 2, 12, 2, -22, -5, 11, -22, 6, 19, 3, 2, 8, 13, -45, -10, -26, 1, -37, 0, 5, 11, -2, -8, 0, -22, -3, 3, -9, -17, 17, 19, 3, 13, -4, -38, 1, -9, 6, 19, -5, -30, 5, 11, -32, 16, 5, -10, -6, -2, -30, -14, 9, 27, -37, -5, 13, -25, 14, -13, -15, -60, -22, 2, -38, 15, -13, 1, 1, 6, 14, -3, -33, -3, -37, -17, -46, -31, -17, -10, 15, -45, 11, -13, 16, -4, -16, -18, 7, -9, -10, -5, -2, 6, 8, -10, -56, -46, 10, -1, -27, -25, -13, -3, -23, -51, -12, 2, -15, -46, -5, 6, -17, -9, 2, 1, 8, 0, 42, 11, -19, -6, -29, -16, 7, -13, 2, -5, -6, -32, 12, 0, 12, 14, 4, -14, -4, -6, 9, -16, -6, -22, -7, 3, -8, 14, -3, 6, 22, -4, -10, 14, 13, -17, -21, -12, -6, -35, -13, -2, -7, -14, 15, 11, -9, -7, -4, -5, -18, 3, 13, 6, 5, -2, 3, 2, 7, 3, -7, -11, 15, 5, -15, 6, 18, 3, -8, 7, -6, 12, -20, -2, 6, -8, -6, -10, 13, -6, -17, -20, 4, 8, 2, -60, 22, 8, -5, -24, 18, 8, 9, -11, 7, -13, -14, -27, -10, 1, -10, -3, -8, -36, -32, 1, 5, 15, -21, -3, 13, -26, 4, -16, 14, -8, 12, -24, 11, 3, 2, 0, -29, -8, -8, -56}
, {41, 9, 1, -8, -8, -19, 21, -1, 9, -34, 8, -16, 5, 14, -6, -13, -24, 11, 15, -6, 11, -41, -25, 11, -9, 11, -14, 7, 21, -3, -10, 2, -11, -4, -5, -13, -19, -22, -12, 7, -28, -4, 22, 7, 4, -3, 10, 29, 2, -5, 0, 1, -47, -6, -23, -6, 5, 13, 10, -18, 6, 43, -12, 7, -13, 11, 20, 9, -16, -30, 14, -6, 5, -2, -1, 0, 15, -4, -1, -18, -1, -10, -32, -3, 20, -29, 26, -11, 13, 7, -3, 11, -4, 14, -16, 0, 8, 4, -8, 11, 6, -19, -34, 35, 2, -10, -22, 5, 41, 44, -15, -9, 19, -12, 0, 1, -32, -3, -2, -4, -17, 3, -14, 14, -2, 3, -44, 6, -28, 2, 10, 13, -7, -27, 2, 23, -21, -8, -26, 0, -1, -38, -7, 17, 8, 7, 6, -40, 7, 0, 8, -19, 5, -23, -17, -9, 12, 10, 9, 15, 28, 17, 13, 16, -8, -29, -30, 8, 2, -15, -30, 12, -4, -10, 11, -14, 14, -23, -23, 14, -5, 15, 20, -24, -27, -29, -34, 15, 2, -20, -46, -8, 9, 10, 22, -15, 13, 1, -18, -7, 17, -16, -32, 29, -7, -18, 7, 1, -3, 28, 6, -32, -5, -2, 7, -20, -19, 3, -7, -14, 15, 15, 0, 6, -17, 22, -21, -9, -21, 3, -16, -16, 12, -21, -17, -23, -17, 0, -17, 5, -29, 8, 20, 1, -10, -9, 18, 2, -10, -14, -29, 12, -7, -4, 3, -26}
, {44, -56, 11, 17, 19, 3, -12, 36, -10, -10, -7, -24, 21, 7, 12, -23, -4, -1, -12, -17, -19, 24, 14, -5, 9, 0, 9, -10, -42, 17, 3, 6, 19, 18, 9, -7, 10, -3, -5, 5, -17, -9, -12, 2, -5, 7, -21, -17, -8, 4, -17, -3, 20, 6, 17, -17, -6, 3, 11, -18, 13, -8, 7, -17, -12, 9, -21, 7, -4, -6, 14, -16, -20, 2, -35, -10, 10, -3, -29, 12, 14, -14, 1, 22, -5, 15, 10, -24, -3, -4, 14, -4, 1, 25, 14, -3, 8, 1, 20, 11, -12, 2, -19, -11, 0, -4, -4, -5, -3, 11, -10, -13, -10, -11, -13, -7, -29, 25, 2, -20, 11, 7, 18, 1, 10, 7, -12, -12, -32, 5, 16, 3, -28, -8, 20, -20, -8, 6, -21, 1, -2, 5, 3, 11, -9, 7, 16, 7, -15, -17, 21, 3, 13, -9, -14, -5, 1, -6, 14, 13, 13, 19, 14, -8, -3, -18, -21, 8, -18, 2, -1, 23, -24, -10, -8, -18, -7, -34, -10, 26, -13, 0, 15, 13, -25, -54, -14, 0, 25, 18, -32, 9, 0, 4, 7, 15, 15, -2, -7, -33, -4, -13, -23, 3, 18, 27, -18, 1, -5, 22, 22, 9, -25, -13, -22, -6, -41, 40, -14, -9, 16, -44, 10, -34, 3, 27, -9, -6, 42, -27, -8, 11, -2, -7, -10, 31, -25, -21, -8, 11, 29, -9, -7, 22, 4, -14, 10, -11, 35, -21, 18, -10, 12, 24, -31, -12}
, {-29, -9, -2, -12, 2, -15, -7, 16, -8, -8, 2, -9, -21, 3, 12, -15, -19, -12, -1, 15, 4, -17, 4, -2, -1, 4, -17, -2, -15, 13, 2, -4, -1, -2, 12, -12, -5, 4, 4, -8, 14, -6, 2, 9, 19, -24, -25, -21, -9, -23, -8, 1, -20, 3, -2, 8, -10, 19, -8, 0, 1, -20, 8, 4, -9, 21, -8, 1, 3, 0, 0, -36, 1, -42, -23, -10, -15, 18, -8, -5, -9, 8, 2, -6, -19, -7, -39, 9, -10, 0, -42, -6, -2, -48, 9, -3, 16, -2, -6, -5, 7, -7, 12, -3, 5, 8, 17, 20, -34, -45, 11, 9, -24, 4, 14, 4, -18, -41, -11, -5, -20, -17, 7, 2, -4, 1, 2, -7, 40, -9, -17, -48, -3, 4, -9, -43, 24, -22, 28, -8, -12, -2, -1, -54, -13, 9, -15, 22, 10, -31, -37, 24, -52, 3, 1, 9, 13, -55, 19, -8, -11, -6, -51, 0, -15, -17, 9, -44, -4, 21, -10, -15, -10, -35, 7, 39, -31, 16, 12, -26, -6, 8, -40, 18, 4, 9, 17, 5, -11, 7, 1, -28, 11, 7, -25, 22, 6, -29, -1, 9, -46, -9, 28, -31, -25, -20, -10, 15, 15, 0, 0, 18, 14, 9, -3, -32, 3, -14, 3, -4, 2, 2, 1, -4, 17, -8, 1, -13, -25, 12, -14, -11, -8, -6, -4, -16, 12, 14, -17, 4, 5, -18, -27, -14, -11, -23, 2, -8, -6, 13, 10, -17, -33, -3, 5, 20}
, {0, -5, -11, -7, -23, -31, 2, -11, -7, 18, 0, -27, -27, -21, 7, -11, 4, 3, 11, -10, -5, -3, 2, 9, 4, -6, -15, -8, -18, -23, 13, -22, 7, 13, 6, -15, -6, -4, -7, -27, -24, -5, 2, 10, 3, -3, -5, 7, -9, 15, -22, -10, -18, 17, 15, -11, -10, -17, -7, 5, 8, -9, -36, -3, -12, -3, 7, -9, -4, -39, 12, -30, 7, -48, 1, 2, 7, 10, -16, -2, 5, 11, 0, -8, 11, -12, -41, 7, -16, -13, -20, 22, 19, 6, -2, -6, 19, 21, -27, -9, -3, -5, 9, -2, 7, 0, 2, 5, -1, -10, -7, 1, 12, -13, 6, -7, -13, -62, -3, 0, -9, -14, -24, 0, 15, 7, -4, -10, 15, -29, 25, -14, 13, 11, -6, -22, 12, 4, -53, -39, 0, -5, -19, -22, 4, 3, -32, 11, 0, -38, 16, 10, 0, 19, 8, -8, 4, 10, -5, 19, -8, 2, -29, -22, 7, -37, -18, 9, -16, -5, -21, -2, -20, 8, 16, -11, 6, -17, -16, -43, -17, 23, 3, 2, 4, -3, 3, -27, 11, 8, -14, -33, -36, -29, 8, -10, -18, -19, -21, 8, -27, -13, 13, -41, -7, -41, 9, 10, -26, -2, -54, 11, 5, -14, 18, 5, 0, -41, 10, -32, -21, 11, -45, -22, -19, 2, -3, -9, -37, 35, -6, 11, -5, -16, 8, -6, -1, 23, -60, -48, -1, -27, 11, -17, -22, -4, -29, -19, -13, 15, 5, 0, -18, -8, -12, 0}
, {19, -21, 3, 14, 13, -17, 5, -6, 17, 17, -14, -6, -10, -21, -7, -8, 12, 11, -2, 13, -7, -4, -13, -13, -7, 0, 16, -6, 19, -8, 4, 10, 6, -21, 9, -18, 7, 10, 1, 0, -3, 1, -24, -6, -9, -17, -9, -7, 8, -11, 4, -6, -10, 22, 10, -17, 6, 31, -19, 4, -2, 1, -11, -13, -10, -10, 26, -9, 14, 11, -1, -21, 0, 2, -11, -3, 11, 0, 24, -23, -19, -8, -10, -20, -37, -1, -11, 22, 22, 17, 11, -5, -26, 4, -14, 8, 12, 16, 40, -19, 27, 19, 9, -5, -3, -12, 6, 28, 7, -15, -6, -27, 5, 16, -5, 17, 2, 22, 1, 9, 16, 19, 18, -15, 17, 16, 22, 13, 29, -6, -6, -23, 4, 3, 5, -32, -10, -6, 7, -41, 3, -11, -13, -40, 6, 6, 11, 10, -5, -30, -23, 12, -30, 0, -6, 0, -10, -13, 1, 6, 7, 18, -26, -26, 17, -24, -4, -7, -30, -11, 17, 2, 17, 14, 27, 22, -7, 2, -14, 33, -19, 25, -19, 5, 8, -26, 5, -22, -6, 12, -15, -7, 9, 22, -2, 23, -21, -44, 32, -1, -35, -21, -2, -6, -14, -4, -12, 22, 4, -25, -30, 23, 13, 6, -2, -20, 9, 5, -41, 7, 16, 5, 10, -10, 14, 4, 4, -3, -47, -23, -20, -28, 4, -13, 5, -13, 0, -10, -5, 7, -15, -15, 6, -26, -9, -14, -49, -16, -1, 12, 38, -24, 13, -27, -20, 14}
, {24, -4, 4, 15, 9, -2, -29, 8, 16, 4, -15, -17, 35, -4, 5, -1, 1, 7, 11, 2, 11, 20, -1, 0, 16, 10, 34, 17, 1, -5, 12, 33, -8, -6, 11, 0, 5, -21, -3, 18, -12, -26, 0, 2, 0, 4, -10, 14, 16, 0, -8, 4, -2, 14, 17, 0, -8, -18, -5, -2, 8, 52, -16, -15, -19, -28, 25, -6, -5, 11, -17, -11, -10, 2, 2, -8, 10, 3, 23, 5, -16, -11, 6, 4, -32, 8, 8, -14, 30, 17, 30, -24, -9, -8, -10, 4, -19, -4, 21, 11, -6, 17, 6, -14, 11, -6, 10, -1, 2, -22, 0, -17, -25, 30, 14, 0, 28, -2, -3, -6, 0, 2, 21, 3, 0, 5, 27, -2, -29, -7, -7, 33, -1, -5, 6, -20, -16, -15, -11, -3, 11, -7, 15, 3, 7, -7, 1, -2, -17, -18, 13, -1, 37, 0, -14, -4, 5, 14, 15, -2, 7, 19, -6, -20, 5, -22, -2, -5, -6, 3, 11, 11, -2, -10, 9, -16, 24, -37, -21, 6, -3, 3, 0, -6, -12, -26, -8, -4, 18, -17, -14, 16, 15, 1, -26, 7, -1, 10, 8, -32, -8, -3, -10, 15, -8, -18, -23, -13, -21, -7, -29, 2, -16, -30, 10, 4, -14, 5, -1, 11, 15, -12, 27, -31, -21, 18, -10, -24, 17, -53, -12, -31, -12, 4, 25, 13, -6, -47, -1, 25, -38, -3, 2, 33, 7, 17, -16, -19, 40, 24, 6, -31, 9, -11, -24, 3}
, {26, -1, 7, 4, 11, -39, 6, 6, 21, 26, 9, -30, -17, -28, -8, 2, 6, -21, -7, 9, -9, -5, 1, -12, 5, 20, -10, 6, -10, 4, 29, -14, 23, -5, 9, -31, 5, 1, -4, -7, -23, -2, -12, -10, -7, 2, 3, 10, 13, 4, -30, -15, -42, 12, 2, -32, -9, -27, -23, -20, 6, -7, -39, 2, -20, -2, -1, 2, -1, 0, -4, 8, 26, -10, -23, -2, 29, 11, -7, -1, 23, 11, 28, -5, 41, -12, 10, 12, -1, 6, 14, -7, 8, 25, -2, -5, -17, 9, -4, -8, -12, -26, -3, 18, 9, -6, 10, -10, -22, 3, -13, 18, 7, 7, 6, -16, 6, 6, 5, 1, -16, 23, 5, -15, 1, 2, -4, -22, 7, -52, 10, -2, 13, -5, 2, 27, 12, -2, -57, -29, -2, 4, 1, -30, 21, 14, 7, 15, -25, -16, -15, -13, -14, 21, 12, 3, -2, -17, 0, 0, 9, 4, -33, 8, 21, -23, -13, 11, -6, 1, -15, 5, -2, 4, 19, -11, 15, 10, -15, -39, -16, 20, 14, -10, 1, -13, -3, -22, 13, 9, -5, -16, -47, -21, 21, -19, -46, -32, -1, -13, 5, 12, 16, -9, 3, -28, 24, -13, -27, -28, -29, 4, 13, 0, 1, -4, -2, -28, -37, -13, -12, 20, -11, -15, -13, 6, 1, -26, -22, 1, 17, -1, 3, -7, 4, -11, 8, 10, -24, -32, 3, -41, -7, -30, -21, 9, -37, 22, 4, 6, 5, -8, 8, -7, -44, 2}
, {-7, 13, -11, 15, -31, 17, 21, -30, 36, -10, 5, -3, -1, -18, 7, -23, -17, -5, -14, -11, 23, 0, 8, 9, 7, -19, -14, 17, 25, 7, 9, -10, -1, 43, 21, -8, 12, 0, -15, 6, -27, 17, -7, -5, 17, 20, 4, -2, 10, 10, -17, 5, -6, 3, -17, 1, -4, -7, -6, -13, 8, -29, 11, 1, 7, -11, -20, -38, -20, -10, -24, -51, 0, -2, 24, 26, -28, -5, 18, -14, -38, -6, 16, -18, -9, -8, -12, 16, -32, 11, -35, -23, 23, -12, -13, -41, -40, -8, -21, 12, -14, 15, 3, 13, 16, -1, 3, -6, -7, -2, -1, 2, -21, 34, 15, -5, 8, -14, -14, 18, -9, -3, -4, 11, 8, -19, 3, 22, 21, -17, -2, 20, 1, 30, 2, 19, -9, 10, -3, -2, -2, -16, 10, 19, -20, -17, -26, -23, -7, 7, 21, -23, 4, 14, 7, -4, -14, -5, 13, 0, -3, -1, 9, -14, -10, -4, -13, -11, -10, -4, -6, -3, -1, 1, -19, -18, 0, 1, -11, 3, -2, 4, 20, -13, 15, -1, -9, -13, 32, -13, 12, -30, 2, -6, 8, 8, -30, -9, 5, 20, 6, 4, -1, -32, 11, -41, 13, -12, -12, -12, -2, -11, 25, 13, 15, -36, 4, -19, -10, 2, 2, 6, -12, -19, -8, -27, 22, -4, -22, 6, -21, -7, -6, 9, -33, -31, 18, 11, 1, -5, 9, -27, -17, -51, -26, 16, -32, -17, -8, 1, -24, -37, -8, -46, -4, 9}
, {-10, -4, -1, -19, 27, 0, -44, 7, 2, 6, -10, 9, 21, -4, -8, -20, 7, -13, -5, -3, -5, 3, -3, 2, 0, 20, 19, -17, -28, 12, -2, -1, -11, -20, 2, 5, 3, 9, -6, 0, 18, 6, -4, 1, -10, -3, 8, -29, 2, -4, 4, -3, 19, -13, 17, 4, 9, -19, 10, -5, 14, -14, 1, -5, 10, 7, -27, -3, 0, -7, 11, -9, -19, -4, 12, -8, 4, 7, -18, -2, -14, 13, 14, -8, 16, 16, 2, -18, 14, -3, -8, -19, 7, -16, 23, -7, -10, -20, 31, 19, 2, 7, -5, -56, 10, -12, 14, -1, -9, -25, -12, 8, -26, 15, 3, -29, -1, 21, 2, 1, 25, 16, 14, 1, -19, 12, 20, -3, -10, -6, 4, 44, 1, -20, -8, 3, -30, -12, -22, -28, 14, 9, 6, 1, 18, 21, 4, 15, -12, -19, -4, -14, 20, -31, -8, 15, 2, 7, -27, 3, 28, 7, 19, -34, 24, -37, 0, 10, -30, 30, 10, 4, -30, -9, -7, -17, 30, -25, -23, 20, -10, -13, 6, 6, -36, -27, -6, -33, 17, -5, -41, -1, -1, -2, 10, 0, -6, 14, -10, -38, -14, 22, -32, -4, -8, 6, -23, -22, -9, -11, -16, 13, -1, -32, 3, 8, 19, 12, 6, -15, 4, 11, 14, -24, 6, -9, -16, -10, -8, -44, 8, -23, 9, -1, 5, -1, 5, 1, -39, 10, -10, -17, -6, 15, 10, 15, 6, 14, 31, 8, 20, -19, -6, 1, -17, -5}
, {-4, 38, -37, 5, -2, 6, 16, 14, -10, -7, -6, 15, -31, -1, -28, 17, 5, -7, 0, 10, 6, -9, -21, -15, -24, -16, 8, -26, 6, -29, 14, -48, -14, -5, 3, -3, -13, 2, -4, -4, 35, -11, -5, -45, 7, -25, 5, -27, -2, 1, 4, -26, 17, -15, -17, 0, 1, 33, -8, 10, -52, -44, 4, -14, 8, 5, 16, 11, -7, 8, -9, -4, 7, -9, 17, -5, -3, 0, -7, 5, 8, 9, -12, 8, 6, -1, 9, -26, 4, 1, 21, -7, -12, -9, -5, 1, -2, -16, -9, 17, -30, -17, -17, -6, -2, -22, 21, 1, 22, 10, 4, 5, 19, 46, -20, -16, 19, 7, 5, 3, 3, 22, -3, -14, 2, 4, -5, 8, 1, -7, -4, -31, -23, 13, -1, -19, -7, -14, 29, 9, -36, -5, -6, -21, -38, -23, -39, 21, 19, -25, -5, 30, 2, 14, 8, 17, 22, -21, -5, -18, 9, -5, -7, 4, -43, -4, -2, -45, -22, 31, 1, -17, -6, 0, 6, 36, -26, -5, 21, -15, 3, -5, -29, 7, 14, 7, 1, -14, -30, -27, -9, 11, 7, 12, 10, -4, -1, -5, 20, 13, 0, 9, 10, -8, -6, 15, -21, -19, 2, -27, -14, 12, -3, -12, 4, -3, 27, -4, 0, 25, -1, 21, 22, 4, 0, -10, 11, 16, -10, 2, 24, -32, 1, 17, 15, -28, 6, 6, 16, 20, -28, 5, 6, 19, -14, -10, -12, -9, 17, 33, 1, -16, -6, 0, -8, 19}
, {-51, 13, 29, -35, 12, 11, -8, -5, -37, -27, -7, 1, 9, 10, 1, 10, 2, -11, 12, 12, -9, 1, -20, 0, -8, 12, 7, -15, -14, 16, -35, -15, 15, -4, 9, 19, 12, -3, 4, 2, 18, 15, -10, 0, -7, -30, -12, -16, -10, -11, -2, 5, -7, -54, -7, -16, -10, 2, -5, 16, -3, -34, 0, 6, 16, 30, -39, 20, -25, -19, 15, 10, -14, -16, 3, -31, -34, -4, -31, 15, 25, -20, -12, 20, 6, 26, 10, -14, 0, -16, -57, 9, 19, -7, 21, 9, 32, -13, -21, -15, -12, -3, -19, -26, -32, 23, -30, -23, 0, 14, -4, 26, -28, -44, -37, -16, -43, 19, 6, -10, -2, -14, -15, 5, -38, 23, -45, -3, -24, -4, 9, -13, -26, 9, -11, -10, -22, -31, 2, -3, 19, -15, 3, -1, 4, -7, 0, -2, 20, -11, -6, 3, 13, 27, -21, 1, -12, -8, -17, 6, 1, 22, -3, 12, 15, -2, -9, 0, 1, -9, 10, 28, 2, -26, -3, -4, 6, -1, 7, 0, -10, -33, -9, -9, 9, -7, -6, -11, 17, 1, 3, -18, 3, 2, -32, 6, 11, 3, 23, -20, -10, -6, 3, -2, -31, -1, -4, 12, -2, 39, -8, 18, 8, -8, 0, 1, 5, 8, 5, -20, 15, -15, 2, -25, 12, 5, 4, 16, 17, -26, 3, -33, 12, -17, 1, -40, -12, -14, -3, 18, -13, -33, -3, -19, -10, 8, -38, -2, 16, 8, 17, -25, -54, -1, -21, 10}
, {-33, -50, 20, -15, -29, -10, 2, 23, -21, -1, -19, -31, -8, -5, 11, 14, -14, -16, 20, -36, -8, -34, 7, 16, 2, -13, -11, -28, -15, 12, -11, 5, 4, 10, 7, -41, 6, -33, 7, 16, -39, -3, 6, 9, -13, 16, -12, -39, 7, -24, -12, 4, -29, -27, 8, 4, -35, 19, -13, -13, 7, -7, -33, -5, -12, 19, 37, -1, -9, 22, 2, -5, 12, -35, -20, 23, -9, -6, -22, -8, 14, 1, -7, 2, -31, -1, -28, -2, -5, 21, 9, -16, 1, -27, 20, -4, 4, -4, -1, -5, -5, -9, -12, 2, 3, 6, -5, 0, -13, -37, 3, 13, -6, -4, 11, 7, 26, -40, 5, -1, 15, -32, 14, 3, 19, -15, 27, 3, -9, 1, 17, -40, -52, -25, -15, -12, 29, -24, -27, 6, -30, -3, -7, 7, 5, 8, 31, 5, 26, -8, 29, 8, 18, -20, -42, -3, 14, 19, 0, -2, 2, -5, -3, 16, 1, -1, -28, -1, 11, -12, -43, 0, 5, 7, 11, -16, 5, -12, -11, -15, -10, -10, -48, 4, -22, -8, -23, -3, -12, 9, -6, -12, 15, -8, -25, -2, 1, -4, -14, 24, 4, -7, -29, -2, 9, 8, -35, -13, 21, -3, 34, -21, -19, 3, 13, -26, -35, -7, 12, 3, -3, -20, 5, -8, 4, 22, -21, -21, -2, 1, -35, 9, 1, -14, -6, -13, -42, -25, -24, -8, -8, -7, -4, -4, -3, -9, 9, -37, -4, -14, -55, 11, -38, 3, -11, -17}
, {55, 11, 1, 32, 25, -7, 14, 8, 31, -2, 12, -1, 0, -25, -7, 18, 4, -4, 6, -7, 8, -7, 2, -34, -4, -20, 32, -19, -3, 16, 27, 3, -12, -19, 16, -28, 0, -31, -26, -11, 5, -4, -30, -29, -10, 1, -10, -2, 13, -25, -30, -15, -13, -8, -4, -39, -10, -10, -30, -31, -14, -16, -22, 14, 8, -35, 37, -13, -8, 14, -6, -3, -7, 7, -18, -20, 40, -6, 24, 1, -2, -1, 16, -15, 19, 1, 9, 13, 28, 11, 12, -1, -36, 24, -9, 4, -32, -8, 5, 4, -2, -8, 0, 19, 16, -22, -23, -15, 8, 14, -15, 0, 2, 12, -6, 2, 20, 1, -11, -9, 6, 9, -3, -38, -14, -3, 14, 12, 17, -19, 1, -13, 7, 7, 8, -33, 1, -31, -14, -2, 11, 24, -4, -7, -5, 12, 6, 4, -9, -8, -24, 11, -19, 12, -16, -1, -17, -21, 3, -1, 0, 8, -30, 11, 8, 6, 10, 10, -33, 5, -15, 7, -14, -10, 11, 10, 8, 6, 16, -13, -26, -6, -11, 15, 12, -19, 20, 18, 17, 21, -12, -14, -15, -7, -17, 13, -14, -33, 24, -17, 8, -7, 5, 4, -6, 0, 11, -15, -20, -34, -18, 9, -14, 3, -10, 34, 18, -2, -30, -7, 19, 3, 12, -27, 5, 15, -4, 37, -12, 16, -2, -7, -12, 6, 7, 18, 5, -34, 1, 6, 3, -5, 26, 16, -8, -6, -11, 41, -12, -1, 12, 7, 1, -15, 9, 18}
, {-31, 20, 1, -18, -3, 23, -25, -18, -8, -12, -11, 28, 24, -4, -3, -15, -2, -2, 5, 3, -21, 30, -23, 0, -8, -4, 11, 6, 11, -11, -18, -8, 17, 6, -11, 3, 6, 9, -15, 7, 40, 30, 4, 1, 2, -9, 6, -18, 4, -10, -5, 12, 17, -31, -6, -2, 6, -9, -1, 1, -6, -17, 13, 13, 5, 5, -52, 7, 7, 6, 6, -38, -26, -17, 0, 0, -23, 0, 35, 18, -34, 1, 8, -23, -5, -13, -23, -7, -49, -3, -22, -10, 1, -51, 6, 9, -4, -4, -16, 17, 10, -1, 17, 7, 7, -6, -25, -14, -44, -18, 14, 10, -9, 4, -3, 10, 10, -15, 1, 10, 1, 6, 2, 11, -12, -2, 2, 18, -29, 15, -1, 6, 0, -47, 9, -12, -41, 26, -32, -13, -15, -18, 18, 24, -14, -18, 0, -18, -25, 8, 21, 6, 42, -22, 11, 4, -26, 21, -18, -23, 14, 5, 16, -18, 16, 1, 21, 4, 4, -9, 33, 20, -3, 12, -18, -35, -3, 3, 6, 41, -6, -16, 2, -18, -22, -28, -35, -16, 26, -50, -16, 7, 3, -20, 28, -11, -2, 18, -35, -7, 25, -13, -24, 32, 14, 7, 16, -25, -15, 2, -15, -21, -16, -6, -5, 30, -9, 2, 3, 1, -4, 0, -25, -3, -9, -8, -36, -9, 5, -10, 3, 6, -20, -11, -10, 5, -27, -22, 7, 12, -8, 28, 23, 0, 3, 3, 5, 23, 20, -8, -20, 3, 5, -1, 5, -19}
, {32, 11, -20, 33, 7, -20, 9, 23, -8, -7, -9, -26, 10, 0, -58, 15, 16, 22, 5, -7, 24, -20, 7, -16, 10, -16, -8, 2, -10, 11, 11, 47, -8, -20, 15, -39, 4, 6, -32, -1, -44, -27, -35, -13, 6, 12, 2, 4, -8, -14, -40, -14, -35, 20, -3, 2, -8, -2, -24, -25, -7, 29, -4, -8, 6, -18, 26, -14, 18, 25, -29, 0, 8, 14, -6, 13, 25, -8, 9, -29, -5, 4, -1, -5, 16, -1, 17, -11, 29, 17, 27, 5, -44, 7, -44, 17, -23, -14, -11, -26, -15, 14, -8, 9, 14, -28, -4, -2, 7, 17, -25, -6, 25, 21, 9, 0, 26, 10, 2, -17, -10, 16, 3, -13, 24, 1, 20, 16, -22, 33, -35, -7, 15, 9, -18, -2, 9, 11, 21, 4, -13, 12, -20, -3, 6, 3, -4, -8, 11, -1, -5, 5, -12, -6, -5, -11, 4, 10, -23, 6, -30, -16, -24, -6, -11, 4, -13, 1, 1, -20, -17, -15, 17, 15, -6, -22, -7, 3, 25, -7, 13, -1, -13, 21, 3, 5, 7, 17, -29, -12, 12, -2, 8, 3, -13, 0, -11, 4, -1, 6, 18, 16, 2, -1, -12, 6, -16, -15, 7, -6, 23, -6, 3, 6, -1, -1, -11, -35, -2, 20, 1, -9, 0, 26, -4, -1, -6, 1, 12, -3, -2, 0, 13, 12, 13, -1, -4, -2, 35, -5, 6, 4, -6, 18, -19, -4, 4, 14, -33, 22, -16, -3, 4, 5, 23, 14}
, {-28, 1, -6, -9, 19, 22, -5, -1, -21, 2, 11, -3, -6, 6, -20, 9, 10, 5, -8, 4, 3, -27, -14, 3, -12, 5, 32, -4, 22, 0, -24, -24, 14, -20, 13, 12, 8, 12, -16, 3, 6, 0, 0, -1, 27, -17, 17, -27, -5, -14, 11, 4, 32, 0, -9, -4, -7, 27, 6, 10, -6, -4, 6, -1, 6, 4, -7, 7, -6, 17, -1, -5, -41, -17, -2, -2, -9, 0, 7, 5, 8, -17, -11, 22, 5, 23, 9, -15, -7, 26, -5, -31, -5, -5, 2, 16, -3, 10, 9, -13, -12, 21, -9, -7, 2, 3, -13, -2, 4, -12, 1, -2, -29, 20, -4, 6, 16, 1, 5, 0, -12, 2, 20, 4, -15, -3, 25, -5, -31, -4, 16, 4, -22, -1, 18, -45, 0, -14, -7, -1, 10, 8, -2, -2, -4, -5, -2, 6, -3, -14, 2, 4, 4, 7, -18, 1, -10, 17, -28, 9, 7, 6, 36, -1, 12, -23, -4, 1, -2, -18, 29, 0, 20, -4, -1, -32, 14, -17, -3, 14, -9, -24, -2, 7, -1, -16, -14, -11, 13, -20, 1, -8, 4, 27, -44, 11, 11, -10, 53, -16, -11, -31, -10, 22, -6, -12, -29, -11, -21, 1, -5, 14, -36, -17, -2, 7, -8, 29, 6, -12, 48, -8, 48, -40, -29, 27, 7, 11, -4, -51, -24, -47, 10, 18, 29, -20, -15, -46, 27, 40, -59, -16, 11, 10, -4, -18, -7, 5, 29, -12, 29, -17, -23, -3, -52, 33}
, {-35, 16, -16, 28, 8, -19, 11, -15, -10, 18, -16, -1, -5, 0, 11, 1, -2, 3, -7, 7, -6, 5, -1, 9, 0, 8, 28, -4, -8, 5, 15, 1, 15, -7, 4, -7, -23, 24, 12, -5, 13, -6, 2, 12, 9, -10, 5, -5, -11, -5, 5, -5, -14, -22, -1, 9, -5, 30, 10, 9, 3, -14, 1, 5, -12, 2, 3, -4, -10, 14, -20, -29, -2, -14, 5, 2, -14, 11, 16, 10, -17, 29, 32, -1, 6, -39, -42, 3, 5, -3, -30, -9, 1, -35, 5, -20, -11, 8, 11, 7, -14, -2, 4, -18, 10, -3, 4, 9, -14, 4, 6, 2, -20, 14, 20, -3, 21, -1, -16, 2, 4, 16, 6, 6, -10, -12, 8, -1, 23, 5, 4, -34, -3, -6, -11, -13, -1, 0, 20, 6, -31, -8, 20, -11, -27, -18, -42, 8, -4, -23, -19, 26, -26, 22, 21, 6, -27, -24, -9, -22, -13, -10, -20, -4, -10, -1, 4, -9, 4, 26, 5, -18, 4, -6, -14, 34, -24, 16, 19, 8, 0, 13, -25, 21, 37, 17, 15, 1, -2, -5, 38, -13, 11, -7, -3, -19, 11, -5, 19, -5, -26, -7, 25, -19, -8, -33, -20, 3, -21, -1, -4, 9, 12, -28, -2, -19, -7, -13, 28, -28, 3, 6, -15, -5, 1, -20, 0, 9, -24, -12, 8, -20, 4, 5, 6, -22, -3, 43, 9, 0, -19, -6, 5, 13, 1, 13, -28, -8, 29, 30, 30, -33, -10, -9, -6, 18}
, {32, -22, -3, 12, 25, -41, -6, 16, 22, 1, -13, -10, 6, -26, -8, 15, 6, -27, -18, -21, -6, 31, 13, -47, -8, -11, 12, -25, -1, 16, 17, -11, -9, 2, 5, -16, -9, -6, -15, -3, -19, -13, -22, -20, -28, -1, -24, 5, -2, 20, -34, -19, 15, 16, 13, -18, -16, -27, -19, 0, 22, 13, -1, 5, 14, -26, 28, -10, 19, 25, -22, 7, 0, 14, 9, -20, 12, -9, 26, -5, 9, -31, -9, -33, -32, 4, 4, -19, 33, 1, 7, 4, -61, 7, 3, -15, -2, -22, 35, -3, 0, 10, -7, 10, -9, -7, -16, 10, 19, -20, -34, -2, -51, -6, -28, -11, 4, 24, 15, -42, 35, 11, -1, -20, -4, 0, 23, 6, 19, -41, 14, -9, 16, -12, -18, -11, 6, 4, -44, -31, 4, 1, -9, -37, 16, -13, -1, -15, 12, -1, 2, -7, 7, -24, 9, -3, 11, 12, -22, 30, 8, -2, -20, -21, -8, -4, -12, -5, -37, -10, -11, 0, -4, 25, -13, -8, 3, -49, -16, -9, -8, 13, 4, -10, -9, 1, -17, -36, -8, 10, -37, -7, -8, -5, 12, 0, -22, -45, -1, -9, 0, -9, 0, -1, 13, -7, -16, -14, 6, -19, -42, 16, -20, -7, 0, 0, -3, 0, -41, -5, 15, 2, 0, -17, 26, 7, 2, -17, -31, -14, -17, -17, -13, 15, -5, 17, 0, -24, 9, 4, -1, -19, 11, 18, -31, -12, -11, 7, 6, -15, 10, -10, 7, 13, -14, -2}
, {-29, 6, 15, 19, -28, -4, -11, 44, 5, 6, -19, 10, 2, -20, 0, 17, -16, -8, -23, -12, -36, 26, 4, 12, 4, 9, 14, -4, -21, 17, -5, -9, 15, 11, 4, 12, 11, 16, -11, 13, 27, 14, 4, 19, -20, 17, -11, -17, 16, 5, 1, 14, 27, -4, 3, -12, -28, -26, 5, 8, 10, -18, 10, -2, 17, -3, -2, 6, 16, 0, 0, 13, 1, -63, 9, 7, -24, -3, 4, 1, -15, 0, 7, -49, -27, 17, -35, -25, -23, -17, -3, -6, -2, -30, 25, 2, 4, 8, -8, 0, 15, 6, 2, 3, 22, 19, -3, -10, -24, -44, -3, -4, -11, 12, 20, -10, 12, -60, 8, -3, 2, -25, -17, 0, -3, -10, -21, -1, 7, -4, -7, 11, -4, -10, -16, 24, -8, 18, -23, 2, -14, -6, 16, 10, -20, -20, -10, -18, 1, 1, 15, 10, -3, 9, 17, 17, -22, 8, -11, -6, -6, -12, 6, -5, 14, 9, 7, 4, -12, -6, 15, 12, -40, 29, -25, -13, 17, -10, -5, 34, -6, 10, 23, 2, -19, 4, -20, -1, -1, -12, 2, 4, 13, -30, 6, 5, -14, 2, -12, 18, 4, 33, 0, -40, 24, 6, 32, -19, 9, -38, -9, 4, -1, 12, 10, -5, 8, -38, -6, -3, -52, 15, -35, 6, 13, -37, 7, -28, 0, 1, 6, 19, -32, -2, -20, -3, -10, 40, -25, -41, 8, -6, 13, -23, -13, 7, 0, -4, -29, 14, -38, -5, -15, 7, 23, -12}
, {-15, 4, -14, -15, -15, 17, -8, 21, -2, 3, -6, -12, 9, 2, 5, -14, -6, 11, -5, -15, -15, 12, 4, 5, 6, 7, 30, 6, 11, 5, -10, 2, 2, 12, 0, -5, -7, -1, -11, 0, -15, -9, 19, 10, -3, -5, 13, 15, 23, 3, -11, 9, 19, -3, 4, -20, 22, 21, 9, 8, -10, 23, -7, 20, -14, -10, -36, -14, -12, 17, -10, -30, -20, -14, 21, 7, -38, 9, 5, -2, -31, 32, 17, 11, -16, -16, -2, 23, -5, 13, -21, -29, -4, -32, -8, -6, -10, 2, 6, 18, -2, 1, 19, -1, -1, -6, 10, 8, -26, -20, 9, -8, -13, 24, 8, 10, -3, -5, 0, 18, 23, 23, 8, 2, -3, -13, 7, 19, -2, -6, 5, 65, 19, 10, -13, 12, -13, 7, -5, -10, 9, -8, 12, 14, 5, -20, 14, -4, -19, 25, 16, -24, 37, 12, 9, -8, -6, -1, 4, -3, 11, 13, 20, -18, 11, -10, -4, -1, -16, 16, 0, 8, -12, -1, -5, -10, 0, -27, -26, 29, -10, 17, 31, -16, -4, -21, -11, -23, 24, -18, -6, -12, 10, -11, 34, 8, 2, -16, -1, -8, 6, -6, 8, 6, 15, -7, 4, -11, -12, -9, -22, 20, 23, -21, 13, -8, 11, -1, 0, -8, 10, 33, 10, -28, -24, -1, 25, 2, -28, -17, -10, -10, -1, -17, 11, 3, 13, -1, -14, 28, -3, -6, 18, -33, 13, 7, -14, 10, 11, 20, -2, -34, 2, -1, -8, 17}
, {-3, 0, 4, -4, -1, -37, 3, 11, -17, -20, 0, -43, 6, -14, 12, -10, -4, -8, 21, -4, -19, 20, 10, -3, 4, 2, -52, 8, -19, 20, -15, 8, -6, 7, 2, -68, 26, -19, -3, -6, -26, 1, 7, 6, 6, -7, -18, 10, -5, 15, -9, 18, -23, 2, 20, 11, -3, -55, -4, -17, 17, 1, -42, 7, -5, 9, -16, -7, 14, -42, -6, -3, -25, -24, -4, -2, 6, 21, -30, -9, 3, 3, -2, 13, 27, -32, 1, -13, -22, -52, -8, 1, 14, 32, 8, -22, 3, 12, -36, 1, -20, -20, -28, -14, -1, -15, -3, -13, 21, 5, -4, 5, -24, -43, -15, -7, -37, -14, -30, -11, -12, -10, 0, -16, -24, -27, -8, -18, -15, 21, -5, -28, -2, -1, -18, -25, 24, -31, 11, 13, -4, -2, 1, 14, 4, 32, 13, 6, 14, 4, -3, -21, -8, -19, -13, -9, 19, 0, 14, 33, 1, -3, -8, -7, -18, -16, -7, 4, 5, 6, -63, 17, 6, -24, 33, 3, 2, 5, -5, -56, 1, -4, 8, 18, -17, -9, 5, 28, 2, 12, -15, -6, 3, 11, -45, 16, 10, 18, -19, -7, -7, 13, 4, -16, -12, 12, -9, -7, 0, 5, 23, -38, -15, 9, -9, -20, -15, 12, 13, -20, -8, -28, -29, -2, -21, -5, -21, -14, 14, 9, 0, 21, -3, -29, -22, -8, 7, 28, 0, -16, 15, -12, 1, -21, 12, -7, 21, -2, -22, 2, -12, 15, -23, 10, 3, -37}
, {61, 2, 4, 31, 19, -3, -7, 26, 0, 10, 25, -18, 6, 3, -20, 10, 10, 2, 2, -14, 7, 18, 21, -14, 4, -50, 32, -1, 8, -1, 7, -2, -14, -32, -8, -20, 2, 20, -2, 5, -21, -23, -11, -19, 4, -9, 7, 27, 5, -10, 3, -10, -14, 10, -25, -5, 5, 9, -4, -14, -22, -9, 25, -12, 9, -12, 8, 20, 0, 7, 0, 17, 31, 23, -15, 1, 28, -13, -6, 23, 17, 4, -9, -27, 22, 30, 17, -44, 21, 18, 36, 4, -1, 28, -4, 18, -2, -13, 7, -14, -15, 1, -21, 17, -5, -27, -39, -20, 37, 33, -21, -6, 27, -7, -36, -14, 0, 20, 17, -28, 0, 3, -28, -35, 6, 13, 9, -25, -13, 10, 20, -29, 19, -19, 25, -9, -2, 14, -8, 19, -3, 8, -5, 23, -7, -7, 12, 1, -10, -15, 14, 4, 17, 9, 16, -6, -4, 5, -1, -6, 11, 18, 21, 7, 0, 0, 0, 9, 18, 9, 12, -5, -16, 17, -9, -18, 0, -6, -5, 29, 5, -8, 3, 0, -19, -4, -1, -4, 6, 12, -3, 16, 0, -37, -25, -12, -2, 17, -26, -33, 24, 25, -20, 15, -15, 32, 6, -26, -10, -8, -1, 24, -46, -6, -7, 22, -32, 3, -12, 21, -18, -32, -4, 6, -10, -2, -13, 3, 15, 6, 22, -1, -13, -2, -5, 5, -27, -5, 30, 12, -4, 26, 3, 13, -6, -1, 8, 13, 3, 20, 6, 7, 14, 6, -8, -11}
, {40, -14, -2, 16, 11, -11, -7, 17, 5, 2, -18, -12, 14, -14, -8, 21, 10, 9, -12, -34, 2, 4, 9, -19, 17, -5, 2, 12, -2, -2, -6, 9, -12, -22, 8, -53, 24, -18, -12, 1, -10, -34, -24, -14, 20, 7, -27, 25, -12, -1, -35, -9, 18, 5, 36, -51, -15, -37, -11, -12, 3, 4, -32, -4, -19, -58, 39, -7, -14, 25, -11, 0, -11, 32, -9, -19, 22, -24, 20, -54, -18, -20, -36, 7, -15, -17, 33, 9, -1, 25, 20, -6, -48, 24, -52, -16, -14, -14, 41, -6, -27, 21, 0, 15, -3, -35, -14, 12, 16, -8, -49, -24, 15, 18, -11, -13, 19, 15, 10, -29, 2, -3, 10, -24, 12, -8, 17, 8, 17, 14, -6, -2, 12, -12, -13, -19, -5, 3, 8, -11, 8, 5, -20, 17, -17, 13, 13, -15, -2, 1, -32, 13, -15, -1, 1, -8, 5, 8, 0, 1, -18, -21, 0, 6, 5, -9, 9, -3, -21, -14, 5, 21, -4, -6, 13, 12, -3, 0, -9, 14, -3, -6, -17, 3, 3, -19, 15, 5, -3, 7, -4, -30, -1, 0, -1, 5, 4, -3, 22, 3, 5, -2, -29, 18, 3, -33, -3, -13, -7, -11, -15, 13, -41, 3, -1, 25, -2, 14, -20, 5, 0, -5, 27, -30, -12, 17, -3, -12, 18, -6, -2, -8, 4, 1, 12, 12, -7, -38, -4, 21, 6, -7, -5, 20, 24, -16, -6, 0, -9, -15, 12, 16, 59, -16, 7, -8}
, {5, 9, -12, -13, -2, 13, 1, 14, 15, -33, 1, 17, -32, -3, -14, 11, -27, 9, -5, -5, 24, -2, -11, -13, -23, 2, -14, -11, 2, -29, -4, -12, 4, -7, 13, 0, -5, 10, -20, -11, -16, -11, -1, -15, 0, -11, -4, -13, -3, -4, 7, 2, -2, -21, -16, 11, -6, 28, -5, 22, -13, -18, 16, -9, -15, -5, 22, -3, 8, -2, -14, 13, 11, 3, 0, -1, -1, 3, 20, -6, 22, -20, -29, -11, 3, -9, 34, -16, 25, 10, 24, -21, -17, 15, -25, 0, -14, -11, 2, -28, 2, 16, -15, 3, 2, 8, -4, 9, 11, 45, -11, -19, 27, 33, 3, -10, 7, -2, 2, 0, -9, 5, 1, 9, -7, 7, 13, 5, 26, 10, -11, -25, 19, 13, 0, 11, -9, -2, 24, 14, -37, -14, 3, -7, -51, -23, -41, -16, 7, -4, -36, -7, -58, 19, 18, 19, -19, -37, -4, -47, -15, -38, -8, -12, -51, 5, -2, -31, 6, -19, 0, -19, 15, -15, -1, 19, -39, 7, 9, 6, 16, 13, -35, 0, 17, 23, 2, 10, -31, -12, 15, -6, -1, 13, 13, -23, 11, -8, 19, 17, -30, 1, 33, 5, -7, -11, -23, 9, -14, -5, 8, 0, 19, 3, -21, -21, -5, -13, 9, 14, 3, -10, 18, 9, -28, -29, 7, 27, -16, -11, -6, -30, 0, 5, 11, -42, -6, -11, 6, 27, -35, -35, -8, 8, 8, 0, -13, -6, 20, 4, 4, -17, 7, -7, -3, 30}
, {32, -7, -4, 26, -19, -13, -11, 6, 14, -18, -9, -21, 6, 2, -8, 9, -19, -3, 2, -11, 24, 2, 18, 12, 8, 8, 2, 3, -4, 1, -11, 13, -21, -9, 26, -16, 19, -37, -6, 5, -24, -23, -1, 10, 3, 23, -14, 34, 2, 5, -9, -3, -16, 6, 15, 9, -2, -17, 1, 3, 18, 43, -40, -5, -21, -32, 16, -4, -10, 17, -13, -4, 6, 44, -8, 11, 25, -19, 24, -28, -23, -27, -30, 8, 7, -28, 47, 2, -1, 32, 38, -29, -27, 25, -46, 4, -9, -9, 13, -6, -17, 9, -11, 22, 5, -20, -17, 3, 41, 34, -32, -40, 9, 22, -10, -4, 2, 7, 6, -13, -11, 5, 11, 26, 12, -5, 10, -1, 6, 10, -9, -8, 8, -6, 4, -15, -1, 20, 16, 6, 3, 4, -2, 24, 12, 11, 20, -13, 5, -6, -10, -10, 1, -3, -5, -22, 16, -9, 8, 5, 20, -5, 17, 4, -12, -22, -2, -17, 1, -5, -12, 7, 19, -10, 20, 4, 2, -12, -13, 7, -9, -8, 7, 10, 8, -3, -5, 15, -19, 0, -19, -26, 10, 9, -26, -9, 2, -7, 4, -7, -7, -7, -10, 24, 10, -4, -8, -12, -5, -4, 5, 5, -27, -2, -3, -7, -8, -3, -18, 9, 16, -23, 25, -22, -5, 40, -6, 1, 4, -18, -14, -9, 22, 11, 4, -5, -20, -50, -3, 21, -16, 19, -1, -1, 8, -11, -18, 7, -6, -11, 2, 3, 36, -10, 0, -15}
, {38, 8, -33, 10, 8, -13, 13, 25, -8, -37, 1, 4, -34, -5, 16, 5, -2, -8, -6, 1, -14, 12, -40, -13, -37, -8, -22, -35, 9, -42, -10, -22, -4, -11, -33, 22, -23, 27, 0, -13, -3, 9, -4, -4, -14, -29, 2, 10, -5, -5, 20, -7, 2, -23, 9, 8, 17, 3, 2, 2, -21, -33, 10, -7, 21, 11, 14, 11, -5, -3, 0, 52, 24, 7, -3, -14, 22, 5, -13, -2, 14, -23, -29, -6, 14, 3, 9, -22, 8, -14, 27, 13, -4, 6, 10, 16, 4, -13, 7, -3, -11, 0, 0, 17, -34, 13, 6, -4, 9, 52, 3, 3, -10, -34, -9, -4, -17, 33, -7, -7, -12, 10, -8, 5, -13, 12, -17, -2, 22, -14, 6, -30, -14, -9, 20, 12, -12, -9, 9, 9, -22, -16, -11, -42, -56, -20, -19, -13, 34, -24, -48, -9, -42, 6, 11, -14, -5, -33, -16, -44, -29, -26, -21, -8, -33, 2, 4, -37, -7, -34, 19, -10, 22, -7, -41, 3, -32, 12, 3, -3, 4, 3, -14, -37, 1, 12, 2, -2, -22, -6, 0, -8, -1, 0, -15, -7, 12, 6, 12, 12, -11, -22, 9, 16, 4, -30, -42, 9, -13, 9, 2, 0, -18, 11, -13, -38, 4, 31, 19, -11, 7, -12, 1, 14, 4, 0, 7, 10, 22, 0, -22, -30, 7, 11, -13, -34, -18, -19, 17, 17, -3, -21, -16, -18, -7, -11, -12, -15, 17, -20, 29, -22, -14, 4, -18, 11}
, {30, 16, -12, 2, -7, 5, -17, 1, 20, 26, -1, -3, -8, 8, -5, -13, -7, 12, -8, -2, 3, 3, 3, -8, -3, -19, -7, 18, 15, -19, 9, 0, -16, -9, -10, 4, -26, 27, -7, -6, -14, 6, -1, 1, -3, -27, 11, 18, -3, 18, 6, 1, 14, 11, -8, 11, 33, 13, -8, 15, -8, -8, 7, 3, -8, 11, 13, 15, -5, -11, -20, 6, -3, 19, 25, 12, -4, -15, 4, 5, -19, -14, -7, 3, 4, 14, 5, 1, -22, -6, 29, -16, -13, -4, -49, 5, -5, -14, -4, 31, -2, 12, 0, 12, 15, 0, -10, -3, 4, 5, -3, -14, 28, 12, -1, 8, 24, 36, 26, -8, 20, 24, -5, 20, 6, -15, 1, 10, -10, 0, -9, 10, -2, 18, -4, 24, -22, 51, 23, -3, -24, -3, 18, 12, -55, -36, -15, -3, -15, 8, -10, -7, -19, 26, 38, 9, -20, -11, -24, -51, -14, -13, 9, -14, -22, 5, 41, -15, 1, 35, 17, -31, -2, 8, -49, -9, -9, -15, 2, 41, 1, 16, -8, 2, 18, 16, -6, -2, -11, -47, 26, 19, 13, -5, 31, -6, -11, -5, -9, 2, 19, 20, 14, 4, 27, -9, -12, -28, -13, -22, -8, -7, 21, -9, 1, 18, 14, -25, 3, 2, 8, 14, 4, 9, 6, -16, 8, 27, -18, 5, 22, 13, 5, 11, -13, -28, -8, 22, 10, -8, 0, 8, 7, -4, -37, 38, -6, 17, -10, 22, -19, -6, 21, -11, 21, 23}
, {17, -15, 2, 1, 4, -15, 5, 8, -37, -12, 3, -48, 10, 7, -4, 0, -19, 13, 31, -26, 14, -30, -6, 19, -2, -13, -8, 18, 9, -4, -13, 26, -15, -23, 5, -41, 22, -31, -4, -15, -67, -18, -6, 2, 2, 20, 0, 28, -9, -31, -17, 8, -45, -6, 23, -10, -3, -9, -10, -29, 23, 28, -18, -15, -7, -5, 6, 2, -7, -8, -7, 10, 11, 3, -9, 8, 7, -15, -9, -21, -3, 18, -17, 15, 9, -19, 17, 4, -18, 5, -5, -15, -6, 4, -14, 1, -14, 1, 6, -22, -17, -23, 11, -6, -6, -21, 1, -15, 20, 4, -25, 3, 19, 2, 26, 4, -2, -8, -5, -10, 3, -9, -5, -10, 15, -18, -2, 10, -16, 30, 13, -11, 5, -10, -21, 32, -5, -62, 10, 23, -12, -25, 0, 45, 3, -4, -10, -15, 6, 7, 14, -13, -1, -1, -13, -26, 15, 26, 10, -10, 22, 16, 20, 12, -6, -18, -3, -11, 9, -7, -16, 10, -12, 1, 9, 10, -8, 1, -7, 20, 3, -29, 13, 9, -7, 5, 11, 7, 19, -39, 0, -18, 27, -3, -31, 6, 6, 26, -24, 3, 4, -6, -9, 8, -11, 0, -12, -1, -6, 17, 22, -8, 6, 2, -12, -1, -16, 2, 23, 9, -9, -14, 1, 21, -21, 2, -18, -6, -11, 0, -25, 5, 7, -27, 4, 1, -46, -25, -13, -12, 6, 31, -17, -2, 30, -4, 28, -34, -9, 11, -21, -16, -19, 2, 16, -22}
, {15, -3, 1, 37, 15, 21, -8, 20, -15, 5, 15, -3, 14, -23, -9, 13, 3, -17, -14, -15, 2, 26, 19, -19, -5, -13, 18, -2, -16, 5, 20, -4, -9, -23, 13, 9, 18, 1, -18, -2, -14, -14, -33, -27, 33, 4, 2, 28, -13, -18, -8, -18, 15, 10, -10, -15, -10, 6, -2, -36, -28, -16, 11, 5, 14, -3, 9, 5, 5, -4, -12, 5, -5, 9, -24, -7, 44, -23, -26, 24, -4, -8, 20, -15, 26, 28, 12, -26, 23, -12, -1, 14, 1, 21, -9, -6, -42, -6, 17, 6, 11, 14, -26, 3, 1, -23, -35, -5, 2, 12, -15, 5, -6, -7, -38, -1, 3, 8, -17, -25, 5, 14, -27, -23, -21, -5, 16, -7, -12, -16, 2, 9, 28, -18, 5, -25, 8, -40, -47, -6, 20, 24, 9, -9, -7, 20, 0, 28, -20, -10, -6, 17, 23, 13, 13, 21, 5, -4, -21, 9, 9, 1, 5, -5, 25, 30, 7, 20, 3, 0, 10, 20, -29, 3, 1, 7, -10, 8, 18, -12, 1, -23, 27, -5, 3, -43, 1, -1, 18, 5, 8, 8, -25, -18, -6, -2, -27, -26, 0, -22, 5, -26, -21, -8, -8, 32, 10, -33, -20, -8, 18, 6, -22, 4, -16, 17, 9, -5, -29, -3, 13, -31, -15, -13, -1, -13, -9, 26, -8, 21, 7, -10, -17, 6, -13, 13, -19, -23, -2, -4, 12, 1, 15, -9, -9, 6, 16, 33, 19, -25, -18, 8, -8, 2, -6, -19}
, {2, -20, 0, 9, 18, -6, -9, -26, -21, 0, -5, 11, -14, -5, 7, 10, -10, -24, 0, -1, -9, -3, -3, -5, -12, 27, -15, -7, -12, 5, -15, -26, 13, 2, -9, 23, -14, 12, -13, -4, 14, -5, -13, -4, -1, -9, -16, -4, -14, -26, 11, 17, -5, -23, -2, 2, 8, 4, 10, 19, 0, -30, 9, 8, 16, 1, -10, 29, 10, -34, 4, 18, -5, -17, -8, -25, -4, 3, -41, -10, 34, 1, 12, 7, 18, 3, 4, -27, 0, -46, -22, 18, 11, 11, 12, 8, 9, -8, 3, -10, -11, -19, -22, 28, -29, 1, -25, -5, 1, 35, 16, 18, 23, -51, -29, 7, -25, -4, 3, -16, -8, -12, -24, -18, -12, 18, -11, -9, -2, -13, 0, -41, -22, 3, -23, -15, 15, -7, 12, -5, -5, 1, 19, -41, -7, -5, -37, 16, -5, 10, -22, -6, -26, 4, 5, -39, 6, -29, -5, -6, -10, -26, -25, -14, -30, 30, 17, -13, -5, 0, -2, -18, 0, 11, -7, -5, -26, 2, 16, -30, 3, -29, -27, -2, 3, 15, 1, 6, -1, 34, 20, -1, -5, 12, -34, 8, 25, -1, 6, 12, 14, -15, 4, 0, 6, -15, 9, 6, -5, 13, -4, -33, 4, 15, -5, 2, 0, -10, 8, -3, 7, -25, -11, 9, 6, -22, 4, -7, 26, 7, 4, 32, -3, -21, -11, 0, -6, 8, 14, -13, 34, -3, -41, 1, -7, -34, 17, -4, -14, -16, 25, -2, 16, 5, -24, -8}
, {-15, -35, -4, -27, -11, 10, -19, -12, -6, -2, -10, -40, 1, -30, 8, 8, -9, -7, 0, -15, -12, 0, 4, 21, -5, -13, 0, -4, -16, 15, 13, 16, -7, 5, 16, -45, 13, 0, 19, 23, -27, -16, -3, 10, 14, 7, -12, -44, 12, -12, -13, 4, -12, 7, 7, -14, -2, 0, -29, -2, 4, -3, -15, 11, -14, -19, 19, -15, 45, 19, -5, -5, 4, -7, 5, 32, 1, 2, 4, -23, -16, -17, -9, 2, -39, 13, -23, -1, -2, 15, 40, -21, -10, -27, 5, 2, -27, 3, 11, -20, 7, 15, 10, -8, 28, 4, 22, 2, -10, -36, -7, -14, -19, 24, 18, 20, 29, -37, 4, 7, 10, -11, 29, -7, 24, -9, 38, -5, -37, 11, 4, -16, -28, 0, -20, -15, 4, -45, -13, -3, 16, -6, -4, -9, -9, 8, -8, -13, -9, -19, -10, 9, -2, -7, -46, -25, 12, 1, 10, 5, 7, -28, 9, 4, 9, -6, -37, 12, -9, -7, -33, 17, 9, -12, 6, 10, 2, 4, 5, -6, -23, -27, -24, 2, -4, -4, -11, 10, -14, -11, -10, 1, 26, 12, -36, 20, -2, 7, 20, 23, -6, 2, -14, 9, -6, -8, -47, 8, 5, 2, 6, 0, -32, -2, -1, -4, -20, 4, -3, 38, 13, -8, 4, -3, -33, 11, -13, -17, 13, -19, -10, -17, 16, 13, -1, -6, -10, -34, -21, 22, 2, -12, -18, 21, 13, 6, -6, -9, 13, -6, -15, 6, 2, 2, -19, 3}
, {-7, 15, -17, 11, 7, 2, 4, 22, -15, 9, 3, -14, 6, -8, 19, 29, -5, -18, 2, 9, -23, 16, 16, -2, 15, -6, 20, -4, -10, -1, 1, -3, -8, 12, 10, 3, -15, 18, 26, -3, 3, 2, 9, 3, 11, 6, -6, 15, -2, 3, -9, 9, 16, -9, 5, 0, 15, -12, -5, 6, 2, -27, 12, 6, 37, 5, -11, -6, -14, -46, -3, -18, -9, -56, 8, 10, -26, 4, -28, 24, -4, 4, 26, 0, 29, 16, -28, -3, -28, -7, -57, -2, 11, -15, 7, -19, -3, -11, -6, 15, -8, -15, -3, -4, 19, -8, -14, -7, -29, -5, -6, 21, -29, -5, -1, -17, -8, -44, -6, -12, 2, -35, -29, -8, -2, -15, -13, -10, 1, -25, -2, 0, 6, 2, 6, -13, 8, 8, -43, -14, 3, 6, 1, -14, 3, -5, -2, 17, -39, -42, 21, 29, 10, -2, 20, 10, -17, 10, -4, 10, 11, 39, -1, -39, 15, -41, -12, 18, -1, 8, 3, -11, -12, 8, -7, 1, 3, -25, -37, 22, -5, 8, 7, 6, -5, -26, -5, -21, 22, 7, -41, -3, -14, -64, 11, -11, -12, -3, -14, -40, 7, 24, 2, -51, 20, 13, 6, -43, -10, -27, -11, 0, 5, -46, 19, 26, 18, -35, 0, -16, -10, 19, -10, -22, -3, -9, -12, -17, 21, -29, 16, 10, -32, 7, 7, 15, 23, 16, -20, -39, 10, -33, 3, -40, -19, 17, -24, 1, -2, 25, 1, -33, -2, 2, -4, 7}
, {-29, -4, -22, -13, -18, 3, -13, -27, 19, -29, 23, 15, -23, 5, 4, -6, -17, -10, -5, -8, -3, -22, -12, -9, -42, -15, -8, -22, 4, -37, 5, -32, -4, 2, -35, 1, -34, 5, -2, -15, 6, 21, 8, -9, -15, -38, 1, -29, 4, -11, -2, 3, -7, -30, -25, -2, 3, 2, -4, 24, -9, -16, 5, 13, -3, 13, -43, 17, -1, 5, 10, 2, -17, -2, 8, -38, -14, -15, 28, -11, 4, -25, -22, -5, -31, 24, 7, 0, -6, -6, -27, 11, 5, -14, -5, -6, 22, 6, -8, 7, 23, 2, 5, 14, -28, 11, 1, -4, 0, -27, -4, 7, 14, -41, 0, 6, -11, 8, 16, -3, -10, -2, -10, 15, -7, 18, -20, 10, 39, 6, -3, 18, -6, 16, 24, 0, -18, -11, 4, -4, -13, -26, 6, -18, -42, -25, 5, -39, 18, -2, -8, -5, -27, 16, -1, -32, 4, -17, -7, -29, -57, -34, -10, -16, -34, 12, 3, -43, -1, -46, 6, -22, 13, -9, -23, -5, -11, 2, 5, 13, 0, -12, -15, -34, -6, 10, 2, 2, -31, 5, 5, -1, -21, 5, 4, -1, 18, -28, 24, 20, -3, -34, -15, 22, 3, -58, 7, -8, 6, -1, -25, -6, -27, 10, -1, 8, -9, 15, -16, -33, 0, -1, 9, -18, 22, 18, 24, -9, -12, 20, -38, -14, 9, 5, -21, 6, -7, 13, 15, 5, 8, 14, -24, -8, -6, -36, 9, 0, -2, -37, 8, 13, 18, -27, 1, -4}
, {-39, 16, -5, -25, -27, 2, -13, -1, -23, 7, -4, 10, 1, 4, -10, 12, 3, 10, 6, -10, 0, -9, -7, -17, 2, 13, 14, -7, 0, 10, -2, -9, -3, -19, -2, 1, -6, 8, 6, 4, 4, 3, 8, -6, 7, -18, -6, -6, 20, 0, -2, -7, 6, 5, -8, 13, -11, 0, 5, 1, 3, 7, 19, 6, -13, -2, -3, -8, -5, 11, 5, -23, -22, 0, -12, 1, -44, 6, 12, 15, -6, -2, -13, 2, -32, 30, -12, 5, 5, 6, -35, -23, 3, -9, 14, -23, 11, 1, 5, -1, 15, 25, 15, -17, -2, 2, 13, 15, 3, -19, 6, -12, -48, 20, 2, 16, -2, -2, 3, 1, -5, -2, 6, 8, -3, 9, 8, 10, -11, 15, -2, -20, -23, 12, 10, -38, 12, -18, 43, 15, -37, -2, -15, -5, -27, -18, -31, -19, 29, 9, -15, 15, -38, -5, -21, -38, 8, -6, -4, -42, -59, -42, -18, -6, -43, 10, -17, -43, -2, -26, -24, -16, 29, 33, 4, 18, -25, 25, 4, -10, 11, -3, -39, 2, 9, 15, -2, 0, -37, -32, 26, -12, 2, 26, -46, 16, 11, -11, 6, 16, -17, -1, -10, -8, -9, -8, -4, 15, 0, -6, 25, 0, 13, 30, 4, -36, -11, -8, 9, -17, 1, -4, -7, -11, 4, -45, 25, -17, 26, 15, -30, 2, 5, -9, -9, -41, -25, 18, -14, 10, 21, -3, -18, 5, 11, -22, -24, -29, -24, -3, -13, -8, -3, -15, 2, 12}
, {16, -29, 8, 5, -29, -55, 4, 9, -6, 3, -21, -34, 5, -19, -1, 9, -11, 5, 5, -29, -9, -1, -1, 6, 9, -31, -25, 2, 2, 11, -14, -6, -1, -3, 5, -32, 5, -38, 11, -20, -52, 1, -2, 3, 7, 10, -2, 20, -15, -12, -24, -7, -57, 4, 7, -19, 1, -19, -44, -18, 7, -30, -32, -8, -4, -9, 23, -7, -10, -34, -12, -19, 10, -24, -28, -7, 15, -2, -48, -17, 10, 17, 30, -21, 29, -19, -25, 15, 4, -7, -22, 21, 12, 11, -3, -13, 0, 6, -46, -1, 2, -14, -22, -12, -9, -12, -11, 5, 5, 28, -7, 17, 6, -10, -13, -29, -35, -28, -20, -12, -18, -2, -33, -2, 1, 1, -9, -4, 10, -19, 5, -21, 3, 7, -36, -1, 30, -7, -10, 7, -1, -7, -21, -4, 15, 3, 9, 6, 21, -11, -6, -12, 6, -3, -1, 7, 20, -7, 5, 19, -5, 0, -20, 10, 5, -16, -6, 4, 4, 8, -31, -4, -11, 11, 19, 16, 4, 26, 13, -43, 14, 2, 5, 3, -9, 11, 3, 16, 7, 32, 15, -24, 10, -8, -14, -1, 13, 3, -36, 9, 4, 1, 21, 12, -16, 0, 14, 0, 0, 17, 16, -25, 27, 4, 1, -3, -15, -20, 13, -12, -27, -2, -37, 31, -15, -9, -23, -16, 10, 19, -10, 22, -20, -36, -7, -8, 0, 38, -3, -47, 31, -8, -21, -31, -1, -11, 30, -10, -32, 16, -30, -2, -17, -2, 15, -6}
, {16, 24, -15, 38, 6, 18, 4, -23, 24, 8, 4, 20, -6, -1, -4, -7, -7, -13, -9, 7, 6, -20, -22, -6, -11, 9, 10, 2, -2, -9, -4, -6, -5, -3, -6, 7, 8, -8, -6, -2, 7, -5, -7, -1, -13, -2, 13, 45, 21, -22, 1, 10, 1, 12, -30, 11, 6, -4, -2, 8, -11, 7, 9, 20, -2, -15, 7, 6, 10, 18, -15, 15, -17, 47, 4, -22, 23, -24, 13, -12, 22, -29, -18, 9, -27, 14, 35, -11, 7, 29, -5, 7, -30, 2, -10, 12, -6, -6, 0, -24, 12, 21, -7, 9, -9, -15, -6, -21, 21, 32, -20, -18, 16, -14, -4, 8, 23, 13, 11, -18, -9, 6, 15, 12, 1, 20, 7, -6, 13, 2, 5, 21, 13, 0, -15, 16, -11, 11, 2, -5, -9, -1, 7, 4, 0, -3, 4, -32, -11, 16, -9, 1, -22, 11, 19, 4, -25, -15, -4, -14, -32, -16, 5, 13, 2, 9, 8, -12, 16, -6, 0, 9, -12, -2, -1, -2, -12, -5, -3, 16, 4, -24, 9, -11, -14, 2, -2, -3, -12, 1, -8, -16, -5, 14, 23, 10, -9, -29, 23, 1, 15, -30, -7, 28, 8, -52, 10, 7, 3, -1, -20, 2, 10, 15, -3, 3, -4, 4, -36, -16, 29, 5, 19, -59, 5, 44, 22, -14, -6, -10, -20, -9, 21, 10, -4, 10, 15, -31, 4, 24, -2, -9, -44, -10, 11, -23, -14, -19, -5, -37, 26, 8, 3, -42, 5, 13}
, {-41, -13, 6, -9, -1, 15, 4, -9, 10, -3, 14, 6, -5, -27, 2, -3, -18, 9, 20, -16, -11, 10, 7, 1, -3, 1, 7, -2, -18, 1, -11, -16, 0, 17, -16, -9, 10, -4, -6, 13, -4, 32, 5, 5, -2, 5, 0, -26, 10, 27, 0, 12, 9, 13, 16, 5, -6, -9, 2, 1, 5, -12, -1, 2, 6, 20, -50, -14, 9, 10, 29, -55, -26, -25, 24, 2, -61, -6, 14, 0, -48, -10, 8, -5, -25, -36, -57, -10, -31, -21, -6, -38, 3, -63, 19, -21, 3, 11, 1, 20, 17, -8, 19, 5, 31, 26, 19, -30, -43, -53, 15, 6, -32, -15, 0, 13, -22, -16, -19, -6, 18, -17, 12, -15, -8, -36, 6, 18, 24, -14, 20, 1, -2, -9, 7, -14, 5, 9, 8, -1, -33, -28, 9, -8, -14, 5, -2, 12, -4, 6, 18, -2, 7, -9, -2, -11, -16, 8, 0, 12, 9, 3, 13, -14, -18, -12, -10, -9, 2, -13, -15, -13, 9, -1, -9, -3, -18, -8, -7, -2, -8, 4, -9, 5, 10, 6, 8, -14, -8, 22, -11, 14, -22, -5, 13, -2, 7, 5, -15, -3, 6, 3, -30, -20, 19, 9, 3, 2, 17, -17, -12, 0, -42, 0, 5, -25, -7, 22, 3, -7, -26, -15, -13, -23, 30, 13, -7, -36, 8, 2, 4, 4, -11, -23, 12, 35, 1, 25, -27, 2, 12, 4, 23, -10, 10, -20, 14, 1, -1, -4, -17, 14, -15, 3, 7, -41}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_5_H_
#define _DENSE_5_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 10

typedef int8_t dense_5_output_type[FC_UNITS];

#if 0
void dense_5(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_5_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_5.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 10
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int8_t
#define LONG_NUMBER_T int32_t


static inline void dense_5(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q7(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q7(
#endif
                             (q7_t*)input,
                             (q7_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q7_t*)bias,
                             (q7_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 128
#define FC_UNITS 10


const int8_t dense_5_bias[FC_UNITS] = {7, 10, -26, 6, -10, -12, -20, -5, 36, 3}
;

const int8_t dense_5_kernel[FC_UNITS][INPUT_SAMPLES] = {{17, -20, -11, -40, 31, -31, -19, -49, 17, 22, 8, -42, -19, 24, 19, -19, 20, -11, -28, -21, -54, 15, -30, -17, -47, 10, 22, 15, -19, -4, -16, -47, 19, 0, -20, 6, -38, 9, -25, -56, -32, -57, 31, 13, -36, 26, 33, 23, -43, -61, 13, -26, 17, -30, 21, -21, 17, 23, -16, -39, 12, -15, -68, 19, -25, 26, 20, 12, 10, -28, -53, 19, -16, -40, -29, 29, -45, -52, 27, -22, 19, -58, 15, -38, -7, -53, -25, 11, -44, -7, 30, -5, -44, 17, 39, 21, -5, 27, 25, 17, -24, -18, -29, 32, -33, -41, -53, 15, 12, 11, 19, 9, -25, -28, -5, -43, -31, 7, -28, 20, -19, -30, 31, -10, -36, 23, -1, 9}
, {24, 22, 31, -15, 18, 29, -9, -32, -23, -63, 14, 4, 14, 16, 26, 30, -19, -66, 26, 0, 17, 3, -60, 27, 19, -50, -29, -39, -28, -48, 30, -36, -51, 12, 31, 25, -50, -48, -15, 4, -55, 2, 32, 9, 24, -64, -13, 28, 36, 3, -4, -20, -37, -26, -29, -21, -24, -45, -47, 18, -6, -3, 19, 16, -13, -36, 4, -13, -40, 17, 24, 24, -27, 30, 16, -77, -28, 25, -59, 27, 7, 4, -52, 17, 0, -33, -8, -45, 24, 25, -19, 22, -31, -45, 22, -46, -1, -37, -31, -14, 29, -20, 16, -43, -33, 24, -32, 4, -47, 11, -41, -47, 22, -53, 27, -10, 31, 20, 22, -26, 14, -3, -17, -24, -42, 26, -46, -51}
, {-31, -13, -26, -38, -21, -14, 34, -42, 23, -36, 10, 23, 10, -43, -6, 28, -42, -1, -49, 24, 15, -21, 22, 29, -22, 10, 19, 14, -44, -48, 27, 9, -29, -17, 20, -3, 11, 13, -25, -8, 15, -18, -4, 11, -11, -16, -26, -16, 2, -31, 11, -43, 10, -65, -28, 13, 17, -39, -10, 19, 27, 23, -32, 18, 23, 26, -43, 13, 16, 12, 20, 19, 31, -40, -8, -10, 21, -11, -8, 26, -44, 19, 9, 13, -6, -41, 25, 30, 24, 1, 12, -39, 17, 13, -42, -31, 11, -10, 12, 22, 20, 14, -55, 18, 23, 0, -1, 7, -22, 27, 17, -25, 15, -34, -25, -43, -21, 20, -38, 29, 7, -57, 34, -23, -21, -51, -27, 14}
, {-31, -25, -42, 21, -20, -12, -16, 2, 22, -16, -37, 29, -23, -36, -43, -28, -14, -11, 3, 20, -16, 16, 8, -14, -17, -37, -4, 27, -23, -21, -39, 15, -38, 27, 22, -40, -15, 12, -22, 19, -34, -5, -10, 10, -7, -13, 28, -17, 18, 4, 17, 21, 17, 17, -46, -15, -20, -24, -16, -30, 28, 26, 22, -12, 30, 0, -45, 8, -19, 14, 20, -28, -18, 29, -29, -54, 20, 13, -35, -27, -29, 22, 11, -2, -49, 26, -38, -11, 23, -50, 9, -47, -4, 23, -29, 19, -42, 1, -24, 3, 28, 19, -41, 3, 15, -15, 7, 8, -5, -23, -1, -22, -20, -43, 21, -27, 36, 5, -30, -28, 20, -22, -12, 31, 29, -33, 17, 14}
, {13, 18, 23, 18, -19, -40, -8, 21, 21, 14, 15, 16, 7, -21, -24, -37, 4, 24, 16, 20, -32, -41, -35, -50, 20, 27, -25, 26, -23, -12, 25, -30, -18, -32, -54, 26, 21, -54, -14, -58, 20, 20, 25, -36, 25, -40, -37, -49, -15, 9, -25, 25, 19, -10, 22, -43, 22, -26, 10, -27, -19, -45, 15, -60, -5, -12, -26, -26, 19, -51, 16, 26, 10, 4, 25, 14, -55, -14, 4, 8, -40, -41, -33, 17, 6, 20, 16, -42, -32, 32, -52, 12, -2, -9, -11, -33, -16, 28, -46, -30, -45, -38, -33, 20, -38, 21, -66, -7, 7, -64, -43, 25, 18, 12, -17, 11, 9, -19, 23, 30, 23, -22, 21, -33, 16, 21, 9, -29}
, {-18, -50, -8, 15, -1, 20, -36, 24, -25, -15, -45, 3, -31, 20, -23, -48, -25, -35, -29, -42, 16, 5, 6, -53, 14, -26, 5, -25, 31, 17, -36, -35, 13, 22, -2, -29, -54, 0, 27, 17, -27, 3, -23, -17, -52, 22, 29, 22, 21, 6, 12, 11, 24, 18, -37, 26, 20, 20, 18, -14, -24, -39, -13, 14, -34, 10, 15, -59, -45, -2, -29, -11, -19, 31, 21, -43, 11, -30, 26, 6, 11, 25, 14, -59, -14, 27, 19, 18, -29, -70, -30, -44, -30, 21, -36, 20, 13, -37, 15, -3, 26, -59, -12, 25, -15, 24, 1, 13, -8, -35, 19, -35, -37, 5, 18, 11, -15, 14, 23, -8, -43, 17, -24, -23, 26, -9, 15, -56}
, {8, 7, -22, -67, 33, -8, -9, -32, 26, 19, 9, -40, 9, 29, -25, -10, 30, -40, 18, -54, -23, 19, -53, -35, -28, 1, -34, -24, 36, 21, 23, -36, 23, 13, -1, 11, 0, -62, -5, -2, -20, -49, 25, -39, -46, -3, -18, 3, -37, -78, 13, 31, -13, 26, 21, 23, -19, 19, -8, 15, -27, -34, -61, 7, -12, -26, 19, -39, -16, -28, 3, 19, 20, -32, -49, 8, -29, -27, -51, 6, 9, -40, 15, -31, -37, -26, 29, 12, -49, -28, -43, -17, 15, -33, -10, 25, 15, 16, -27, 26, 0, -34, 1, 29, -67, 20, 7, -10, 47, -33, 13, -11, 15, 21, -24, 13, -24, -13, -35, 30, -58, 19, -3, -33, -54, -30, 16, 2}
, {-5, 27, -7, 17, -46, 30, -32, -20, -8, 15, -25, 26, -37, -38, 23, 8, -69, -38, -24, 25, 19, -66, 7, -8, 19, 14, 14, -19, -46, -44, -11, -43, 28, -36, 6, -39, 21, 9, 20, -29, -15, 21, -40, 13, 23, 3, -29, 33, -13, 4, -38, -28, 24, -30, 21, -34, -49, 23, 17, 12, 22, 26, 17, 19, 29, -39, -39, 11, -34, 15, 23, -33, -41, -27, -36, -11, -34, 27, 29, -35, 4, -32, -47, 18, 2, -17, 16, -45, 25, -23, 7, 20, -40, -31, -26, -51, -53, -9, 28, -35, -5, -32, -13, -34, 19, -40, -30, -33, -6, 30, 14, -20, 6, -13, -24, 2, -8, 21, 5, -5, 14, -63, -10, 29, 5, -12, 19, 16}
, {17, -40, 33, 17, 21, -28, -32, 15, -17, -9, -49, 26, 11, 23, 11, 1, -51, 24, 20, 23, 16, 15, 1, 28, -47, -28, 23, -14, 2, 16, -9, 29, -38, -2, -50, 20, 20, 5, 31, 17, -28, -59, 22, 3, -18, 7, -4, 1, -34, -7, -15, -12, -33, 9, -1, 31, 19, -11, -43, -6, 25, -13, -9, -41, 26, 20, 1, 8, 10, -27, -26, -32, -40, 27, -18, -7, 23, 19, -8, 12, 8, 27, 14, -47, -54, 26, 5, -5, 20, -17, 21, 20, 11, -21, -27, 18, 15, -38, -24, 15, -23, 19, 8, -8, 8, -18, 11, -44, -35, -16, 17, 20, 1, 0, -46, 8, -21, -46, 23, -5, -33, 12, -37, -32, -12, -13, 10, 14}
, {-31, -11, 5, 20, -21, -21, -3, 21, -51, -14, 4, -39, -29, -31, 16, -14, 33, 15, 19, 21, -5, -46, 19, -24, 20, 23, -34, -25, 7, 19, -23, 38, 8, -40, -42, 25, 7, 7, -33, -22, -53, 16, 15, -38, 24, 19, 24, -2, 18, 8, -40, -25, 22, 13, 14, -10, -9, -21, 18, -53, -45, 27, 17, -25, -37, -22, -19, 9, 12, -40, -24, 13, -14, 27, 23, 25, -22, 26, -5, -53, 7, 15, 8, 19, 6, 24, 0, 25, -32, -27, 23, -15, 1, 22, 13, -28, -23, 7, -6, -48, -27, 8, 18, -20, -24, -4, -36, -3, -30, 18, -16, 23, -42, -28, -30, -28, -34, -30, 19, 10, 23, 16, 32, -16, 25, 20, -30, 19}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "conv2d_6.h" // InputLayer is excluded
#include "conv2d_7.h" // InputLayer is excluded
#include "conv2d_8.h" // InputLayer is excluded
#include "flatten_2.h" // InputLayer is excluded
#include "dense_4.h" // InputLayer is excluded
#include "dense_5.h"
#endif


#define MODEL_INPUT_DIM_0 28
#define MODEL_INPUT_DIM_1 28
#define MODEL_INPUT_DIM_2 1
#define MODEL_INPUT_DIMS 28 * 28 * 1

#define MODEL_OUTPUT_SAMPLES 10

#define MODEL_INPUT_SCALE_FACTOR 7 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_INPUT_NUMBER_T int8_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

#define MODEL_OUTPUT_SCALE_FACTOR 7 // scale factor of last layer
#define MODEL_OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_OUTPUT_NUMBER_T int8_t
#define MODEL_OUTPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[28][28][1];
typedef int8_t input_t[28][28][1];
typedef dense_5_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "conv2d_6.c"
#include "weights/conv2d_6.c" // InputLayer is excluded
#include "conv2d_7.c"
#include "weights/conv2d_7.c" // InputLayer is excluded
#include "conv2d_8.c"
#include "weights/conv2d_8.c" // InputLayer is excluded
#include "flatten_2.c" // InputLayer is excluded
#include "dense_4.c"
#include "weights/dense_4.c" // InputLayer is excluded
#include "dense_5.c"
#include "weights/dense_5.c"
#endif


void cnn(
  const input_t input,
  dense_5_output_type dense_5_output) {
  
  // Output array allocation
  static union {
    conv2d_6_output_type conv2d_6_output;
    conv2d_8_output_type conv2d_8_output;
    flatten_2_output_type flatten_2_output;
  } activations1;

  static union {
    conv2d_7_output_type conv2d_7_output;
    dense_4_output_type dense_4_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d_6( // Model input is passed as model parameter
    input,
    conv2d_6_kernel,
    conv2d_6_bias,
    activations1.conv2d_6_output
    );
  
  
  conv2d_7(
    activations1.conv2d_6_output,
    conv2d_7_kernel,
    conv2d_7_bias,
    activations2.conv2d_7_output
    );
  
  
  conv2d_8(
    activations2.conv2d_7_output,
    conv2d_8_kernel,
    conv2d_8_bias,
    activations1.conv2d_8_output
    );
  
  
  flatten_2(
    activations1.conv2d_8_output,
    activations1.flatten_2_output
    );
  
  
  dense_4(
    activations1.flatten_2_output,
    dense_4_kernel,
    dense_4_bias,
    activations2.dense_4_output
    );
  
  
  dense_5(
    activations2.dense_4_output,
    dense_5_kernel,
    dense_5_bias,// Last layer uses output passed as model parameter
    dense_5_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif
