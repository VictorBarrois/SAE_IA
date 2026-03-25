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

#define NUMBER_MIN_INT16_T -32768
#define NUMBER_MAX_INT16_T 32767

static inline int32_t min_int16_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int16_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int16_t(
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
static inline int16_t clamp_to_number_t_int16_t(
  int32_t number) {
	return (int16_t) max_int16_t(
      NUMBER_MIN_INT16_T,
      min_int16_t(
        NUMBER_MAX_INT16_T, number));
}
static inline int16_t scale_and_clamp_to_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int16_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int16_t) * 8);
  }
#else
  number = scale_number_t_int16_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int16_t(number);
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


typedef int16_t conv2d_6_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
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
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
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


const int16_t conv2d_6_bias[CONV_FILTERS] = {-78, -6, -5, 1, -11, -14, -2, 6, -5, -9, -3, -7, -5, -25, -7, -15}
;


const int16_t conv2d_6_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{79}
, {106}
, {71}
}
, {{-71}
, {94}
, {27}
}
, {{29}
, {54}
, {-82}
}
}
, {{{10}
, {133}
, {160}
}
, {{112}
, {128}
, {29}
}
, {{-217}
, {-269}
, {-160}
}
}
, {{{-110}
, {58}
, {-134}
}
, {{97}
, {190}
, {118}
}
, {{-2}
, {-10}
, {7}
}
}
, {{{-306}
, {-127}
, {106}
}
, {{-210}
, {139}
, {196}
}
, {{-309}
, {-130}
, {57}
}
}
, {{{-134}
, {-26}
, {104}
}
, {{-168}
, {-62}
, {174}
}
, {{-105}
, {110}
, {155}
}
}
, {{{8}
, {-49}
, {-54}
}
, {{84}
, {166}
, {90}
}
, {{55}
, {105}
, {-127}
}
}
, {{{3}
, {196}
, {173}
}
, {{-107}
, {48}
, {186}
}
, {{-344}
, {-289}
, {-167}
}
}
, {{{-16}
, {-160}
, {-127}
}
, {{-128}
, {-78}
, {-72}
}
, {{161}
, {147}
, {231}
}
}
, {{{111}
, {54}
, {-211}
}
, {{142}
, {-22}
, {-145}
}
, {{170}
, {9}
, {-153}
}
}
, {{{-104}
, {6}
, {-67}
}
, {{21}
, {193}
, {180}
}
, {{-90}
, {29}
, {40}
}
}
, {{{-126}
, {-223}
, {-195}
}
, {{154}
, {121}
, {54}
}
, {{68}
, {82}
, {54}
}
}
, {{{13}
, {-99}
, {-150}
}
, {{-31}
, {138}
, {-7}
}
, {{66}
, {209}
, {82}
}
}
, {{{-66}
, {-13}
, {177}
}
, {{10}
, {100}
, {192}
}
, {{25}
, {-109}
, {-118}
}
}
, {{{-80}
, {76}
, {-124}
}
, {{122}
, {256}
, {-1}
}
, {{-105}
, {100}
, {-2}
}
}
, {{{6}
, {-81}
, {-42}
}
, {{146}
, {-123}
, {-296}
}
, {{262}
, {114}
, {-9}
}
}
, {{{-64}
, {-2}
, {-53}
}
, {{54}
, {96}
, {140}
}
, {{-51}
, {76}
, {59}
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


typedef int16_t conv2d_7_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
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
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
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


const int16_t conv2d_7_bias[CONV_FILTERS] = {139, 13, -96, -18, -35, 16, 79, -44, 25, -4, -15, 34, -25, -15, -22, 68, -59, -55, -1, -15, -36, 84, -14, -52, -1, 6, -2, -6, -45, -13, -38, 19}
;


const int16_t conv2d_7_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{125, 144, 56, -29, -34, 83, 108, -56, 159, 25, 10, 60, 144, 69, 96, 29}
, {100, 136, 17, -50, -8, -14, 80, -3, 87, -18, -68, 36, 132, 17, 86, -13}
, {56, 95, 7, 120, 100, -35, 119, 5, 145, 51, -3, 2, 84, 10, 35, -14}
}
, {{70, -39, -63, -129, -137, -7, 10, -110, 6, -114, -125, -35, -17, -17, -7, -62}
, {-52, -68, -186, -216, -165, -107, 3, -11, -18, -169, -150, -130, -121, -94, -49, -195}
, {-33, -44, -73, 99, 61, -73, 34, 7, 57, -26, 5, -241, 30, -133, -145, -72}
}
, {{-44, -12, 4, 45, -67, -9, 39, 85, -36, -25, -4, -8, -32, -32, -55, -9}
, {-200, -292, -102, 24, -208, 13, -112, 67, -1, -182, 43, -13, -253, -13, 53, -60}
, {-271, -264, -251, -136, -152, -171, -55, -113, 201, -201, -305, -314, -153, -329, -26, -219}
}
}
, {{{-6, -1, 20, 29, 1, 38, 75, 34, 91, 80, 32, 41, 35, 86, 36, 79}
, {-28, -166, -61, -37, -248, 46, -231, -34, 29, -98, 47, 6, -101, -12, 132, -73}
, {-149, -113, -168, -80, -58, -191, -73, 91, -36, -262, -18, -141, -109, -172, 12, -232}
}
, {{58, -52, 75, 128, 204, 74, -73, 60, 150, 98, 47, 57, 65, 15, 106, 57}
, {15, -97, 21, -78, -187, 27, -267, -174, 65, -82, 19, 0, -39, 10, 100, -36}
, {-324, -261, -132, 96, 13, -165, -278, -62, -32, 61, -83, -117, -111, -112, -118, -100}
}
, {{101, -30, 90, 15, 144, 76, -98, -133, 98, 112, -42, 41, 100, 82, 64, 24}
, {-49, -97, -50, -102, -269, -72, -216, -203, -1, -288, -237, -171, -188, -47, -3, -154}
, {-128, -172, -67, 51, 57, -136, -68, 68, -200, 76, -59, -22, -48, -69, -258, 2}
}
}
, {{{-6, -29, 23, 4, 32, 21, -57, 89, 37, 38, 110, 37, -48, 23, 54, 14}
, {-75, 71, -98, -76, -13, -97, 52, 110, -33, -93, 39, -81, 1, -105, 44, -77}
, {-17, 39, -49, -19, 34, -66, 55, 33, 29, -51, -80, -40, -8, -10, 115, -6}
}
, {{58, 58, 42, -58, -16, -26, 99, -39, -2, -14, -31, 70, -5, 40, 87, 17}
, {-64, -44, 21, -148, -129, 23, 51, 75, -37, -18, 135, 51, -4, 42, 109, 14}
, {-68, 33, 16, -164, -38, 14, 48, 38, 17, 25, 50, -41, 6, 18, 55, -13}
}
, {{36, 113, 58, -231, -67, 32, 34, -47, -83, 38, 50, 15, 52, 53, -17, 22}
, {95, 114, 36, -206, -84, 86, 159, -7, -68, 0, 119, 98, -7, 36, 70, 23}
, {-23, 77, 29, -235, -41, 0, 170, 38, -59, 34, 162, -15, -11, 17, 54, 14}
}
}
, {{{-24, -220, -26, -58, -24, -28, -202, 40, 35, 9, -11, 26, -10, -9, 49, -5}
, {-47, -174, -50, 22, 36, -6, -212, -19, 56, 2, -17, -15, 2, -58, -44, -21}
, {-17, 28, 12, 54, 62, -35, -111, 76, 11, -15, 105, -1, -15, 2, -3, -2}
}
, {{2, -66, 18, 56, -41, 2, -62, -12, 123, 18, -49, 18, 13, 50, 40, -17}
, {9, -245, -23, 141, 10, 19, -223, 1, 98, -31, -20, 44, -54, -33, 160, -18}
, {-35, -45, -89, -28, -103, -34, -144, -92, 53, -88, -98, -96, -34, -48, 53, -79}
}
, {{-2, -104, -21, 30, 7, -1, 67, 33, 11, -35, -166, 28, 9, -1, 53, 13}
, {52, -119, 15, 95, 41, 50, 48, 1, 119, 55, -42, 90, 1, 22, 156, 66}
, {20, -292, 50, -37, -136, 28, -128, 103, 113, -3, 160, 128, -38, 25, 208, 14}
}
}
, {{{26, -62, 19, -77, -86, 26, -25, 23, 47, -59, 20, 38, -36, 3, -6, -18}
, {-48, -72, -99, -16, -67, 25, -57, 70, 15, -138, 8, 35, -96, -46, 44, -42}
, {-91, -64, -241, 164, -107, -133, 17, -156, 20, -182, -210, -211, -102, -190, -114, -178}
}
, {{66, 65, 16, -164, -49, 10, 157, -34, -12, 58, 99, 85, 8, 27, 24, 44}
, {21, 99, 89, -40, -175, 114, 223, 9, -12, 143, 86, 95, -2, 79, 49, 96}
, {-118, -156, -58, -174, -208, -45, -10, 17, 173, -129, 105, -6, -177, -6, 192, -25}
}
, {{22, -52, 15, -111, 24, -29, 40, -47, -67, 28, 53, -38, 35, 6, -81, -4}
, {8, 78, 21, 9, 37, 19, 85, -47, 1, 66, 78, 4, 86, 35, 54, 50}
, {-8, -68, 106, -215, -119, 80, -2, -38, 7, 12, 62, 42, -52, 118, 30, 40}
}
}
, {{{9, -112, -10, 118, 31, 21, -181, -16, 102, -3, -67, 28, 80, 14, -8, 48}
, {58, -5, 40, 57, -12, 38, -79, -65, 47, 24, -5, 0, 8, 43, 9, 40}
, {14, 41, -44, -145, 21, 11, 25, 87, 6, 34, 25, 12, 11, 29, 53, 28}
}
, {{-21, 0, 32, 176, -24, 28, 78, -164, 83, -47, -74, -16, 34, 24, 87, 20}
, {44, -8, 39, 223, 61, -4, 29, 6, 21, 98, 51, 87, 12, 67, -22, -9}
, {59, -3, 64, 86, 66, 76, -15, 41, -38, 107, 80, 38, 82, 69, -38, 55}
}
, {{-9, -71, -146, 19, -150, -95, 182, -166, 9, -156, -126, -163, -93, -139, -26, -157}
, {15, 123, -134, -165, -176, -148, 180, -11, -131, -162, -84, -236, -23, -169, -232, -246}
, {-72, 161, -207, -230, -198, -179, 13, -40, -115, -244, -186, -244, -116, -156, -248, -209}
}
}
, {{{-28, -20, 11, 104, 7, -21, 91, -101, -45, 37, -134, -80, 21, 20, -79, 25}
, {59, 53, 56, 194, 102, 54, 22, -19, -7, 53, 87, 64, 62, 69, 42, 39}
, {105, 138, 124, -16, 82, 155, 43, 8, -55, 91, 49, 53, 156, 112, -37, 47}
}
, {{-95, -15, -219, 2, -71, -205, 223, 23, -219, -148, -258, -278, -24, -236, -227, -168}
, {26, 139, -90, -67, -144, -146, 164, -16, -291, -155, -197, -342, 134, -139, -317, -129}
, {-19, 151, -73, -241, -132, -156, 103, 61, -161, -178, -137, -197, -78, -73, -243, -173}
}
, {{-171, -173, -98, -191, -113, -146, -131, 68, -68, -57, 59, -40, -160, -84, -29, -16}
, {-261, -320, -72, -61, 81, -137, -353, 121, -30, -33, 170, 33, -215, -88, -24, -58}
, {-74, -492, -51, -39, -17, -22, -484, 64, -108, -11, 124, 59, -100, -44, 12, 18}
}
}
, {{{51, -44, -27, 73, 6, -40, 24, -65, -149, 12, -114, -1, 20, 4, -255, -44}
, {48, -118, 52, -36, -1, 30, -68, -39, 49, 49, 27, 76, 34, 78, -6, 8}
, {-56, -103, -39, 130, -185, -15, -45, 36, 27, -27, 27, -2, -2, -68, 44, -30}
}
, {{-11, -14, -103, 117, 66, -27, 0, -135, -4, -41, -144, -72, 40, -78, -141, -33}
, {78, -91, 31, 109, 2, 60, 25, -227, -47, 108, -159, 60, 71, 82, -7, 80}
, {28, -82, 61, -84, -298, 73, -75, -24, 21, 20, 42, 55, -60, -6, 57, -33}
}
, {{-102, -12, -100, 21, -37, -96, 23, -100, 47, -17, -247, -84, -4, -129, -26, -34}
, {62, -114, 2, 121, 113, -13, 80, -221, -6, 63, -203, 28, 81, 53, 47, 56}
, {50, -166, -13, -72, -168, 86, -106, -128, 111, -32, 9, 70, -27, 36, 186, 38}
}
}
, {{{47, -44, 28, -38, -77, 25, 62, -21, -43, 39, 51, -11, 24, 39, 29, -3}
, {-82, -97, -98, -159, -311, -47, -27, -4, -36, -162, 58, -34, -110, -65, 36, -79}
, {-28, -71, -82, 72, 56, -86, 125, 9, -10, -15, -75, -133, 32, -113, -55, -90}
}
, {{24, -109, 85, 53, 55, 56, 90, 39, 25, 84, 109, 79, 2, 65, 49, 59}
, {17, -490, 19, -18, -142, 72, -368, -8, 38, -39, 62, 17, -27, -2, 109, -15}
, {-273, -228, -150, 270, 42, -136, -140, -59, 125, -151, -100, -123, -64, -152, 111, -145}
}
, {{9, -94, 68, 146, 94, 56, 26, 13, 95, 48, 40, 51, 50, 57, 85, 24}
, {73, -106, 88, 33, 24, 131, -155, 6, 130, 61, 31, 55, -12, 71, 142, 22}
, {-88, -169, -55, 44, -37, 14, -155, 59, 112, -105, -56, -22, -50, -90, 125, -41}
}
}
, {{{-48, 42, -34, -53, 7, -31, 95, 33, -133, -25, -31, -48, -78, -32, -59, -101}
, {20, -129, -33, -62, 8, 11, -35, -10, -245, 7, -78, 7, -5, -48, -284, 59}
, {60, -108, 25, 25, 7, 51, -240, -16, -10, 18, -30, 30, 47, 55, -9, 17}
}
, {{-58, -155, -21, 69, -55, -54, -147, 72, -216, -11, -85, -74, -61, -74, -165, -20}
, {24, -246, 47, 129, 41, 40, -249, 13, -57, 80, -90, 30, 50, 66, -195, 38}
, {69, -50, 59, -40, -33, 35, -262, -140, 162, 54, -94, 63, -45, 89, 48, 3}
}
, {{6, -117, -49, 87, 43, -27, 6, -24, -268, -21, -87, 17, 8, -6, -249, 25}
, {22, -30, 6, 169, 51, 1, 12, -63, 64, -11, -129, 22, 74, 72, 3, 83}
, {27, -49, 37, -25, -90, 68, -108, -196, 53, 40, -150, 51, 0, 0, 44, 29}
}
}
, {{{-6, 6, 9, -4, -72, -3, 46, -39, 6, 3, 70, 35, -9, 63, 62, -3}
, {20, -132, -22, -4, 47, -43, -63, 13, 54, -22, 28, 44, -17, 13, 83, 7}
, {-15, 21, -17, 17, 19, -35, 29, 15, -34, -20, 94, -4, -6, 0, 17, 20}
}
, {{57, -113, -30, -65, 81, -10, -102, -72, -32, 78, -26, 10, 10, 35, -77, 47}
, {100, -166, 11, 34, 77, 62, -113, -29, 68, 46, 109, 67, 63, 49, 16, 29}
, {35, -3, -13, -126, -85, 39, -28, -98, 27, 31, -84, -26, 55, 20, 63, 40}
}
, {{60, -157, 19, 11, 79, 36, -229, 28, 10, 21, -114, 72, 2, 24, -29, 1}
, {36, -111, 23, 74, 46, 23, -320, -181, 65, 28, -197, 44, -14, 42, 27, 45}
, {20, -1, -2, 30, -96, -40, -87, -102, 125, -74, -173, -67, -45, 5, 43, -24}
}
}
, {{{88, -2, 54, 193, 68, 49, -1, -9, 116, 59, -60, 33, 7, 87, 111, 47}
, {19, -5, 13, -75, -22, 36, -47, -61, 52, -74, -310, -108, 31, -9, -1, -81}
, {-23, -34, -57, -94, -27, -89, -81, -17, -118, -84, -161, -64, -78, -81, -362, -12}
}
, {{38, 76, 81, -107, -235, 82, -51, -131, 83, -156, -49, -6, 87, 67, 130, -24}
, {-189, -135, -156, 20, 80, -264, -102, 53, 220, -66, -313, -117, -229, -164, -159, -102}
, {-17, -224, 19, 33, 12, 54, -161, 136, -135, 73, 83, 50, 44, 17, -125, 71}
}
, {{-182, 47, -178, -59, -44, -272, -136, -116, 15, -215, -323, -277, -305, -163, -221, -228}
, {-29, -268, 21, 134, 96, -46, -128, 18, -101, 71, -65, 18, -21, -44, -444, -34}
, {45, 4, 88, 103, 107, 51, 1, 94, 145, 81, 45, 42, 55, 73, 50, 38}
}
}
, {{{14, 21, 6, 107, 68, -44, 33, -135, -79, -72, -104, -125, 20, -53, -53, -97}
, {16, 23, 9, -2, 95, -18, 119, -88, -106, 93, -75, -14, 94, 6, -6, 71}
, {20, -53, 26, 124, 104, 65, 73, -31, 91, -8, -19, 66, -3, 62, 83, 57}
}
, {{-172, -66, -174, 12, -59, -131, -58, -51, 62, -273, -32, -98, -127, -145, 85, -123}
, {-7, 37, 16, -33, 48, -114, 113, -222, -198, -8, -261, -92, 61, -70, -71, -54}
, {73, 53, 46, 194, 47, 20, 209, -165, 62, 100, 2, 63, 112, 126, 114, 92}
}
, {{-140, -78, 7, -132, -98, 14, -178, 25, 24, -49, 141, 45, -176, 7, 43, -36}
, {-152, 21, -123, -107, -20, -66, 149, 1, -25, -60, 27, -54, -7, -167, 71, -76}
, {1, 89, 59, -8, 53, -17, 194, -79, -123, 30, -58, -44, 50, 5, 8, -11}
}
}
, {{{-16, -5, -5, -84, 15, -17, -53, 46, -4, -3, 87, 1, -27, -17, 39, -21}
, {-67, -136, -17, 137, -14, -1, -59, 100, 58, -36, 83, 35, -52, -16, 35, -7}
, {-33, -105, -71, 86, -106, 20, -82, -70, 49, -97, 144, 1, -61, -39, 147, -57}
}
, {{5, -114, 15, 3, 10, 27, -76, -20, 64, 23, -52, 26, 3, -39, 118, 16}
, {20, -136, 15, 6, -5, 28, -44, -44, 37, 42, -27, 60, -23, 19, 117, 32}
, {-51, -76, 61, 32, 7, 17, -93, 83, 42, 71, 145, 75, -114, 62, 53, 43}
}
, {{28, -7, -9, -188, -21, 8, -33, 5, -2, 28, 13, 5, 50, -22, -35, 24}
, {-15, -18, 40, -81, 5, -11, -114, -40, -74, 48, -92, 36, 43, 33, -72, 94}
, {78, -10, 4, -23, -7, 46, 19, -12, -140, 51, -31, 12, 81, 30, -83, 60}
}
}
, {{{-11, 41, -19, -9, -41, 6, 25, 31, -57, -1, -1, 71, -31, -19, -48, -10}
, {-15, 78, 8, -139, -60, -17, 62, -17, -227, 10, 69, 29, 10, 43, -128, 23}
, {-5, -38, 21, -94, -11, -10, 27, 2, -88, -6, 30, 21, -68, 45, -51, 34}
}
, {{-16, 63, -53, -13, -10, -70, 93, -64, -162, -72, -139, -156, -4, -89, -173, -75}
, {-45, -13, -33, 39, 40, -55, -1, 57, -260, 21, -55, 24, -24, -24, -298, -35}
, {72, -74, 37, 113, 133, 88, -28, 0, -96, 90, 20, 85, 121, 45, -82, 65}
}
, {{-105, -105, -85, -28, 76, -76, -134, 73, -132, -14, -20, -22, -46, -89, -249, -28}
, {58, -132, 64, 85, 79, 23, -84, 8, -236, 8, -46, 7, 33, 22, -142, 37}
, {23, 21, 28, 97, 51, 30, -32, -89, 123, 40, -170, -20, 59, 33, 100, 37}
}
}
, {{{44, 5, 58, 51, 119, 34, -16, 70, 54, 24, 26, 60, 116, 18, 48, 50}
, {77, 22, 30, -21, 122, 63, -52, -65, 15, -8, -96, 63, 84, 30, 60, 43}
, {33, -9, -49, 63, -66, 10, -70, -228, 167, -64, -86, -57, -59, -12, 135, -63}
}
, {{37, 41, 71, 101, 47, 46, -41, 50, 57, 68, -50, 74, 83, 47, -12, 62}
, {77, 76, 32, 111, -116, 36, -90, -333, 36, 23, -116, -63, 114, 93, 27, -5}
, {-118, -22, -119, 22, -184, -83, -110, -79, 141, -336, -294, -230, -149, -80, -29, -170}
}
, {{2, 128, -9, 31, -139, -28, 58, -117, 0, -94, -241, -115, -65, 6, -26, -68}
, {-124, 131, -91, -94, -215, -136, 41, -38, -58, -234, -266, -126, -175, -99, -142, -146}
, {-226, -171, -71, 122, -140, -129, -159, -27, -38, -116, -39, -63, -139, -65, -133, -76}
}
}
, {{{-4, 81, 20, -20, -50, 53, 24, -124, -9, 5, 138, 2, 28, 11, -27, -28}
, {9, 14, -40, 71, 40, -19, -53, -51, -114, -60, -96, -51, 3, -41, -186, -2}
, {11, -60, 14, -33, 60, -20, -74, 79, -16, -45, -28, 37, -81, -34, -91, -23}
}
, {{-62, -201, 0, -56, 9, -113, -290, 69, -164, -64, 0, -9, -118, -72, -102, 11}
, {-2, -228, 52, -95, 136, 65, -299, 122, -208, 102, 203, 91, 40, -28, -226, 75}
, {64, 78, 47, -221, 90, 85, -41, -15, -43, 37, 127, 49, 124, 76, -76, 32}
}
, {{64, 10, 63, -137, 50, 106, -50, 105, -173, 76, 86, 102, 109, 27, -138, 37}
, {73, 91, 11, -22, -23, 29, -73, -187, -27, 17, -52, -15, 84, 20, -93, 9}
, {16, 4, 6, -190, -144, -34, -103, -146, -15, -115, -241, -71, -77, 1, -57, -40}
}
}
, {{{73, -82, -9, 53, -44, 22, -28, -81, 57, -39, -78, 11, -28, 14, 64, -23}
, {30, -87, -46, -106, -45, -25, -130, 1, -46, -36, 9, -4, 3, 36, -75, 36}
, {-34, 6, -40, -112, -55, -10, 8, 19, -99, 13, 88, 52, 7, 33, -21, 72}
}
, {{53, 66, 36, 35, -44, 5, 63, 15, -88, 13, 11, 25, 9, 11, 28, 59}
, {43, 104, 42, -134, -64, 47, 86, -47, -67, 40, 11, -4, 30, 45, 80, 28}
, {-2, 95, -8, -318, -79, -21, 171, 14, -10, 26, 3, 25, -8, -54, 9, -9}
}
, {{-6, 90, -68, 51, -11, -82, 136, -318, -88, -50, -387, -159, 34, -45, -13, -38}
, {57, 87, 43, -18, -95, 31, 211, -147, -66, 42, 15, 14, -2, 25, -2, 13}
, {22, 20, 60, -138, -71, 44, 53, 12, -12, 47, 117, 18, 15, 52, 35, 39}
}
}
, {{{-44, -25, -77, -59, 59, -38, -34, -13, -108, 33, -32, 13, -13, 22, -86, -21}
, {21, -2, 15, -24, 72, 24, -85, 81, -51, 21, -15, 34, 59, 18, 4, 44}
, {89, 75, 60, -27, 20, 86, -31, -137, 34, 71, 14, 74, 181, 87, -22, 74}
}
, {{35, -6, 57, -20, 83, 13, -121, 138, -28, 61, 31, 39, 54, 34, -35, 6}
, {33, 107, 27, 68, 63, 39, -52, -66, 13, 36, 53, 27, 107, 20, 60, 17}
, {-61, 24, -23, -216, -213, 0, -99, -360, 66, -192, -248, -151, -24, -37, 58, -131}
}
, {{27, 126, 30, -20, -71, 22, -28, -154, 46, -5, -120, -15, 51, 24, 6, 21}
, {-24, 97, -89, -165, -348, -125, -23, -102, 11, -220, -109, -113, -89, -38, -96, -154}
, {-152, -101, -53, 32, 29, -174, -135, 46, -52, -103, 92, -61, -188, -97, -92, -34}
}
}
, {{{28, -16, 31, 5, -48, 26, 5, -14, -62, 8, 36, 46, -8, 15, 37, 30}
, {-9, 44, 52, 20, 16, 58, 189, 1, -97, 62, 102, 66, 67, 16, 18, 49}
, {27, 33, 16, -209, -90, 54, 163, 75, -167, 77, 90, 67, 32, 110, 30, 56}
}
, {{38, 32, -57, 62, -30, -33, 73, -61, -35, -35, -188, -63, -9, -21, 3, -39}
, {29, 78, -26, -76, -93, -49, 151, -157, -263, -51, 225, -125, 47, -8, -171, -44}
, {70, 76, 56, -21, 9, 60, 120, -83, -82, 66, 206, -12, 29, 67, -79, 32}
}
, {{-85, 54, -29, -65, -73, -48, 39, 72, -30, -130, 204, -12, -69, -75, -15, -45}
, {-142, -219, -210, -39, -8, -164, -214, 12, -189, -93, -23, -70, -200, -195, -72, -75}
, {20, -62, -70, -11, 36, -14, -9, 11, -83, -55, 19, -31, 23, -78, -112, 11}
}
}
, {{{23, 41, 46, -38, 27, 17, 15, 12, -14, 4, 64, -22, -11, -13, -37, -18}
, {38, 33, -22, -146, -22, -32, 68, 73, -24, -30, -11, 13, -28, 25, -32, -10}
, {-9, -28, -2, -70, 44, -6, 19, 55, -110, 23, 64, 35, 54, -39, -24, 33}
}
, {{44, 106, 26, -131, 40, 35, -41, 102, -8, 5, 15, 43, 30, 4, -12, 3}
, {85, 69, -13, -126, 78, 25, 6, 97, -47, 42, 77, 68, 40, 29, -130, 58}
, {73, 164, 16, -50, -1, 21, 54, -180, 16, 95, -60, -81, 89, -12, -26, 47}
}
, {{-4, 33, -10, -18, 12, -3, -10, -71, 52, 9, -51, -34, -20, -1, 19, -16}
, {44, 23, -6, -13, -70, -7, 42, -205, 41, -13, -266, -87, 49, -47, 42, -42}
, {-4, -92, -103, 88, 20, -124, -3, -33, 31, -88, -187, -60, -116, -111, 60, -95}
}
}
, {{{24, 46, 53, 33, 30, 34, -22, -18, 92, 16, 8, -1, 25, 37, 114, 34}
, {41, 78, 65, 224, 62, 5, 97, 17, 73, 4, -59, 6, 11, 75, 39, 12}
, {79, -4, 43, -17, 108, 45, 28, 68, 2, 46, 65, 47, 37, 4, -3, 71}
}
, {{-2, 88, 5, 176, -79, -46, 158, -68, -42, 104, -23, -41, -9, 6, -142, 35}
, {22, -31, 51, 180, 5, 90, 68, 37, -39, 100, 210, 36, 44, 23, -55, 56}
, {25, 72, 23, -3, -100, 104, -31, -179, 3, 23, 51, -75, 71, 34, -105, -37}
}
, {{-87, 15, -182, -96, -130, -198, 184, 115, -115, -296, 147, -120, -130, -162, -63, -149}
, {-56, -20, -312, -196, -168, -326, 123, 119, -92, -454, 18, -147, -140, -297, -164, -326}
, {-148, -182, -279, -404, -182, -244, -186, 198, -90, -283, 100, -54, -296, -242, -245, -188}
}
}
, {{{12, -15, -71, -34, -59, -22, -63, 69, -33, -101, 23, 38, -75, -20, 65, 21}
, {-88, -6, -109, -105, -57, -88, -23, 116, -59, -211, -18, 21, -198, -67, 13, -31}
, {-114, -108, -150, -155, -35, -87, -10, 127, -95, -94, 44, 44, -192, -74, 17, -79}
}
, {{48, 97, 59, 31, -46, 51, 26, 76, -143, 76, -24, 63, 118, 46, -42, 48}
, {100, 103, 54, 47, -2, 49, 143, 33, -57, 96, 100, 98, 75, 86, 19, 52}
, {47, 159, 88, 5, 35, 75, 79, 39, -88, 104, 41, 32, 89, 106, 24, 102}
}
, {{-24, 103, -48, -30, -120, -62, 87, -193, -76, -72, 6, -126, 30, -91, -35, -76}
, {-6, 52, -70, 31, -85, -105, 79, -173, -113, -122, -350, -117, -4, -34, -127, -67}
, {50, -3, -54, 1, -25, -37, 78, -96, 13, -78, 76, -57, 23, -49, -21, -77}
}
}
, {{{-14, -1, 3, -18, -56, -29, 4, -5, -18, 27, 53, -8, -15, 43, 32, -3}
, {-4, -122, 1, -32, -8, 35, -96, -7, -13, -49, -21, 39, -58, 6, 125, -21}
, {-28, -13, -39, 114, -19, -61, 6, 1, 75, -112, -18, -25, -27, -53, 153, -79}
}
, {{3, 16, -34, -7, -52, -55, 16, -99, -81, 33, -65, -40, 20, 17, -128, 6}
, {41, -4, 61, -38, -49, 17, 120, -74, -85, -38, 25, 20, 54, 39, 33, 52}
, {-18, -26, 76, -216, -94, 78, 103, 4, 80, 18, 114, 60, -14, 121, 67, 32}
}
, {{-16, -4, -21, -145, -83, 8, -39, -145, 2, -65, -107, -35, -76, -8, 28, -41}
, {3, 188, 7, -48, 7, -67, 206, -281, -189, -23, -380, -127, 38, -65, -109, -58}
, {37, 124, 69, -75, -12, 26, 197, -111, -32, 27, 177, 25, 32, 78, 32, 54}
}
}
, {{{-12, -46, -83, -6, 22, -42, 47, 5, -253, -13, -17, -18, 14, -48, -395, 0}
, {87, -191, 65, -11, 52, 104, -160, -46, 2, 70, 50, 43, 20, 44, -54, 39}
, {19, 44, 41, 47, -205, 58, -10, -209, 33, -50, -68, -55, 28, 2, 120, -9}
}
, {{-50, -164, -49, 57, 48, -54, -110, 64, -383, 4, -13, -19, 40, -47, -279, -13}
, {84, -219, 112, 185, 85, 84, -379, -168, 126, 116, -35, 89, 69, 103, 43, 66}
, {-45, -116, -19, 29, -342, -22, -197, -106, 72, -179, -9, -33, -63, -58, 102, -60}
}
, {{-37, -186, -21, 139, 15, 58, -179, -14, -49, 5, -69, 20, 81, -24, -29, 66}
, {62, -71, 29, 111, 24, 86, -56, -218, 89, 50, -187, 49, -26, 50, 76, 23}
, {-24, -36, -85, -85, -82, -54, -72, -2, -9, -89, 32, -53, -17, -76, 5, -74}
}
}
, {{{-109, -43, -99, 221, 79, -132, 143, -86, -167, -42, 72, -219, -23, -143, -134, -101}
, {60, 128, 50, 119, -11, 19, 96, -173, -140, 110, -134, -210, 113, 14, -276, 36}
, {16, 94, 23, 110, -38, -9, 84, -161, -29, -40, -229, -130, 82, 25, -192, -14}
}
, {{-170, -14, -247, -164, -90, -241, 36, 171, -120, -182, 201, -64, -285, -212, -78, -143}
, {-221, -36, -305, -337, -77, -218, 11, 69, -168, -228, 39, -55, -302, -214, -43, -106}
, {-111, -178, -95, -173, -44, -81, -135, 91, -174, -74, 139, -1, -173, -133, -124, -45}
}
, {{41, -17, 53, 38, 55, 72, -45, 32, 33, 31, -11, 30, 61, 33, 41, 50}
, {40, 66, 35, 38, 84, 79, 49, 85, 66, 98, 59, 81, 10, 63, 63, 65}
, {46, 118, 62, 6, 50, 72, 35, 118, -48, 25, 25, 73, 55, 49, 45, 35}
}
}
, {{{-215, -52, -212, -44, -96, -115, 40, 40, -51, -318, 27, -54, -258, -197, -15, -207}
, {-63, -17, -189, -99, -201, -241, 92, -27, -229, -305, -190, -280, -93, -288, -166, -151}
, {23, 95, -166, 55, -88, -177, 161, -181, -159, -143, -340, -206, 0, -164, -46, -70}
}
, {{-24, -8, 81, -18, -32, 71, -107, 101, -31, 24, 123, 40, -5, 49, -22, 67}
, {-82, -140, -32, 37, -45, 23, -155, 66, -53, -13, 93, 72, -73, -7, 41, -26}
, {-94, -81, -56, -74, -53, -31, -2, 81, -3, -8, 7, 71, -114, -19, 124, -35}
}
, {{25, 107, -7, 30, 52, 17, 34, -16, 46, -7, 96, -15, 102, 37, 8, 32}
, {112, 81, 100, -6, 21, 48, 123, 32, 10, 33, 23, 92, 88, 67, -14, 37}
, {14, 119, 21, -84, -16, 17, 156, 58, -23, 37, 100, 56, 26, 47, 47, 81}
}
}
, {{{7, 13, 8, -65, 1, -31, 50, -22, -85, -26, 25, 40, 9, 35, -10, 18}
, {17, -94, -15, -19, -1, 10, 28, -3, 67, 19, 85, 44, 6, 21, 119, 48}
, {-1, -154, -39, 41, -15, -10, -68, 29, 52, -30, 22, 3, -52, 1, 56, 8}
}
, {{-44, -48, 16, -76, 35, -20, -14, -66, -201, -16, -28, -22, -28, 19, -184, -43}
, {68, -107, 12, -58, 35, 10, 9, 20, 25, 37, -38, 74, 32, 64, -4, 46}
, {-15, -74, 55, -75, 54, 72, -65, -9, -67, 61, 36, 48, -17, 68, 43, 30}
}
, {{-25, -81, -29, -131, 57, 46, -89, 170, -174, -17, 15, -24, 44, -85, -204, 26}
, {76, -59, 54, -18, 53, 41, -102, -58, 22, 78, -145, 9, 28, 13, -92, 4}
, {90, -45, 49, -52, -36, 44, -38, -162, -12, 38, -129, -43, 48, -26, 61, 29}
}
}
, {{{-48, 23, -7, -23, 38, -41, 54, 46, -11, -33, 36, -37, -21, -16, 51, -13}
, {2, 6, -5, -180, 44, 3, 15, 116, 53, -13, -2, 30, -94, -6, 57, 24}
, {-33, -58, -16, -113, 68, -11, 9, 102, -72, 56, -11, 9, 3, 8, -14, 17}
}
, {{14, 101, 33, -100, 29, -4, -95, 58, -61, 15, 104, 26, 2, -7, -75, 18}
, {14, 133, 77, -110, -4, 11, 19, 30, -28, 15, -41, -12, 129, 46, -17, 3}
, {16, 102, 51, -219, -52, 36, 33, 57, -63, -2, 28, -1, 85, 27, 6, 39}
}
, {{31, 56, -18, -40, -46, 41, -83, 27, 23, 25, 46, -2, 14, 9, 11, -10}
, {-24, 140, 24, 3, -64, -24, 121, -60, 23, -10, -40, -72, 39, 33, -27, -38}
, {-47, 114, -62, -4, -11, 1, 187, 61, 42, -42, -53, -28, 40, -38, 80, -3}
}
}
, {{{-25, -50, -35, 118, 18, 13, -22, 10, -80, -36, -26, 20, -17, -16, -70, 11}
, {57, -6, 25, -30, 12, 30, 5, -32, -74, -2, -103, 15, -17, 3, -18, 0}
, {17, -61, 5, -105, -1, -52, -8, 59, -84, -9, 45, 3, -64, -34, -5, -4}
}
, {{96, -60, 29, 23, -10, 46, 21, -1, -136, -47, -10, 44, 25, -35, 43, 14}
, {29, -21, 43, -31, -18, 56, -62, 40, -163, 21, -66, -10, 14, -28, -138, 68}
, {79, -5, 11, -34, -6, 84, 20, 30, -204, 88, 14, 90, 88, 40, -155, 58}
}
, {{13, 117, 27, -16, 29, -14, 50, -29, -101, 40, 91, 15, 23, 20, -64, -19}
, {-11, 72, -28, 85, 11, -19, 55, -170, -32, 4, -40, -121, 30, -9, -47, 12}
, {89, 48, 19, -2, 41, -51, 101, -153, -57, -51, -178, 60, 24, 30, -13, 13}
}
}
, {{{1, 2, 7, 93, 93, 12, -100, 48, -5, 36, 59, -16, 58, -30, -61, 24}
, {-1, 31, 16, 66, -24, -46, -93, -104, 42, -64, -150, -123, 22, 14, -77, -57}
, {-57, -80, -46, 10, 18, -60, -65, -31, 2, -47, -88, -25, -115, -55, 12, -41}
}
, {{-19, -100, -22, 54, 22, -8, -189, 54, 4, -20, -68, -4, -3, 33, 84, -48}
, {-35, -220, -87, -176, 11, -52, -286, 190, -12, -63, -6, 26, -76, -66, -71, -12}
, {-8, 35, 1, -70, 89, 29, -235, 89, -58, 81, 125, 63, 23, 0, -43, 15}
}
, {{58, 53, 34, -111, 66, 4, -55, 119, -22, 49, 33, 41, -35, 6, -36, 21}
, {76, 117, 78, -89, 48, 92, -94, 193, -96, 122, 38, 94, 117, 73, -32, 76}
, {38, 11, 22, -73, 9, 51, -47, 16, 6, -1, -38, -10, 33, 32, -43, -1}
}
}
, {{{-64, 3, -48, 51, -173, 25, 106, -7, 47, -61, 270, 98, -38, -13, 101, 5}
, {-117, -122, -107, 71, 10, -220, -6, -15, 135, -116, -65, -169, -49, -152, 41, -119}
, {39, -104, 29, 46, 115, 19, -23, 84, -177, 71, -73, 46, 26, 68, -268, 41}
}
, {{52, -73, 91, -163, -40, 77, -155, -63, 97, -73, 95, 6, 14, 82, 111, 58}
, {-170, -240, -108, 165, 8, -130, -64, -69, 49, -17, -181, -143, 39, -133, -40, -58}
, {55, -101, 76, 218, 94, 47, 4, 41, -62, 106, -149, 30, 14, 36, -17, 54}
}
, {{4, 1, -57, -125, -84, -62, -100, -54, 49, -115, -80, -22, -131, -16, -13, -111}
, {-83, -43, -79, 14, -55, -147, 96, -1, -49, -44, -117, -73, -37, -120, -184, -42}
, {97, -4, 29, 115, 53, 60, 43, -65, 9, 85, -127, 44, 86, 46, -8, 29}
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


typedef int16_t conv2d_8_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
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
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
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


const int16_t conv2d_8_bias[CONV_FILTERS] = {85, 24, 68, 29, 105, 52, 20, 30, -10, -21, -25, 42, 52, -36, 28, 46, 2, 52, 7, -33, 61, -18, 49, -42, 85, 45, 52, -13, 52, 97, 74, -19, 3, 82, 105, -44, 77, -20, -51, 42, 15, -46, -13, 3, 13, 48, -31, 113, -37, 14, 7, 23, 38, 3, 3, -27, 8, 3, -10, 13, -4, -44, 33, 3}
;


const int16_t conv2d_8_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-39, -29, 44, 23, 69, 27, -147, -45, 2, -52, -7, -25, -50, 23, 3, -10, -18, 0, -64, -55, 30, -22, 12, 62, -20, -32, -7, 46, 20, 27, -3, -5}
, {90, -18, 40, 50, 36, 47, 59, 33, 26, 44, 28, -17, -84, 11, -78, 14, 15, 9, 6, 38, 35, 70, 45, -7, 8, 5, -46, 4, 12, -17, 5, -55}
, {12, -52, -100, -144, -254, 16, 129, -110, -27, -90, -121, 28, 9, -113, 60, -80, -131, -11, -15, 73, 7, 72, 5, -36, -104, 35, -170, -136, -14, -23, -215, -26}
}
, {{-21, 2, 4, 22, 135, -86, -60, 1, 188, -20, -3, 74, 80, -7, 22, 30, -71, -1, 105, -12, -26, -59, -26, 49, 58, 22, 30, 22, -42, -8, -28, 61}
, {35, 105, -14, 4, -40, -89, -214, 83, 46, 53, 10, 120, 57, 30, 62, 73, -115, 17, 29, -90, -124, -167, -145, 71, 37, 81, 1, 126, -75, 16, -7, 51}
, {4, -65, -43, -38, -54, -90, 74, -11, -61, -143, -56, -72, 118, -114, -113, -38, -167, -128, -41, -36, -102, 38, -187, -185, -78, -27, -71, -153, -129, -139, -110, 49}
}
, {{60, 82, 19, -17, 38, 7, 15, 73, 13, 4, 8, 71, 0, 39, 42, 53, 35, -19, 30, 5, 46, 39, 57, -16, 64, 100, 27, -27, 50, -1, 29, 75}
, {94, 136, 2, 35, 39, 29, -121, 50, 139, 53, -8, -22, 83, -5, 41, 5, -75, -29, -58, 10, -4, 27, -16, 36, 45, -235, -120, 70, -37, -11, -83, 98}
, {-24, 20, 22, 23, -22, -129, 29, 9, -29, -100, -10, 11, -40, -10, -50, -105, -58, -72, -191, -160, -75, -63, -44, -3, -7, 11, -4, 6, -91, -76, -11, -12}
}
}
, {{{-102, -7, -49, -45, -105, 2, 23, -89, -140, 32, -41, 30, 61, -36, 56, -37, 40, 40, -5, -43, -52, 59, 20, -17, -9, -33, -19, 14, -25, 38, -38, 52}
, {-87, -63, 3, 20, -87, 7, -10, -12, -47, 75, 44, -5, -80, 11, -6, -46, 60, 18, -2, -72, -20, -84, -18, 53, 0, -11, -58, 65, 16, 46, 11, -28}
, {25, -6, 38, 73, 1, -24, 19, -49, -38, 8, -28, -190, -268, -86, -119, 33, 75, 16, 115, -98, 64, -25, 64, -59, -29, 55, 55, -143, 69, -79, 86, -183}
}
, {{-34, -245, -31, -76, -152, 41, 78, 23, -105, -36, -80, -77, 75, -89, -7, -30, -77, -48, 0, -10, -24, 33, -33, 33, -27, 51, -28, -7, 6, -13, -20, 36}
, {-20, -208, -63, -135, 6, 38, 97, 13, -107, -51, -73, -29, 85, -38, -18, -43, -49, 21, 14, 79, -11, 28, 17, 45, -84, -24, -89, -29, -36, -2, -80, 4}
, {36, 67, 35, 7, 33, -38, -56, 98, 30, 36, 31, -189, 170, 58, 30, -26, -56, 10, -92, 33, -43, -182, 16, 51, 72, -160, -27, 90, -20, 38, -51, -126}
}
, {{-75, 4, 6, 51, -56, 26, 83, -71, 22, -11, 10, 97, -30, 4, -26, 51, 59, 7, 76, 9, 8, 84, 54, -14, 2, 43, 19, -19, 78, 55, 0, 32}
, {-202, -26, 30, -41, -117, 44, 27, -87, -85, 9, -8, 76, -76, -57, 11, 73, 30, -15, 59, -109, 57, 24, 94, -7, -23, 139, 48, -52, 53, -3, 66, 35}
, {23, 27, -22, -20, -35, -17, -155, -45, -82, 3, 10, 112, -50, 12, -19, 48, 30, -34, -14, -101, 19, -39, -129, -61, 85, -10, -61, -12, 8, -49, 3, 44}
}
}
, {{{77, 92, -89, -8, -65, -53, -55, 76, 76, 46, 60, 110, -121, -126, 11, 32, 51, -6, 2, -88, 66, -31, -171, -34, 85, -37, -87, 14, 47, -39, -74, 14}
, {-14, 24, -162, 10, 3, 29, 40, -6, 42, 2, -59, 71, 88, -65, 57, 13, -190, -41, 46, 143, -29, 3, -251, -83, 35, 28, 10, -14, -48, -73, -34, -30}
, {-39, -27, 50, -21, 14, -134, -39, 10, -12, 29, 33, 115, 59, -27, 38, -53, -18, -27, 7, 7, 82, -108, -98, 0, -5, 20, -59, 12, 37, 3, -50, 62}
}
, {{-23, 117, -224, -120, -103, -14, -72, -59, 10, -161, -110, 63, -59, -172, -80, 146, -126, -152, 16, -97, -189, -40, -219, -110, 92, -8, -176, -134, -239, -158, -184, 102}
, {-177, 85, -62, 35, -75, -261, -1, 9, -64, -11, 24, 36, -53, 44, 5, -78, 96, -129, -29, -117, 3, -181, -163, -60, 22, 14, 52, 9, -29, 26, 47, 52}
, {25, 49, -1, 16, -6, 42, -98, -95, 55, 40, 3, 89, -110, -47, 35, 47, 38, -88, 17, -64, 78, 16, -21, -95, 18, 83, -47, 16, 55, 46, 19, 3}
}
, {{91, 118, -42, -103, -24, -155, -74, -233, -30, -21, 35, 57, -16, 40, 45, -29, -4, -228, -24, -101, 65, -117, -91, -125, -75, 35, -51, 15, 9, 39, 18, 53}
, {19, 31, -21, 84, -67, -126, -297, 21, -37, 49, 23, 94, -19, -30, 45, 29, 44, 74, -18, -65, 42, -197, 3, -93, 20, -201, -102, -4, 28, 8, -17, 14}
, {52, 13, -55, 9, -51, -76, 41, -76, 29, -62, -59, -12, -74, -55, -37, 9, -67, -38, -15, -53, -134, -41, 0, -28, -26, -27, -88, -114, -78, -111, 39, -26}
}
}
, {{{36, 69, -45, -9, -9, 21, 42, 12, -12, -43, 24, -39, 86, 16, 16, -23, 40, -36, -15, 43, -38, 57, 36, -29, 12, -72, 22, 1, -19, 10, -29, 16}
, {-10, -126, 39, 63, 88, 77, 7, 29, 19, 85, 28, -22, 3, 32, 45, 3, 40, -43, 39, 41, 40, 0, 100, 38, 50, 1, 62, 42, 15, 31, 41, -3}
, {89, 1, -7, 17, 43, -8, 28, 35, 34, -38, -32, -151, -24, -22, -25, -24, -82, -3, -7, 62, 12, 30, 0, -38, 0, 3, 29, -51, -13, -28, -29, -163}
}
, {{-32, -68, 69, 35, 3, 63, 118, 3, -47, -84, -19, -144, -39, -67, -113, -113, -29, 18, 21, 12, 16, 25, 21, -5, -3, 67, 65, -53, 23, -67, 15, -117}
, {-49, 38, -59, -69, 12, -27, -73, 18, -102, -23, 2, 27, 73, -31, 54, -12, -24, -27, -62, 9, -15, -89, 3, -33, 18, 67, -32, 42, 2, 62, -5, 71}
, {56, 100, 11, 54, 22, -50, -91, 60, 70, 22, 49, 30, -33, -34, -112, 88, -28, -130, -10, -231, -82, -15, -86, -134, 42, -190, -23, -15, -129, -198, -155, 128}
}
, {{21, -85, 44, -35, 21, 23, 41, -46, -72, -14, -57, 45, 2, -17, 34, -37, 77, 46, -26, 2, 13, 74, 83, -45, -83, 34, 1, -17, 49, 21, 88, -38}
, {15, 68, -77, 3, -152, 36, -77, 25, -16, 46, -9, 67, 10, -61, 70, 24, 32, 31, -6, 9, 18, -41, 13, -23, 2, -59, -44, -13, -8, -33, -83, 104}
, {66, -25, -46, -89, 9, -80, -127, -33, -3, -59, -105, 77, -98, -14, -64, -44, -99, -54, -191, -173, -166, 6, -170, -119, -7, -29, -125, -82, -276, -78, 5, 15}
}
}
, {{{-14, -52, 15, -80, -54, 19, -22, 9, -82, -11, -95, 73, 35, -24, 43, 4, -50, 21, -32, 10, 10, -8, 28, 13, -39, -49, -31, -32, -30, 2, -55, 47}
, {-261, -86, -7, 11, -96, -153, 124, 5, -129, 29, 88, 61, -93, 65, 8, -8, 50, 18, 62, -57, 80, -130, -82, -106, 41, -32, -52, 79, 37, -44, 11, 4}
, {41, -7, 14, -17, 42, 60, -88, -28, -17, 63, 30, 39, -111, 26, -160, 53, -16, 13, 27, -55, 57, 33, 7, -11, 94, -179, -77, -3, 34, 5, -105, -35}
}
, {{-56, -287, 24, -23, 32, 9, -62, -4, -118, -15, -7, 54, 104, -4, 37, -1, 18, -16, 11, 0, 38, -23, 14, 11, -29, 14, 41, -18, -24, 1, 27, 60}
, {2, 58, -37, 54, -37, 10, -13, 22, 5, 23, 31, 54, -30, -17, -49, 67, 42, 38, 122, -7, 79, 80, 19, 38, 5, -107, -56, -2, 86, 69, -92, -67}
, {93, -26, -1, 70, 50, -20, -33, -16, -7, -201, -49, -85, 76, -23, -122, -35, -221, 19, -129, -5, -79, 13, 46, 28, 40, -21, 15, 12, -83, -124, -16, -106}
}
, {{25, 29, -31, 42, -19, 38, 36, -32, 16, -11, -37, 21, 83, -30, -24, -43, -59, -13, -31, 102, -83, 2, -14, -4, 21, 13, -28, -53, -17, -48, -34, 18}
, {-134, -45, 36, 20, 62, -83, -30, -40, 32, -27, 49, 42, -15, 26, 31, -93, 38, -24, -42, -47, 18, -115, 38, -21, 37, 51, -12, 60, -36, -8, 60, -12}
, {-49, 6, 10, 48, 52, -63, -2, 68, 9, -107, 56, -3, 76, 77, -57, -8, -26, -19, -14, -102, -29, -126, -134, 50, 51, -63, -38, 34, 6, -32, -57, 50}
}
}
, {{{-22, -57, 4, -22, -11, -15, -75, 100, 19, 39, 52, -78, 26, 18, -21, -25, 32, 11, 12, -29, 51, -45, -61, 68, 58, -58, -43, 59, 22, 78, 22, 4}
, {-58, 58, -33, -2, 30, -124, -68, 53, 21, 7, 67, 27, -27, 1, -25, 80, -33, -42, 24, -118, -11, -157, -33, -14, 66, -70, -26, 5, -55, -30, -49, 31}
, {36, 82, 42, 7, 19, 25, -242, 29, 14, 115, 16, 139, -7, 11, 121, 39, 126, -101, 36, -172, -39, -49, 44, -47, 4, -9, -12, 80, 69, 0, 3, 143}
}
, {{57, -34, 22, -37, -29, 63, -28, 13, -33, -8, 23, 3, -27, -12, 13, 115, -19, -63, -51, -14, -18, 20, -39, -29, 10, -27, -30, 13, 40, 19, 29, -1}
, {42, 43, -29, -67, -101, 101, 16, -50, -49, -18, -15, -113, -21, -83, -22, 44, -17, 35, 17, 29, -12, 15, -13, -28, -14, -125, -80, -52, -8, -66, -80, 18}
, {65, -17, -81, -100, -64, 74, 46, -23, -67, -35, -106, -48, -28, -165, -122, 42, -95, 20, 96, -16, 28, 80, 37, -5, -19, -21, -160, -124, -38, -47, -131, -69}
}
, {{39, 18, 32, 92, -1, -51, -59, 43, 49, 10, 54, 58, -17, 31, -12, 23, 1, 44, 32, -45, -15, -60, 59, 5, 45, 54, 42, 27, -17, 0, 31, 15}
, {4, -21, 29, 58, 5, -108, 107, -64, -39, -7, -13, 106, -193, 72, -8, -63, 27, 32, -109, -94, 5, -104, 124, -16, -87, -1, 5, -1, -1, 52, 39, -74}
, {-74, -16, 14, 75, 49, -19, 114, 2, 74, -38, 8, -150, -159, 82, -53, -192, 8, 5, -179, -97, -112, -128, 132, 63, -93, 45, 19, 61, -35, -39, 8, -154}
}
}
, {{{-92, 38, 2, 25, -26, 2, 3, -42, -61, 52, -58, 126, 29, 16, 32, -22, 85, -34, 64, -50, 9, -4, 19, -48, 41, 53, 2, 29, -16, 10, 36, 98}
, {-41, 23, 13, 55, 33, 78, -33, -58, 27, 27, -26, -32, -90, -26, 35, 58, 49, 51, 83, 13, 26, 38, -40, -21, 18, -38, -17, -10, 42, 25, 74, -68}
, {64, -58, -51, 18, 18, 30, -36, -66, -8, -104, -136, 22, -133, -111, -160, 39, -155, 70, 19, -4, -39, 90, -36, -76, -82, 124, -10, -219, -85, -101, 99, -219}
}
, {{-50, -64, 57, 14, -21, 7, 62, 8, -38, -34, -30, -121, 14, 40, -69, -104, -34, 44, 3, -16, -4, -19, -28, 46, -30, 150, 59, -3, -18, -19, 33, -35}
, {4, -39, -20, -13, 45, -38, 4, -7, 52, -1, -66, -14, -48, -16, 10, -121, 57, 32, -92, 52, -74, -23, -6, -31, -95, 76, 35, -17, -17, -29, 19, -71}
, {-151, 92, 20, 23, 73, -106, -68, 72, 39, 31, 103, -102, 169, 79, 49, -105, -34, 26, -95, 53, -76, -110, 13, 61, 51, -128, 4, 89, -65, 8, -42, 92}
}
, {{-104, -52, 24, -6, -36, -58, -39, -6, 13, 21, 66, -21, -54, 26, -2, -201, 49, 24, -57, -78, -19, -84, -33, 2, -25, 12, -24, 39, 9, 26, 10, -30}
, {-235, -6, 23, -61, 0, -36, -102, -27, -61, -29, 41, -128, -94, 22, -12, 15, 16, 13, -28, -117, 13, -15, -65, -54, -20, -33, -31, 41, 29, 12, 12, -59}
, {-31, 29, -11, -83, -77, 6, -110, -13, -50, 29, -8, 56, -31, -15, 32, 43, 127, -21, 29, -106, 51, -18, -43, -91, -63, -40, 13, 28, 53, 101, 114, -9}
}
}
, {{{-21, -38, -51, -8, -127, -67, 32, -46, -160, 9, 6, 41, 9, 16, 7, -1, -53, -20, 57, 69, -3, -7, -120, -50, -36, -7, -27, -47, -40, 3, -57, -40}
, {31, 114, 4, -76, 32, -3, -17, 58, 51, 15, 57, 76, -85, -3, 41, 31, 17, 4, 71, -5, 45, -12, 10, -16, 4, 37, 33, 44, -11, 23, -6, -19}
, {110, -26, -31, -29, -33, -42, 35, 90, 27, 114, 76, 38, 66, -21, 149, -30, 70, -16, 45, 67, 53, -16, 19, 44, 41, -24, -32, 34, 59, 15, -104, 95}
}
, {{-8, -13, -13, -120, -119, 17, -74, 32, -76, 57, -1, 24, 30, -39, 12, -42, 34, -19, -14, -61, -5, 28, 17, -42, 18, -63, -149, 19, -10, 14, 25, 40}
, {132, 149, -56, 14, 13, 80, 8, 41, 71, 34, 61, -11, 61, 3, 42, 97, 19, 12, -84, 3, -28, 88, -35, -17, 47, -43, -102, 44, 1, 52, -26, 92}
, {-20, 31, -121, -40, -65, -67, -90, -55, -17, -7, -21, 50, 71, -78, 57, 13, -166, -151, -33, -81, -129, -56, -208, -92, 17, -139, -281, -108, -101, -35, -339, 67}
}
, {{83, 111, -5, -67, -136, -1, 76, -51, -88, 21, 21, 100, 73, -27, 65, 88, 74, -32, 79, 37, 0, 48, -6, -60, -34, 66, -24, 50, 6, 13, 44, 21}
, {82, -22, -123, -82, -65, 33, -27, -42, 11, -6, 14, 111, -182, -17, -96, 17, -36, 17, 56, -25, -62, 57, -202, -82, 10, -34, -73, -28, -116, -135, -184, -50}
, {42, 51, -171, -164, -86, -45, -49, -28, -91, -60, 0, 34, -5, -36, 10, 40, 7, -206, -147, -65, -44, 21, -102, -68, -204, -77, -215, 10, -34, 14, -89, 74}
}
}
, {{{8, 55, -88, -14, -27, -16, -24, 77, -32, 40, -34, 107, 4, -6, 85, 42, 73, 13, 99, -43, -20, -16, -74, -101, -24, -7, -42, 20, -1, 12, 59, 19}
, {-42, 107, -16, 7, 26, -62, 11, 40, 70, 37, 44, 139, 21, 1, 31, -6, -19, -36, 101, 62, -9, -71, -172, -111, 4, 41, 56, 11, -55, 52, 33, 22}
, {29, 41, -61, -62, -146, -60, -56, 8, -28, 90, 17, -54, -177, -112, -137, 83, 3, -59, 115, -63, 57, -79, -150, -131, 68, -123, -96, -46, -39, -135, -134, -79}
}
, {{19, 27, -19, -42, 32, -49, -161, -10, -1, 27, -32, 5, -18, -7, -38, 13, -22, -32, 13, -63, -85, -39, -83, -24, 33, 53, 21, 40, 37, 18, -5, 5}
, {118, 49, 53, 66, 12, 1, -316, 29, 98, 100, 2, -106, -7, 80, 40, 44, 6, 20, -31, -32, -58, -85, 0, 2, 52, -59, -58, 10, -39, 31, 3, 76}
, {-132, -10, 15, 17, 47, 4, -45, -42, 42, -197, 0, -154, -129, -29, -169, -100, -44, -8, -85, -233, -61, -25, 20, -41, -56, -19, -43, -81, -21, -37, 49, -147}
}
, {{42, 52, -104, -64, -44, -86, -35, 38, -56, 20, 18, 23, 31, -118, 80, 85, -21, -4, -3, 20, 5, -85, -55, -51, -14, -213, -171, 2, 26, 55, -150, -13}
, {67, -65, -40, -32, -18, 32, -61, 75, 14, -5, 22, -155, 35, -43, -72, -15, -81, 53, -5, 30, 41, 33, 58, 63, 18, -98, -28, 21, -21, -8, -170, -52}
, {58, -54, -7, -159, 55, 13, 64, 1, -80, -123, -44, -87, -20, 35, -9, 37, -81, 37, -7, 43, -49, 49, 102, 38, -148, 14, -35, -68, 37, 26, -94, -92}
}
}
, {{{17, 69, 18, -106, -32, 38, 13, -7, -31, -39, 17, 0, 27, -4, -14, -23, 80, -2, -20, 42, 61, 42, 64, -13, 87, -27, 3, -13, -2, -40, -10, 1}
, {-16, 171, -43, 68, 34, 30, 1, 74, 53, 88, 44, 45, 64, -52, 65, 105, 57, 9, 41, -14, -5, -66, -57, -17, 97, -46, -7, 27, 40, 49, -8, 87}
, {66, 42, -199, -108, -148, -23, -4, -77, 30, -41, 28, -20, -111, -196, -167, 34, 17, 14, 71, -31, 40, -23, -97, -71, -8, -94, -126, -138, -9, -104, -330, -121}
}
, {{6, 125, -16, -10, -8, 8, -3, -68, -9, -51, 21, 39, 33, -11, 49, 155, 38, -57, 50, 73, 51, 78, 19, -102, 68, 57, 34, 7, 1, 0, 37, 16}
, {13, 72, -36, -45, -63, 66, 77, -68, -30, -25, -17, 118, -37, -72, -197, 13, 92, -70, 81, -56, 15, 22, -68, -163, 17, -10, -31, -26, 12, -194, -118, -68}
, {4, -67, -43, 37, -38, -47, -29, -124, -87, 76, -74, 50, -234, -90, 13, -5, 128, -329, -89, -237, -62, -1, -309, -243, -165, 42, 3, -1, -84, 4, 83, -22}
}
, {{5, 37, -5, 30, -101, -13, -105, 29, -2, 102, 44, 153, -100, 20, 8, 37, 44, -8, 32, -104, 7, 11, -18, -49, 32, -53, -10, 16, -5, 39, 37, 112}
, {16, -116, 62, -45, -47, 28, -71, -191, -76, -5, -36, 7, -290, -23, -52, -104, 93, -59, -91, -146, 60, 46, 75, -59, -189, 21, -18, -8, 26, 25, 64, -58}
, {-265, -214, 6, 31, 26, -32, 66, -44, -72, 27, 15, 22, 37, 54, 62, -49, 54, 70, 59, 98, 87, 32, 104, 58, -20, 148, 110, 73, 28, 65, 60, -49}
}
}
, {{{13, 59, -7, -105, -39, 71, -21, 45, -4, -121, -46, -147, -14, -45, -110, 37, -86, 67, 71, 38, 14, 19, 10, 53, 29, -107, -85, -93, 14, 20, -161, -41}
, {23, -142, -89, -132, -23, 20, -26, -11, 22, -56, -105, 58, 10, -8, 26, 25, -59, 41, -32, 2, -30, 22, -43, -1, -38, -18, 20, -15, -63, 36, -75, 68}
, {-111, -35, -17, -33, 36, -37, -4, 61, -60, 29, 16, 42, 20, 42, 50, -76, 99, 51, 1, 33, 38, -33, 10, 26, 2, 16, 19, 28, 14, 21, 9, 7}
}
, {{-31, -51, 62, 28, 101, -47, 17, -10, 123, -50, -96, -49, -149, -95, -88, -138, -156, -24, -171, -91, -219, -63, -120, 8, -31, 97, 100, -87, -197, -142, -13, -114}
, {-67, -150, 17, 23, 92, 28, 66, -192, 68, -204, -218, -130, 37, -164, -108, -265, -179, -121, -173, 61, -94, -2, -26, -52, -154, 113, 21, -219, -158, -171, -47, -55}
, {17, -180, -71, -77, -18, 63, 32, 17, -7, -51, -116, -42, 72, -119, -38, -51, -96, -34, -13, 35, -3, 12, 44, 69, -87, 16, -113, -38, 21, -16, -70, 87}
}
, {{-13, 35, 46, 160, 104, 12, -4, 14, 146, 7, 48, 26, 74, 17, 42, -1, -6, 39, -12, 43, 15, 15, 35, 30, 33, 73, 45, 23, -37, 41, 30, 71}
, {-67, 107, 79, 78, 51, 34, -11, 98, 173, 37, -26, -72, 81, 54, 35, -117, -22, 41, -49, 32, 38, 27, 61, 63, 48, 96, 91, 33, 23, 56, 29, 2}
, {22, -87, 36, 77, 27, -5, 105, -44, -5, -3, 13, -81, -114, 39, -94, 3, 8, 26, -42, -55, -33, 56, 26, -13, -118, 35, 40, -27, -19, 4, 12, -56}
}
}
, {{{-45, 34, 72, -6, 26, -147, -63, 36, 17, 19, 37, -6, 52, 43, 87, -64, 32, -18, -89, 32, 57, -118, 74, 30, 76, 54, 21, 76, 95, 12, 46, -33}
, {-80, 66, 3, 99, 59, -11, -144, -25, 98, -2, 18, 9, 15, 60, -44, -87, 17, 71, -24, -60, -21, -116, 28, 41, 16, -27, -25, 34, 7, 8, -39, 46}
, {3, -46, 1, 19, 25, 54, -17, -36, -25, 28, 19, -51, -77, 26, 4, -13, 57, -36, 36, -83, -13, -36, 42, -61, 24, 13, 13, -1, 36, 37, -3, 4}
}
, {{71, 8, 33, -14, -36, 44, -91, 18, 6, -65, 0, -138, -32, -7, -27, 38, 25, 77, 28, 10, 23, 4, -9, 48, 19, -46, 9, 28, 7, 24, -5, -76}
, {42, -113, 78, -53, 21, 47, 116, -38, 27, -14, -22, -152, 44, -16, -70, -75, -8, 47, 21, 14, 72, 78, 56, 76, -103, 47, -50, -17, 79, -5, -56, -58}
, {10, -70, -16, -106, 35, 75, 71, -58, -86, -130, -57, -100, -5, -41, -4, 4, 16, 16, 60, 33, 43, 19, 89, 46, -162, -9, -14, -50, 70, 46, -103, -101}
}
, {{-33, 0, 61, 33, -11, -32, -33, 25, 74, -62, -57, -177, -52, -41, -143, -113, -104, 8, -56, -77, -65, -68, 81, -55, 26, -45, -44, -49, -18, -93, -3, -167}
, {16, 31, -12, 41, -36, -134, -10, -4, 111, -77, -4, 34, -31, -34, -19, -94, -32, -72, -122, -35, -65, -96, 149, -9, 21, 21, 25, -56, -32, -72, -107, 7}
, {11, 32, 26, 25, 0, 15, -96, -6, 12, 8, -28, -6, 0, -16, 38, -53, 14, -14, -64, -60, 23, 4, 123, 62, -52, 58, 59, 23, 31, 31, 7, 97}
}
}
, {{{56, 122, -61, -12, -58, 24, -57, 61, 183, -9, 74, -62, -49, -94, -36, 36, -99, 6, -5, -1, -5, -37, -60, -40, 47, 20, -123, 24, 37, -50, -120, -76}
, {29, -138, -158, -31, -100, -4, 27, -39, -87, 1, -77, 46, 21, -68, -13, -9, -67, -30, 55, 81, -9, -2, -141, 4, -86, -9, -31, -36, 25, 25, 43, 48}
, {-117, 34, 18, -25, 12, -258, 13, 24, -10, 16, 60, 103, -26, 40, 96, -6, 50, -91, 50, -33, 42, -105, 13, -57, 22, -19, 28, 41, 51, -9, 17, -24}
}
, {{-24, 180, 7, 20, -34, -29, -5, -33, 155, -158, -64, -44, -147, -78, -109, 50, -99, -143, -15, -92, -134, 16, -115, -135, 107, 16, 15, -97, -187, -146, -5, -7}
, {-315, -16, -76, 10, -28, -248, -350, 36, -120, 61, 40, 34, 16, 41, 79, -23, 4, -56, 23, -86, -16, -188, -73, -6, 39, -1, -17, 26, 38, 28, 71, 32}
, {44, 62, -24, 54, -3, 17, -184, 6, 36, 63, -30, 82, -240, -15, -62, 27, 1, 0, 35, -68, 46, 37, -78, -96, 34, -167, -22, 2, -52, -69, 1, -58}
}
, {{-13, 99, 99, 58, 69, -43, -72, -48, 91, -34, 4, 10, -65, 67, 13, 39, 31, 7, 25, -105, -44, -36, 24, 10, 6, -128, -19, 26, 11, 18, -13, -24}
, {3, 23, 1, 92, 40, 55, -161, 6, -42, 8, -22, -18, 68, 34, 37, -25, -65, -27, -64, -21, -41, -46, -87, 32, -2, -140, -178, -19, -13, -37, 4, 38}
, {-8, -6, 49, -23, 12, 17, -168, 3, 27, -55, -51, -186, 3, -7, -91, -89, -72, 76, -190, -85, -33, -30, 53, 71, -119, 74, -55, -13, -13, 12, 79, -42}
}
}
, {{{-7, -108, 79, 44, 139, 20, -27, 62, 25, -66, -93, -22, -11, 1, -61, -7, 12, -18, -16, 21, 10, -36, 123, 11, -83, 39, 57, -2, 33, -11, 43, -146}
, {-114, 14, 30, 63, 43, -199, -44, -53, 79, -5, -6, -2, -21, 26, -53, -51, -55, -121, -128, -40, -126, -247, -66, -16, -77, 46, 25, -49, -111, -68, 28, 19}
, {-23, 74, 78, 51, 98, -30, -95, 8, 77, 29, 18, 47, -21, 53, -17, -101, 3, -82, -137, -231, -18, -213, 37, -37, 6, 36, 28, 49, -47, 6, -15, 98}
}
, {{45, -37, -30, -78, -101, -10, -23, -18, -138, 46, 80, 44, -2, -44, 17, -83, 55, 17, 10, 8, 28, -31, -14, -12, -38, -40, 25, 5, 41, 12, -14, 6}
, {-81, 52, -14, -65, -28, -78, -164, 67, 2, -1, 77, -57, 3, 20, 62, 26, 48, 20, -31, -29, 53, -72, -27, 3, -58, -127, -75, 60, 41, 9, -108, 28}
, {-89, 50, 10, 23, -23, 10, 24, 21, 7, 41, -34, -15, -84, 49, 6, -68, -67, 22, -70, -65, -38, -2, 31, 5, 63, 5, -49, -14, 12, 95, 5, -67}
}
, {{85, 3, 29, -98, 8, 26, 31, -18, -26, -8, -60, -88, -56, 8, -127, 144, 38, 15, 66, -42, -5, 94, -5, -57, -59, -48, 11, -21, -9, -15, 29, -64}
, {61, -48, -78, -112, -58, 33, 35, -13, -106, 11, -113, -6, 33, -53, -19, 64, 3, -2, 67, 45, 72, 13, -11, 23, -29, -197, -175, -30, -9, 89, -120, 2}
, {9, -81, -28, -10, 35, 79, 113, 29, -8, -2, -23, -135, 32, 6, 64, -111, 0, 84, 2, 88, 42, 112, 52, 91, -59, -36, 17, 23, 32, 82, -19, -61}
}
}
, {{{42, 146, -44, 10, 44, 67, 60, 67, 99, 76, 59, 17, -39, 61, 2, 46, 17, 52, -6, -13, 6, -21, 74, 65, 41, 34, -34, 102, 23, 55, -2, -24}
, {26, -1, -269, -97, -177, -9, -15, -12, -8, 25, -22, 35, 88, -96, 60, -22, -69, -19, -29, 74, -40, -43, -47, -37, -34, 5, -42, -111, -13, 3, -136, 80}
, {11, 59, -147, -33, -122, -47, -95, 43, -19, 15, 58, 0, 65, -66, 41, 99, 27, 6, 119, 39, 66, 3, -99, 7, 46, -148, -191, 25, 56, -20, -241, -1}
}
, {{73, 17, -28, -13, 28, -53, -9, -45, -28, 58, -9, 150, -112, -12, -53, 138, 33, -49, 52, -84, -121, -17, -86, -36, 15, 9, 25, -47, -27, -37, 51, -8}
, {-91, -193, 26, 54, -16, -46, 22, -133, -61, 3, -57, 44, -38, 29, -31, -57, -6, -70, -31, -105, -79, -81, -85, -124, -90, 45, 8, -21, -56, -47, 35, -35}
, {-42, -34, 61, 56, -34, 2, -28, -95, 77, 11, -21, -5, -118, 43, 47, -42, 35, -47, -44, -145, 25, -86, 90, -26, -42, 86, -9, 24, 49, 69, 29, -22}
}
, {{12, 52, -14, 47, -14, -2, -52, -36, -27, 33, 56, 63, 31, 11, 77, 45, 9, -52, 44, -11, 28, -91, -44, 21, 20, -66, 30, -10, 11, 23, 37, 25}
, {-43, -108, -42, -51, -118, -12, -116, 10, -166, 31, 42, -37, -53, -65, -2, 9, -69, 40, 19, -13, 38, -42, 33, -15, 47, -174, -48, -52, 15, -65, -148, 66}
, {31, -14, 20, -45, 7, -3, 57, 48, -137, -5, 5, -28, -103, 52, -95, 45, -31, 5, 19, -35, 24, 86, 61, -31, -10, 39, -27, -9, 20, -62, 5, -154}
}
}
, {{{-20, 75, -50, -160, -126, -55, 58, -88, -209, 38, -34, 58, 40, -81, 40, 26, 108, -42, 55, 42, 25, -33, -58, -148, -51, -9, 2, 26, -10, 32, 36, 17}
, {23, 39, -14, -10, -16, 20, 27, 18, -66, 81, 54, 37, -26, 5, 41, -47, 60, -25, 36, -20, 69, 10, -25, 38, 56, -44, -38, 34, 5, -5, -28, -57}
, {23, 24, 10, 11, 85, -39, -53, -18, 24, -4, 15, -17, -17, -12, 34, 55, -17, -8, 83, 74, 96, 15, -33, 2, 23, -8, 32, -56, 71, 20, -46, -11}
}
, {{-1, -156, -59, -61, -105, 19, -39, 105, -116, 79, -16, -73, 31, -17, 54, -27, 20, 2, -20, -32, 30, 65, -15, 9, -34, -89, -49, 12, 20, -2, 23, 36}
, {57, -49, -120, -41, -1, 64, 17, 46, 21, -75, -55, 12, -4, -30, -5, -10, -146, 46, 52, 50, 17, 58, -17, -6, -29, -65, -93, -29, 48, -39, -203, -77}
, {-107, 85, -5, 15, -69, -175, -45, 18, 65, -54, -34, -30, 103, -63, 68, -70, -127, -80, -251, 48, -124, -168, -118, 22, -3, -6, -13, -37, -142, -13, -38, -30}
}
, {{-55, -148, 25, 2, 12, 30, 144, -66, -16, -64, -97, 86, -24, 28, -29, -62, 49, -35, 11, 62, -121, 133, 98, -94, -135, 81, 102, 18, -45, -64, 104, -98}
, {-36, -12, -31, -62, -130, -125, 118, -156, -76, -15, 54, 39, -23, -38, 20, -22, 62, -98, 44, -34, -17, -65, -41, -188, -11, 67, 47, -45, -27, -44, 8, 40}
, {-21, 52, -63, 9, -91, -126, -213, 74, 21, 28, 68, 99, 36, -55, 4, 43, 17, -89, 33, -51, 5, -96, -124, -70, 47, -249, -252, 1, 1, -17, -126, 65}
}
}
, {{{32, 10, 24, -11, -10, -71, 45, -68, -25, -26, -72, -20, 59, -6, 1, 25, 29, -20, 34, -57, -35, 9, 2, -16, -8, 61, 32, -74, -34, -44, -7, 26}
, {-142, -99, 13, 6, 24, -21, 22, 49, 39, 34, 77, 12, -21, 48, 24, -75, 71, 34, -92, -50, 4, 8, -6, 53, 79, -8, 12, 76, -17, 6, 35, 1}
, {-133, 39, 21, 27, -29, -108, -174, 19, -4, 78, 50, -6, 23, 36, 20, 40, 128, -29, 51, -118, 86, -205, -12, 27, 48, -4, 22, 48, 61, 78, 43, 53}
}
, {{-96, -4, -24, -195, -108, 4, 84, -172, -33, -110, -104, -22, -49, -87, -15, -32, 90, -12, 35, 24, 3, 77, 32, -30, -180, 55, 89, -66, 14, -7, 4, -62}
, {-111, 104, -139, -66, -80, -63, -39, 29, -57, 6, 1, 38, -19, -9, 34, 42, 3, -29, 16, 11, 15, -55, -99, -56, 43, 36, -71, 14, 20, 70, 14, -29}
, {-23, 43, -20, 25, -28, -31, -265, 51, -14, 31, 73, 33, -45, 41, 41, 38, -4, -51, 14, -99, -1, -99, -100, -96, 78, -302, -184, 63, -22, -72, -43, -29}
}
, {{7, -55, 50, -15, -57, 20, 70, -129, -50, 7, -75, 51, 1, -16, 43, 25, -14, 76, 49, 61, 18, 68, 99, 19, -91, 66, 62, 21, 41, 43, 50, 0}
, {55, 38, -29, 36, -63, -38, -169, 2, -49, 32, 11, -3, -6, -26, 44, -94, 34, -4, -49, -8, -4, -90, 12, 21, 3, -105, -23, -16, 39, 17, -31, 53}
, {-40, -24, -12, 4, -5, -89, -209, 38, 1, -8, -25, -56, -8, -71, -107, 0, -46, 25, 5, -150, -65, -96, -35, 93, 66, -18, -55, -45, -11, -28, -19, -65}
}
}
, {{{46, 38, 21, 15, -32, -105, -126, -52, 61, -40, 37, 68, 3, -34, 96, -59, 76, -81, 2, -19, 68, -91, 25, 34, -24, -61, -28, 22, 19, 0, 19, 38}
, {-181, 51, -19, 3, -58, -130, -207, 1, -52, 20, 55, 134, -35, 14, 60, 103, 3, -1, 72, -84, 74, -180, -2, -73, 28, -26, -8, 1, 81, 51, 26, 17}
, {87, 35, 37, -2, -32, 51, -29, -25, -8, -7, 16, 44, -172, 11, -73, 61, 32, -10, 78, 37, 78, 18, 31, -106, 9, -41, 65, -54, 66, -74, 2, -53}
}
, {{-26, -10, -12, 15, -11, 29, -37, 53, -54, -32, 16, 12, 25, -12, 24, 38, -36, -4, -18, 26, 30, 10, 7, 69, 31, -137, -69, -42, -2, 10, -51, 53}
, {18, -120, -1, 73, 52, -83, -237, 59, -27, -3, 34, -58, 64, -11, -6, -62, -77, 18, -45, 11, -7, -151, -83, 60, -20, -243, -130, 17, 57, 34, -80, 64}
, {-41, 16, 39, 20, 22, 6, -38, 82, -20, -83, -16, -155, -33, -23, -1, -111, -39, 54, -59, -19, 52, -29, 66, 26, -10, -63, -22, 46, -12, 15, 54, -72}
}
, {{46, 29, -126, 106, 30, -27, -40, 39, 21, 25, -48, 39, 71, -61, 34, -13, -109, -64, -13, 30, -33, -12, -48, 20, 75, -139, -105, -78, -49, -33, -78, 20}
, {47, -122, -75, -36, 49, -61, -4, 57, -104, -26, -114, -37, 64, -26, -26, -199, -144, -57, -184, 56, -60, 3, -90, 67, -96, -57, -142, -72, -78, -50, -58, -46}
, {-59, 8, 26, 88, 45, -42, -112, 29, 66, 12, 22, -73, 145, 52, -1, -39, -73, -4, -52, 50, 14, -97, -68, 50, 14, -41, -45, 72, 0, -26, -68, -6}
}
}
, {{{80, 50, -7, 43, -49, 50, 34, 53, -11, 34, -33, -81, 17, 57, 35, 27, -33, 20, -2, 16, -13, 71, -23, 11, 15, -52, 2, 40, -15, 3, 1, -46}
, {93, 64, 35, 26, 24, 11, 45, 40, -27, 14, 66, -17, 49, 40, -6, 50, 17, 58, 40, 61, 92, 82, 11, 42, 1, -2, 31, 60, 12, 5, 18, -3}
, {-45, -8, 18, 22, 6, 38, -17, 75, 16, -28, 1, -56, -71, 24, -79, -46, -3, -8, -98, -29, 24, -27, 70, 15, -69, 73, 14, -21, -19, -25, 10, -133}
}
, {{15, -38, -8, 71, 58, 51, -10, 77, 40, -10, 2, 35, 2, -8, 50, 68, 1, -5, 36, 4, -59, 4, -39, 11, -29, 23, 39, -16, 26, -35, -12, -54}
, {67, 46, -69, -122, -96, -23, -106, 58, -56, 20, 9, 5, 47, -32, 10, 42, -18, -29, 36, 5, 24, -48, -46, 15, 16, -22, 27, 52, 70, 15, -27, -22}
, {-90, -72, -11, 64, 20, -72, -121, 67, 11, 15, 27, -46, 11, 2, 38, -145, -36, 57, -99, -16, -7, -138, -15, 38, 57, -94, 7, 18, 11, 26, -57, 6}
}
, {{49, 65, -5, 33, 73, 78, 25, -50, 32, 51, 88, 62, 6, -6, 46, 58, 47, 38, 38, 42, 31, 61, 75, 2, 88, 107, 124, 84, 60, -8, 53, 4}
, {163, 74, -126, -122, -239, -26, -56, -166, -5, -150, -181, -7, -54, -240, -48, 52, -120, -68, -38, -1, -44, 23, -89, -137, -18, -142, -155, -272, -77, -48, -192, 30}
, {-205, -53, -128, -7, 33, -37, -109, 76, 23, 16, -76, -11, 100, -31, 66, -228, -98, 32, -220, 75, 21, -138, -125, 132, 25, -140, -236, 32, -31, 87, -197, 89}
}
}
, {{{-12, -8, -24, -36, -83, -77, -16, -65, -48, 1, -25, 36, 8, 18, -9, -31, -31, -24, -27, 1, -38, -45, -36, -67, 3, 34, 83, 1, -6, -79, 24, -21}
, {-172, -42, 68, 10, 37, -9, 30, -60, -15, 6, 11, 13, -102, 26, 17, 3, 83, -1, 136, -1, 56, 37, 49, 41, 21, 58, 40, 19, 105, 35, 35, -109}
, {-12, -53, 38, -2, 116, -29, -4, -11, 67, -144, -31, -222, -80, -2, -78, -134, -18, 54, -78, 131, -12, 76, 71, 95, -106, 86, 102, -53, -31, 28, 7, -191}
}
, {{-106, 38, -6, -44, -119, -77, -32, 68, -33, 81, 35, 31, 58, 2, 39, -125, 37, -18, -63, -53, -32, -89, -14, -17, -55, -43, 3, 56, -30, 18, -21, 13}
, {-91, -45, -17, -42, -38, -26, -76, -69, -152, -4, -10, -50, -118, -10, 36, -68, 23, -3, 55, -21, 3, -2, 4, -41, -80, -97, -32, -14, 8, 10, -13, -69}
, {-132, 76, -19, -68, -14, -70, -57, 50, -45, -43, 19, 63, -42, 20, 32, -5, 44, -40, 26, -42, 24, -143, -28, -5, 71, -15, -32, 16, 18, -20, 2, 102}
}
, {{-49, -174, -10, -137, 29, 84, 29, -3, -106, 13, -140, -99, 97, 7, -23, -83, 8, 40, -48, 30, -5, 35, 9, 74, -46, -143, -26, 33, 61, 39, -55, -4}
, {-78, -152, 61, -25, 0, 71, 78, -40, -69, -138, -41, -182, -6, 54, -115, 31, 7, 69, 78, 15, 35, 105, -5, 4, -75, -18, -1, -45, 12, 33, 9, -36}
, {70, -35, 40, -111, -83, 104, 15, -148, -96, -84, -84, -28, -155, -78, -210, 91, 128, -5, 83, 11, 20, 130, 0, -72, -73, -105, -39, -133, -4, -49, 54, -239}
}
}
, {{{73, 44, -46, 25, 23, 77, -48, 95, 108, -156, -50, -62, -37, -94, -140, 20, -126, -6, -45, -9, 44, 51, 83, -28, 42, -3, -31, -112, -139, -114, -90, -186}
, {0, -45, -52, -90, -80, -7, -28, 40, -23, -72, -66, 31, 15, -61, 8, -251, 5, -26, -78, 6, -25, 22, -22, -8, -132, 5, -29, -13, 0, -51, -48, -24}
, {-92, -40, 16, -39, 87, 7, 10, -39, 36, -6, -40, 63, 7, 7, 40, -121, 47, 15, -172, 15, -88, -55, 4, 22, -109, 86, 67, 6, -21, 83, 63, 70}
}
, {{-23, 59, 12, 18, 75, -17, 10, 67, 94, 34, 8, 8, -137, 21, -20, 46, 21, 11, -1, -54, 5, -13, -55, 28, 45, 24, 66, 59, -4, -73, 54, -78}
, {67, 109, -98, 18, -45, -53, -45, 41, 116, -51, -104, -16, 125, -112, 49, -50, -182, -127, -77, -48, -95, -7, -67, -72, 4, -108, -88, -118, -107, -37, -100, 151}
, {-51, 107, -1, -12, 13, 13, 10, 52, 65, 46, 4, -41, 114, 58, 56, -56, -65, 37, -9, 68, 16, 33, -12, 66, 29, -112, -136, 37, 56, 81, -89, 53}
}
, {{3, 98, 6, 64, 43, -13, 56, 25, 44, 42, 21, -33, -14, 59, -10, 44, 140, 9, 6, 4, -33, -30, 20, -5, 9, 109, 45, 36, 7, 20, 9, -84}
, {-94, 61, 23, 27, 22, -107, 18, -24, 14, -22, 8, 6, -23, -5, -74, -75, -65, -85, -191, -63, -140, -138, -39, -75, -35, 63, 57, -55, -95, -103, 17, -20}
, {35, 47, -4, 17, -12, -54, -119, -72, -29, 7, 75, 49, 23, -28, 63, 51, 35, -106, 61, 1, 26, -15, -198, -109, 26, -8, -18, 9, 41, 12, 32, 109}
}
}
, {{{-20, -42, 60, 14, 61, 75, 75, -37, 33, 30, -14, -74, 69, 30, 2, 6, -9, 58, 42, 76, 42, 93, 37, -19, 29, 30, 26, 27, -58, 36, -12, 17}
, {-31, -75, 19, -50, 45, 44, 14, 10, -116, 45, 72, 8, 0, 49, 24, -54, 82, 52, -2, 29, 63, -1, 79, 34, -50, 1, 87, 7, 28, 86, -8, -36}
, {51, -38, 2, -7, 36, 51, -34, 27, 15, 16, 14, -75, 41, 65, -48, 41, -14, 77, -21, 76, 0, 78, 71, 67, -6, 8, 79, 23, -32, -62, 23, -128}
}
, {{-32, -100, -49, -233, -134, 70, 33, -25, -196, -83, -150, -15, 94, -61, 24, -29, -8, 72, -57, 40, 8, 68, 107, 46, -93, 1, -90, -16, 12, -9, -95, -44}
, {-34, 76, -107, -62, -22, 10, -30, 69, -95, 22, 10, 11, 57, -41, 60, 28, 45, -34, 4, 13, 23, 3, -93, -6, -10, -46, -128, 15, 14, 41, 12, -6}
, {0, 75, -89, 4, -52, 61, -37, -3, 16, -3, -42, 1, 30, -100, -16, 30, -148, 15, 22, -8, 15, 9, -16, -5, 67, -135, -54, -120, -25, -28, -277, 61}
}
, {{12, -41, -141, -26, -167, -152, 4, -103, -157, 45, 12, 26, 69, -26, 63, 61, 53, -234, 45, -10, -32, -53, -107, -249, 28, 23, -25, 62, -16, -14, 55, 36}
, {49, -34, -208, -120, -207, 182, -13, -89, -120, 33, -109, 114, 5, -110, -164, -3, -21, -39, 69, 22, 7, 162, -133, -120, -30, -97, -89, -100, 16, -96, -136, -74}
, {53, -47, -268, -225, -123, 13, -71, -95, -158, -147, -163, 70, -78, -160, -50, 6, -52, -197, -180, -103, -66, 101, -187, -171, -201, -148, -211, -126, -161, -66, -158, -35}
}
}
, {{{30, 146, 42, -38, 1, 22, -23, -3, -12, -26, 53, 101, -13, 80, 36, -70, -10, 30, 12, -46, 31, -23, 42, 25, 52, 13, 27, 29, 40, 49, 13, 71}
, {60, 53, -59, 21, 33, -67, -74, 46, 26, 45, 21, -20, -14, -52, 84, 50, 22, -10, 42, 83, 17, -70, -47, -59, 116, -63, 30, -25, 24, -13, -19, -19}
, {0, -22, -157, -48, -72, -12, 40, 30, -120, 44, -24, 32, 83, 13, 28, -41, -70, 13, 5, 76, 56, -13, 63, -5, -2, 12, -42, 55, -57, 90, -169, 66}
}
, {{6, 145, -28, -16, -24, -13, -31, 27, -26, 18, 35, 43, 22, 40, 7, 60, 50, -55, 72, -39, -22, 20, -100, -83, 46, -37, -52, 20, 33, 1, 44, 4}
, {-37, 44, -33, 29, -39, 15, 62, -78, 4, -19, 17, 118, -158, -11, -87, 41, 13, -144, 90, -122, -36, 24, -32, -132, 38, 22, 2, -36, -67, -113, -6, 66}
, {-70, -72, -57, -29, -93, -71, -72, -69, -92, 60, -36, 101, -61, 18, 55, 52, 18, -170, 100, -149, 93, -105, -145, -150, 17, 121, 22, -52, 15, 18, 19, -10}
}
, {{-38, 50, -34, -2, -99, 28, -129, 0, -3, 28, 54, 71, -96, 8, -43, 24, 43, 70, 20, -62, 37, -65, -49, -13, -3, -83, 29, 8, 53, 50, -7, -7}
, {-102, -23, -22, -28, -87, -89, -19, -101, -97, -15, 39, 64, -93, -8, -34, -4, 31, -20, 23, -65, 9, -66, -40, -125, -70, 2, -49, -19, -32, 1, 71, 39}
, {26, -5, 19, 18, -150, -15, -130, -27, -116, 60, 33, 65, -101, -9, 32, 52, 74, 13, 40, -86, 17, -35, -65, -25, 9, -134, -61, 9, 30, 24, 14, -3}
}
}
, {{{-83, -151, -24, -81, -74, 7, 66, -25, -137, 25, -30, 29, -4, -29, 91, -37, 54, -6, 16, -19, 9, 47, 54, -36, -88, 3, -25, 5, 1, 74, 37, 15}
, {-13, -27, -68, -23, -27, -26, -45, 23, -31, -2, -59, 14, -66, 10, -74, -33, -27, -3, -23, 0, -46, -14, -101, -7, -53, -5, -43, -8, -48, 34, 18, -54}
, {-69, 18, 28, 48, 79, -99, -2, -42, 25, -38, -29, 13, -121, -4, -71, 15, -82, -91, -40, -94, -44, -50, -120, 44, 7, 72, 55, -52, -101, -85, 32, -12}
}
, {{-174, -56, 4, -23, 70, -75, -205, 10, 4, 99, 2, -55, -58, 6, 12, -76, 49, 27, -85, -162, -73, -67, 74, 24, 77, -78, -71, 67, -2, 19, -12, -4}
, {-68, -56, -25, -16, 26, -80, -175, -4, -3, 18, 41, -16, -161, 29, 1, -99, -34, 26, -71, -220, -18, -210, -49, -24, -17, -32, -12, 50, -47, -3, 33, 28}
, {-170, -48, 72, 4, 56, 100, -61, -137, 14, 4, 14, -87, 44, 25, -6, -90, 54, 27, -74, 71, 74, -4, 97, 66, -184, 28, 58, -13, 70, 109, 9, -57}
}
, {{15, -35, -42, -83, -131, -25, 52, 48, -48, 83, 68, -12, -11, 47, 22, 37, 13, -24, -20, 27, -18, -11, -78, 58, -9, -19, -87, 47, 1, 35, -2, -20}
, {2, 2, -45, 10, 9, -5, -105, 38, -34, 48, 17, -45, 19, -16, 25, 8, -102, 60, -27, 10, 46, 2, -50, 85, 44, -136, -110, -3, -42, -2, -191, 77}
, {80, -60, 21, -20, 14, 91, 53, 4, -20, -120, -14, -107, 36, 11, -36, -76, -134, 47, -43, 70, 0, 89, 98, 86, -34, 15, 12, 3, 33, 44, -31, -15}
}
}
, {{{81, 119, -33, 37, -4, 11, 19, 66, 20, 17, 12, 120, 58, 28, 92, 44, -6, 26, -3, -17, 98, 46, 60, 44, 65, -4, 44, 6, -24, 67, -9, 103}
, {109, 20, -50, 13, -28, 55, -40, 93, -31, -25, -45, -49, 5, -17, 7, 22, -76, 35, -53, 7, -5, -20, -22, 48, -1, -104, -105, -30, -5, 8, -147, 29}
, {-7, 20, 38, 5, 17, -132, -21, 50, 7, -26, 24, -55, 97, 10, 19, 13, -82, -48, -24, -3, -52, -73, -33, 7, 17, -8, -22, -63, -17, -90, -18, 0}
}
, {{72, 52, -4, -21, -9, -53, 71, 70, 91, 19, 12, -163, -100, -19, -122, 11, -54, -7, -1, -57, -85, -22, -53, -38, -9, -1, -56, 60, -86, -110, -56, -71}
, {13, -26, -45, -15, -93, -73, 35, -6, -161, -21, -33, 53, 17, 25, 57, -34, -43, -136, -88, -29, -110, -30, -290, -164, -52, 37, 0, 11, -83, -8, 48, 36}
, {-172, -100, 70, 25, 72, -96, -113, 19, -29, -8, -4, -32, -64, 23, -5, -7, 41, 27, 11, -92, 63, -135, -49, 11, -4, -5, 12, 48, 37, 38, 61, -40}
}
, {{-20, 30, -61, -41, -67, -122, -155, -13, -40, 14, 11, 111, -53, 65, -14, 22, -36, -95, -85, -200, -133, -70, -103, -55, 73, -29, -47, 1, -80, -12, 32, 73}
, {-30, -106, -22, 122, -195, -132, -353, -7, -196, 39, 130, 32, -1, 27, 10, -1, 72, 7, 73, -144, 6, -193, -89, -45, 102, -126, -117, 59, 44, 34, 8, 8}
, {-66, -28, 59, 34, -57, 67, -67, -3, -8, 14, 1, -15, -3, 46, 9, -82, -4, 31, -11, -22, 40, -47, 32, 58, 24, -80, -100, 16, 52, 74, 18, 28}
}
}
, {{{55, 77, -6, 30, 67, 2, 3, 72, -28, 70, 11, 7, -60, -27, 29, -7, -81, 66, 10, 15, -19, -16, -2, 67, 38, -33, -39, 38, -65, -44, -42, 122}
, {-135, -117, 72, -17, 54, -157, 10, -224, 9, -80, 8, 50, -169, -60, -22, -209, -7, -188, -221, -134, -32, -198, -71, -113, -165, 70, 71, -3, -37, -67, 28, -65}
, {-188, 44, 15, -19, -46, 31, -19, -16, -64, 66, 46, 43, -51, 35, 76, -7, 93, -10, 56, 12, 118, 58, 66, -43, -27, 99, 75, 51, 117, 88, 79, 8}
}
, {{-103, 21, -34, 18, -29, 19, -65, 101, 1, 11, 2, 73, 15, 2, 49, -143, -46, -46, -96, -87, 4, -86, -8, 8, -56, -35, -153, -24, -9, 6, -87, 41}
, {-36, -79, 38, 67, -154, 100, -56, -50, -82, -26, -28, -80, -49, -65, -129, 57, -13, 51, -13, 0, 14, -9, 11, 32, -3, 94, -42, -94, -14, 10, 9, -12}
, {77, -56, 8, -36, -17, 54, 96, -113, -34, -133, -102, -74, -10, -163, -126, 2, -168, 27, -5, 48, -13, 45, 30, -23, -133, 83, -7, -156, 14, -37, -20, -145}
}
, {{-12, -72, 68, 6, -14, -30, 31, 7, 6, -18, -33, -80, -1, 28, 1, -52, -24, 15, -12, 56, 27, 1, 45, -10, -28, 27, 1, 13, 42, 28, 78, -5}
, {1, 56, 23, 24, 31, -49, 63, 34, 96, -26, 38, -102, -41, 61, 13, 15, 33, 12, -39, 14, 25, 1, 23, -16, 53, -63, -2, 53, 32, 10, -37, -67}
, {32, -58, 31, 26, 85, 3, 69, 79, 35, -108, 88, -59, 2, 17, -11, -53, -20, 21, -51, 31, 18, 79, 18, 13, -81, -53, -50, -8, 6, 2, -71, -41}
}
}
, {{{20, -46, -36, -119, -105, -49, -63, 62, -66, 64, -52, 11, 37, 21, -17, -44, 33, -13, -7, -65, -4, -25, -67, 14, -66, -31, -61, 1, 46, 12, -15, 116}
, {-165, 82, 6, 12, 0, -8, 18, -9, -14, 34, 104, 121, -46, 52, 53, 73, 52, 0, 44, -90, 70, -17, 8, -31, 73, -16, 1, 53, 87, 57, 40, 47}
, {92, 110, 33, 17, 68, -17, -58, 5, 43, 119, 109, -11, -14, 9, -6, 114, 88, -43, 41, -18, 106, 5, 58, 5, 129, -151, 22, 46, 28, 28, -107, 14}
}
, {{127, -108, -1, -17, -45, 0, -41, 16, 1, -28, -14, -18, 28, -16, 13, 94, 30, 28, 59, 12, 16, 89, 64, 4, -53, -34, 40, -36, 23, 18, 37, 72}
, {86, 37, -60, -37, -66, 12, -4, 34, 11, 9, -12, 10, 12, -84, -11, 90, -29, -13, 10, -26, 26, 8, -134, -34, 42, -65, -254, -40, -10, 42, -240, -20}
, {-28, -18, -175, -73, -155, -42, -32, -119, -43, -128, -114, 69, 23, -88, -28, -71, -164, -173, -102, -29, -109, 26, -113, -63, -53, -51, -192, -143, -142, -110, -89, 41}
}
, {{182, -8, -108, -39, -32, -27, 31, -17, -69, 4, -15, 84, 33, -104, -15, -19, -25, -59, 12, 40, -28, 75, -178, -54, -10, -44, -46, -14, -46, -14, -70, 1}
, {-70, -22, 1, 17, -21, -114, -32, 26, -27, 1, 1, 74, -25, 50, -27, -31, -63, -38, -102, -86, -76, -113, -84, -57, -6, 65, -3, 21, -79, -56, 13, -21}
, {30, -8, -34, 3, 38, -50, 128, -45, 60, -49, -62, -41, -73, 33, -43, -61, 1, -95, -150, -120, -65, -90, -4, -38, -91, 38, 71, -16, -53, -13, 17, -7}
}
}
, {{{78, -43, -45, -24, 16, 28, 20, 34, -4, 75, 36, -41, -25, -5, -48, 22, 37, 30, 76, -15, 53, -11, -97, -3, 1, -38, -97, 25, -14, -16, -36, -19}
, {7, -15, 1, 29, 40, 99, 53, -27, 45, -36, 3, 23, -15, -54, -118, 35, -142, -27, 5, 64, -95, 110, -135, -64, -30, 16, 51, -16, -147, -136, 25, -94}
, {56, -70, -58, -37, -57, 15, 64, -33, -57, -75, -152, -18, -95, -57, -153, -69, -63, -79, -6, 44, -108, 19, -176, -100, -85, 77, 62, -98, -188, -239, 23, -203}
}
, {{-27, 40, -33, 3, 64, -100, -56, 113, 113, -48, -64, 16, 25, -6, 13, -80, -78, -18, -108, -4, -156, -99, -134, -17, -13, -41, -110, -38, -83, -3, -81, 51}
, {-87, 44, -2, 33, 2, -143, -132, 71, 112, 64, 20, 97, -7, 46, 79, -49, -57, 2, -94, -147, -144, -195, -104, -14, 103, 18, -45, 15, -118, -8, 14, 121}
, {-48, 33, 9, 62, 76, -101, -12, 78, 64, 64, 26, 25, -104, 28, -28, 3, -11, -100, -13, -123, -73, -191, -101, -74, 54, -30, 51, 27, -66, -26, -49, -4}
}
, {{-2, -28, 75, -66, -24, 55, -176, -20, 17, 51, 22, -45, -31, 24, 35, -103, 2, 5, -19, -71, 38, -60, 83, 12, -3, 77, 57, 30, 64, 85, 65, -6}
, {49, 36, 22, 18, 33, 21, -201, 4, 89, 81, 28, 59, 25, 39, -9, -44, 28, 26, -40, -65, -12, -105, -43, 55, 62, -49, -121, 55, 36, 70, -35, 80}
, {-72, 8, 43, 61, 84, 5, -111, 63, 11, 42, 0, -207, 7, 51, -108, -43, -30, 20, -23, -102, -38, -90, 10, 56, -21, -26, 4, -10, -26, -18, 9, -37}
}
}
, {{{-20, 3, -5, -24, 7, 11, 56, -96, -72, -2, -70, 68, 9, 0, 92, 7, -1, 0, -74, 27, -40, -33, 59, 26, -123, 57, 55, 11, 39, 13, 3, 21}
, {1, 28, 40, 18, 11, -146, -83, -63, 43, -34, 6, 28, 16, 34, 31, 14, -40, 5, -11, -112, 1, -162, -84, -59, 1, 37, 21, -36, -56, -43, 12, 1}
, {66, -29, 1, 37, -79, 63, -32, -60, -12, 44, -95, -1, -52, -55, 30, -15, 24, -2, 45, 28, 76, 56, 57, -76, -51, 10, 63, -50, 19, 65, 43, 9}
}
, {{-43, -12, -70, -52, -144, -14, 12, -25, -248, 4, 27, 19, -27, -81, 27, 16, 20, -4, 3, -38, 33, -27, 7, -65, -17, 35, -61, -48, 13, 17, -28, 0}
, {-106, -130, -17, -1, 110, -1, -125, 74, -101, 6, 30, -28, 84, 3, 44, -63, 20, 17, -25, -13, 31, -26, -20, 33, 14, -180, -72, 60, 52, 67, -39, 40}
, {-83, 22, 23, 37, 54, -142, -246, 60, 43, -13, 23, -106, 112, 47, -10, -130, -19, 20, -171, 15, -144, -244, -35, 59, -26, -191, 29, 38, -10, 37, 1, -16}
}
, {{-55, 96, 18, 22, -71, 56, -45, 0, -5, 37, -38, 46, -15, 76, 31, 34, 47, 34, -8, -41, -31, 6, -12, 69, -1, 53, 3, 37, 43, 12, 50, 9}
, {-79, -56, -28, -48, -45, -25, 40, -92, -94, -17, -70, 39, 14, -37, 10, 49, 83, -29, 40, 42, 43, 63, -64, -63, -43, -48, -24, -64, -14, -32, 62, 22}
, {-66, 58, -91, -9, -77, -33, -53, -6, -15, 50, 49, 66, 59, 43, 106, 15, 46, 13, 9, 37, 28, -53, -91, -58, 36, -39, -161, 56, -4, 32, -18, 98}
}
}
, {{{22, 54, -100, 2, -90, -106, -34, 45, 53, 53, 39, 115, 60, -98, 14, -46, 6, 21, -63, -29, 42, -111, -176, -19, -19, 62, -5, 15, -7, 12, 14, 20}
, {-15, 22, -52, -31, -21, -39, -56, 142, 4, -20, 16, -4, 96, -3, 10, -78, -62, 32, -59, 44, 54, -63, -125, 85, 50, -85, -64, -24, -3, 6, -97, 12}
, {-71, 20, -9, 7, -2, -96, -37, 42, 11, 58, 26, 42, 41, -7, 9, -32, -43, 6, -25, -48, -14, -138, -40, -6, 65, -26, -61, 14, -107, 39, -39, 83}
}
, {{-25, 86, 7, 63, -49, -82, 38, 79, 142, 53, 26, -28, 113, -2, 31, -8, -23, -50, -21, -29, -55, -113, -204, -49, 53, -15, -31, 6, -56, -49, -3, 83}
, {-34, 111, -20, 66, -74, -86, -22, 1, 19, 25, 12, 11, 20, 18, -3, -25, -58, -203, -2, -120, -133, -72, -43, -110, 5, -22, -50, -31, -135, -68, 1, 74}
, {-32, 57, -21, -33, -23, -97, -121, 24, 24, 91, 21, 134, 46, 37, 94, 25, 48, -70, 42, -129, 49, -141, -142, 26, 41, 45, -22, 42, 20, 10, 27, 25}
}
, {{16, 52, 22, 72, -26, -154, -149, 1, 62, 17, 52, -37, -71, 9, 7, 34, 72, -31, 22, -267, -63, -242, -156, -78, 23, -122, -102, 42, -12, -1, 34, -28}
, {-326, -96, 5, 63, -72, -113, -180, 23, -85, -19, 32, 25, 1, 27, -34, 3, -2, -34, 30, -132, -59, -228, -47, -38, 70, -40, -49, 0, -37, 18, 71, 54}
, {-6, 19, 36, 32, -51, -82, -207, -7, 33, 48, 28, -34, -152, 19, -153, -5, -6, -8, 15, -192, -59, -118, -250, -85, 59, -221, -121, -11, -18, -101, 2, -70}
}
}
, {{{1, 20, 44, -4, 123, 2, -14, -25, 35, -45, 33, -18, 82, 58, -11, -21, -43, 85, -33, -31, -5, -87, 75, 29, -100, 5, 55, 27, -13, 20, 16, -72}
, {-9, -7, -19, 15, -53, 5, 82, -44, 12, 0, 53, -124, -30, -45, 6, 53, 3, 35, 52, 60, 15, 56, -76, -102, -26, 26, -13, -24, 10, -8, 2, -42}
, {89, -70, -60, -110, -143, 66, 138, -95, -112, -65, 0, 4, -105, -24, -64, 82, 36, 65, 67, 94, 67, 80, 115, -17, 47, -30, -121, -60, 95, 18, -1, -160}
}
, {{-70, -160, -1, -56, 35, -22, -40, -64, -113, -12, 12, -37, -25, -22, 39, 26, 72, 40, 24, 61, 45, -7, 0, -30, -49, 57, 32, 55, -10, 10, 52, -63}
, {-91, 18, 8, 35, 1, -12, -164, 62, 60, -84, 27, -172, -85, 25, -20, -104, -28, 22, -102, -41, -29, -147, 6, 26, -19, -44, -7, 22, -14, -9, -59, -111}
, {45, -12, 24, 11, 62, -150, 42, -13, 33, -260, -47, -202, -87, -54, -182, -91, -222, 41, -242, -69, -265, -190, -42, 26, -136, -14, -30, -148, -61, -160, -19, -82}
}
, {{9, 23, 25, 24, -82, 42, -14, -2, -10, 6, -74, 55, -75, 17, -58, -69, 62, 58, -27, -26, 37, 53, -11, 35, 25, 16, -52, 38, -6, -6, 45, -66}
, {18, 43, 20, -54, 25, -106, 114, -8, -3, 23, 94, -131, -103, 29, -62, 48, 126, 9, 20, -26, 11, -19, -4, -75, 3, -2, -37, 29, 36, -95, 3, -117}
, {-30, 26, -38, -11, -8, -65, -58, 59, 21, 10, 47, 103, 19, 41, 93, 64, -53, -111, -46, -141, -79, -66, -105, -26, 23, 26, -7, 54, -72, -21, 10, 66}
}
}
, {{{-79, 2, -4, -53, -75, 26, -80, -20, -76, 33, 9, 5, -11, 7, -26, -177, 32, -10, 5, -56, 9, -8, 2, -33, -23, -52, -42, 9, -2, 15, 43, 31}
, {-73, 87, 33, 38, 37, -105, -63, 44, 42, 30, 41, 51, -116, 38, 4, 3, 23, 7, 72, -102, -25, -163, -61, -27, 12, 57, 10, 20, -10, 22, 28, -11}
, {-39, 11, 3, 0, -87, -53, -4, -99, -7, 31, 15, -32, -95, -17, 38, -52, 46, -82, -21, -85, 50, -70, -51, -113, -36, 14, -26, -38, -5, -65, 45, 5}
}
, {{61, -129, -122, -244, -130, -13, -52, 74, -117, 2, -12, 3, 76, -24, 18, 59, -4, 6, -56, 10, -29, 10, -22, -19, -58, -140, -139, 11, -39, 43, -87, 4}
, {-153, 51, -68, 26, -20, -227, -255, 24, 43, 10, 49, 23, 49, 6, -13, -13, -57, 24, -51, -40, -2, -269, -133, 19, 69, -158, -191, 30, -2, 51, -94, 53}
, {3, 75, 17, -11, -36, -8, -69, -5, -2, -5, 37, 5, -120, 21, -24, 42, 56, -74, 61, -71, 11, -39, 20, -64, 39, -158, -66, 9, 7, -7, 62, 48}
}
, {{2, -184, 19, -172, 19, -53, 42, -136, -42, -81, -258, -8, 27, -4, 57, -59, 0, -57, -140, 40, -41, 10, 27, -78, -297, 101, 153, -26, -59, 37, 6, 46}
, {89, -4, 4, 23, 68, -3, -129, 32, 71, 77, 16, -75, 51, 83, -8, -102, -38, 54, -149, -1, -27, -15, -42, 41, 63, -251, -238, 61, -4, 13, -31, 81}
, {-104, -81, 20, 26, 32, -1, -135, 64, -8, 2, -9, -122, -82, 0, -25, -141, 10, 86, -78, -80, 51, 12, 56, 11, -17, 43, 53, -41, 49, 3, 85, -83}
}
}
, {{{-8, 17, -3, -48, -25, -19, -15, -96, 14, -19, -101, 41, 33, -31, 93, -5, -2, -11, -106, -12, 10, -42, 43, -32, -139, 44, 18, 3, -78, 66, 28, 16}
, {54, -144, 33, 39, -29, 2, 26, 25, -57, 22, -66, -75, -31, -13, 11, -26, 6, 62, 81, -22, 19, 54, 121, 17, -24, 61, 68, 1, -2, 6, 16, -72}
, {39, 17, 12, 61, 79, -91, -29, 103, -3, 42, 1, -94, 79, 0, -11, 43, -34, 54, -49, 14, -95, -28, 2, 75, -27, 22, -26, -7, -21, -99, -41, 41}
}
, {{13, 32, 17, 9, -6, 24, -17, -1, -95, 14, 3, 42, -3, 22, 30, -82, 44, 33, -45, -42, 6, 43, 10, 103, -111, 78, 18, 23, 13, 39, 44, -2}
, {-111, -227, 11, -127, -63, 17, 128, -45, -255, -39, -48, -69, -36, 19, 16, -114, 66, 48, -52, 68, 20, -9, 32, -31, -93, 56, 27, -7, 14, 43, -5, -80}
, {-169, -10, -16, -72, 25, -117, -154, -60, -38, 48, 74, -16, -21, 44, 73, 9, 74, -18, 68, -34, 58, -155, 14, -86, -8, -103, 9, 60, 23, 16, 5, -30}
}
, {{38, -67, -21, -102, -30, -58, 66, -17, -108, -22, -146, 29, 39, 29, 93, 22, -46, -4, -39, 89, -49, 28, 49, -45, -117, 6, 35, 56, -43, -2, -11, 38}
, {-199, -105, 8, 31, -108, -34, -134, 22, -107, -6, 60, -26, 25, -38, -17, -48, 10, 12, -1, -80, 39, -55, 39, 26, 53, -147, -36, 24, 39, -3, 6, 14}
, {-38, -2, -14, 30, -42, 19, -93, -35, -18, -43, -54, 27, -162, -82, -89, 17, -14, 14, 15, -75, 23, 3, -51, 21, -3, -169, -281, -83, 62, 8, 23, -84}
}
}
, {{{89, -26, 27, -4, 43, -43, 15, -3, -10, -28, -2, -26, -11, 80, 4, -95, -16, 46, -76, 45, -31, -36, 103, 66, -25, -13, 37, -46, 60, -7, -20, 40}
, {3, 11, 8, -3, -12, -3, 93, -89, -7, -18, -30, 36, 24, 17, 6, 43, -28, -24, 92, 28, -36, -34, 59, 10, -22, -37, -68, -11, 40, -8, 5, 22}
, {108, -18, 16, -32, -7, -47, -80, -23, -19, -19, -6, -27, 31, -10, -26, 23, -35, -10, 3, -14, 14, -73, 18, -27, 40, 17, -50, 32, 2, -10, 32, -81}
}
, {{-84, -56, 33, 19, -6, -87, 60, 41, 20, -35, 6, 31, 79, -16, 64, -117, -71, 4, -72, 18, -20, -78, 15, 56, -4, -2, 5, 45, -59, -51, -34, -9}
, {-74, -241, 61, -22, -18, -39, 5, -44, -156, -7, 62, -6, -99, 82, -34, -70, 92, -11, 69, 26, 54, -38, -42, 9, -82, -18, 51, 20, 76, 32, 71, -158}
, {-80, 15, 29, 22, 51, -30, -20, 2, 21, -8, -23, 38, 6, 39, 60, 64, 16, 2, 52, 31, 42, -71, 69, 20, 11, -50, 80, 32, 54, 81, 63, 1}
}
, {{-101, -30, -166, 14, -67, -10, -130, 19, -99, 57, -56, -5, 64, -10, 64, -125, -126, -29, -82, 11, -27, -154, -101, 36, 73, -217, -301, -27, 15, 32, -136, 33}
, {-143, -80, 55, 79, -26, -151, -296, 5, -41, -95, -5, -72, -25, -22, -79, -154, -46, -6, -119, -136, -10, -233, -5, 7, -27, -200, -169, -1, -90, -7, -7, 67}
, {-53, 8, -13, 70, -8, 27, -77, 86, 23, 42, -18, -20, 6, 11, -13, 27, 4, 39, 36, -34, 44, -94, -110, 15, -4, -36, -94, -4, -6, -39, -36, 31}
}
}
, {{{114, 142, 5, 28, 34, 90, 52, 54, -11, 78, 77, 8, 90, 46, 100, 77, 37, 37, 72, 64, 88, 88, 15, 24, 93, -57, 19, 42, 26, 57, -48, 42}
, {52, 58, 42, -52, 20, 49, 15, -7, 51, -2, 36, 1, 24, -27, 48, 24, 21, 39, -42, 76, -23, 64, 27, 32, -5, -36, 30, 2, 11, 21, -50, 44}
, {136, -3, 7, -8, 10, -72, -45, 14, 29, -41, -18, -54, -19, -76, -103, 67, -76, -35, 70, 0, -6, -15, 52, 24, 18, 31, -29, -29, -17, -61, -52, -28}
}
, {{28, 58, 20, 67, 51, 60, 61, 52, 111, -17, -22, -37, -68, -39, -134, 36, -63, 62, 21, 1, 17, 13, 41, 58, 23, 52, 17, 37, 1, -48, -39, -125}
, {66, 31, -74, -16, -48, -29, 78, -129, -20, -46, -170, 4, 29, -121, -12, 45, -85, -47, 33, -14, -85, 36, -38, -78, -165, 22, -26, -73, -96, -20, -68, 33}
, {-208, -71, 1, 9, 33, -206, -13, -15, -73, -36, -43, -46, 24, -21, 12, -202, -11, -52, -191, -51, -70, -140, -154, -5, -32, 17, 35, 58, -45, -1, -1, -26}
}
, {{98, 7, -93, -84, -90, -26, 127, -53, -58, -40, 9, 41, -55, -20, -79, 3, -55, -61, -66, 13, -29, 120, -50, -63, -9, -28, -116, -66, -50, -74, -107, -90}
, {-122, 7, -46, 14, 7, -100, -164, 31, -113, 75, 83, 103, -28, -45, 126, 5, 71, -45, -100, -150, 37, -134, -16, -51, -58, 12, -19, 54, 7, 80, 44, -12}
, {-43, 66, 2, -26, -59, -82, -62, 18, -71, 50, 28, 6, -18, 35, 38, 19, -5, 24, -31, -12, -35, -111, 0, -40, 4, -71, 23, -7, 10, 18, -31, -61}
}
}
, {{{-22, 58, 9, 28, 43, -24, -1, 13, -63, -41, -16, 44, -80, 21, 4, 21, 40, -1, 27, -2, 41, -18, -68, -6, -16, -36, -68, 7, 50, -29, 6, -14}
, {-32, 91, 23, -55, 50, 63, -89, 10, -44, 34, 28, 37, -22, 58, 49, -21, 63, -29, 51, -31, 0, -83, -49, -23, 18, 75, 49, 82, -14, -12, 59, 51}
, {45, 17, 44, 27, 40, 11, 19, -93, 34, -11, -71, -88, -37, 45, -35, -14, -41, 11, -23, -2, -18, 44, 16, -36, 14, 38, 74, -118, -54, -58, 33, -55}
}
, {{17, -20, -86, -136, -113, 92, -69, 27, -177, 90, 21, 128, 72, -52, 81, -45, 33, -5, -25, -17, 16, 24, 18, 49, -42, -5, -98, -26, -3, 22, -54, 63}
, {11, -121, -20, 13, 95, 38, 28, -10, -53, 26, 4, -6, -38, 101, -18, -81, 35, 33, 53, 4, 8, 64, 43, 1, 20, -11, -9, 36, 9, -25, 36, -14}
, {-69, -145, 32, 3, -1, 36, 8, -40, 75, -23, -11, -224, 27, -10, 37, -155, 18, 37, 6, 107, 40, 45, 22, -22, -61, 52, 71, 28, 24, 36, 80, -226}
}
, {{46, -83, -35, -118, -81, 132, 114, -80, -52, -35, -104, -81, 21, -114, -144, -8, -3, -37, -2, 34, -4, 100, -17, -89, -169, 73, -42, -16, -4, -54, -41, -69}
, {3, -114, -109, -209, -219, -27, 61, -246, -180, -162, -119, 74, -22, -203, -36, 89, 48, -27, 29, 20, -45, -5, -129, -76, -281, 56, -2, -146, -61, -78, 0, -55}
, {-87, -69, -15, -90, -109, -40, -40, -81, -207, 1, 19, 60, -44, -46, 38, -19, 7, -50, 40, -59, 38, -58, -39, -58, -49, 55, -61, 11, 68, 15, 34, 10}
}
}
, {{{27, 27, -134, -28, -112, 10, 54, 7, -104, -36, -7, 104, -18, -25, 50, 125, -3, -146, 20, -61, -46, 54, 90, -131, 33, -58, -131, -15, -88, 0, -32, 129}
, {93, -52, -66, 0, -149, -45, 10, 86, -60, 49, -32, 30, -7, -5, -21, 59, -66, -55, -14, -40, -26, 31, -36, -58, 60, -41, -106, -33, -19, 9, 1, 89}
, {103, -9, -73, -46, -168, -76, -42, 26, -46, 18, 10, 75, -15, -4, 40, 2, -13, -137, 5, -62, -19, -86, -173, 16, 49, -98, -151, 20, 10, -51, -100, 19}
}
, {{35, -93, 0, 89, 13, 0, -67, 14, 38, -39, -24, -60, -53, -16, -58, -43, -4, 68, -50, -2, -45, 51, 37, 31, -72, 36, 26, -45, -42, -9, 19, -28}
, {-14, -96, 12, -51, -49, 2, -3, 62, -144, -39, 32, -36, -31, -29, -20, -53, -4, 27, -46, -17, 21, -29, 55, -65, -7, -17, -74, -51, 4, 25, -38, -33}
, {-80, 32, 24, 34, 61, -72, -214, 2, 6, 7, 73, 31, -105, 16, -24, 63, 29, -44, 50, -193, 60, -145, -13, -34, 73, 51, 57, 56, 16, -35, 59, 28}
}
, {{89, -99, -29, -99, -97, 72, 57, -26, -114, -13, 38, 59, -84, 44, -44, 71, 15, -1, 61, -19, 31, 99, 76, 53, 29, 69, 36, -27, 44, -15, -9, -72}
, {49, -20, 12, 35, -41, -48, -43, -21, -189, 34, -5, 20, 16, -39, 74, -73, -18, -11, -77, 7, 30, -53, 68, 42, 15, -224, -53, -27, 1, 39, -39, 3}
, {-94, 17, 23, 78, -9, -83, -258, 41, 16, 43, -44, -21, 0, 47, -55, -56, -105, 54, -95, -205, -131, -289, -136, 95, 27, -130, -103, -4, 0, 11, -64, -15}
}
}
, {{{29, -6, 33, 29, 50, 54, 54, -82, -19, -42, -43, 3, -9, 24, -20, 23, -2, 43, 47, 88, 5, 83, 47, 7, 9, 7, -2, -19, -25, 71, 64, -10}
, {-1, -65, -52, -40, -49, -4, 64, 17, -108, 44, 38, 15, -45, 52, 39, 39, 90, 72, 58, 1, 52, -52, 15, -42, 46, -27, 30, 11, 12, 42, 26, -14}
, {107, -13, -4, 71, 37, -21, 0, 72, 2, 33, 50, 29, -38, 29, 14, 37, -9, -18, -15, -21, 17, 47, -37, 37, 55, -7, -11, 63, -21, -46, -7, -17}
}
, {{3, -28, -72, -83, -48, 66, -25, -2, -59, -3, -40, 20, 39, -69, 51, 5, 32, 40, -43, 42, 36, 38, 53, -2, -93, 29, -2, 24, 43, 35, -8, -16}
, {-66, 68, -118, -57, -112, 24, 3, 3, -18, 42, 12, 98, 33, -35, 19, 37, 16, 10, 62, 4, 35, -13, -39, -50, -15, -63, -89, 13, 8, 45, 4, 34}
, {5, 8, -12, 24, -23, 84, 0, 25, -46, 27, 21, 17, -35, -17, 42, 59, 29, -25, 58, -13, 78, 13, 36, -10, 12, 86, -19, 39, -9, 21, 41, 3}
}
, {{51, 80, -226, -46, -157, -129, -61, -44, -75, 31, -32, 39, 124, -125, 95, 37, -29, -265, 7, -16, -29, -77, -257, -194, 57, -72, -83, 9, -88, -22, -54, 44}
, {88, -41, -255, -206, -241, 186, -99, 14, -66, -15, -73, 33, 62, -178, -60, 70, -44, -32, 42, 6, -5, 126, -10, 3, 12, -160, -132, -101, -35, -47, -228, 19}
, {14, -15, -218, -203, -114, -47, 11, -76, -169, -171, -204, 94, -21, -204, -49, 57, -204, -8, 9, 7, -67, -31, -126, -179, -136, -1, -44, -230, -128, -125, -140, -46}
}
}
, {{{-8, -126, -93, -194, -47, 101, 16, -7, -253, -4, -83, -108, 97, -31, 23, 88, -4, -29, 23, 40, 4, 73, 44, 43, 5, -17, -167, 74, 17, 37, -19, 29}
, {-48, -50, -159, -63, -133, 113, -82, 62, -109, 37, -25, 62, 49, -67, -21, 109, 38, 60, 58, -7, -16, 54, -22, 10, 39, -114, -221, -23, -23, 12, -101, 86}
, {-29, 77, -114, -19, -97, 40, -21, 20, 26, 14, -5, -67, 1, -66, -108, 78, 66, -8, 50, -16, 56, -9, -133, -50, 3, -46, -158, -1, -23, -111, -77, -21}
}
, {{-84, -117, -10, 3, -60, -90, 26, -98, -26, -23, -18, 3, -121, -25, 23, -82, 72, -101, 51, -123, -35, -51, -31, -41, -37, 63, 28, 5, 23, 6, 41, -88}
, {0, -56, -39, -35, -50, -116, 151, -82, -96, 1, 7, 15, -148, -23, -53, 106, 111, -44, 6, -115, -14, 75, -39, -109, -90, 87, 23, -14, -44, -25, 32, -58}
, {-12, -21, 5, 32, -34, 67, 1, -26, -64, -5, -27, 23, -162, 34, 7, 27, 24, -68, -79, -59, -2, -9, 105, 15, -53, 60, 46, -43, -23, 69, 67, -62}
}
, {{-68, -86, 39, 19, 22, 3, -134, 8, -56, 24, 44, -6, -134, 80, -1, 27, 30, 66, 44, -29, 31, -23, 31, 10, 7, 10, 94, 22, 31, 37, 79, 5}
, {-39, -118, 41, 2, 47, 27, -66, -20, -61, 12, -14, -27, 1, 71, -71, -54, -7, -9, 10, 21, -12, -4, -15, 57, -26, -30, 2, -14, 25, 20, -2, -27}
, {-63, -114, 11, 42, 110, 52, 87, 35, -28, 24, 3, -80, 120, 55, -13, -98, 7, 30, -48, 67, 28, 34, 46, 54, -21, 122, 42, 13, -11, 63, 74, -13}
}
}
, {{{50, -19, -106, 19, -68, -48, -21, 78, -22, 62, 37, 12, 11, -15, 113, -35, 58, -3, -80, -33, -16, 9, -39, 42, 47, -29, -65, 27, 14, -39, -19, -23}
, {-18, -159, -252, -56, -160, -48, 48, -34, -91, -41, -51, 11, 35, -25, 2, -47, -127, -71, -42, 11, -61, -32, -69, -25, -48, -18, -139, -15, 3, 1, -120, 45}
, {-21, 82, 1, 39, -96, -74, -15, 48, 44, 71, 57, 62, 126, -13, 38, 31, -23, -31, 7, 18, 6, -50, -135, -24, 20, -47, -115, -15, 9, -13, -42, -28}
}
, {{5, 10, 6, 98, 36, -12, -66, 60, -31, 35, 36, 2, -52, -25, -99, -40, -18, 12, -5, -19, -2, -12, 24, 51, 62, 51, -7, -14, -12, -11, -4, -79}
, {-104, 7, -1, -18, -56, -18, -32, 10, -140, -19, -42, -17, -59, 46, 11, -128, 12, 6, -153, -48, 17, 29, -2, -3, 39, 36, 24, -15, -10, 13, 35, 52}
, {1, 24, 0, 31, 41, -41, -274, 27, 15, 27, 15, 46, -87, 78, 20, -6, -15, 33, 30, -111, 32, -77, 39, -39, 98, -80, -62, 11, 53, -11, 11, 107}
}
, {{43, 114, -33, -53, 43, 75, 57, -15, 70, -48, -70, 57, 10, -23, 20, 54, -59, 64, 17, 47, 21, 0, 136, 9, 49, -12, -52, -25, -29, -18, -142, 63}
, {-101, -9, -181, 85, -229, -116, -84, 59, -146, 23, 49, 87, 9, -81, 29, 16, -93, -44, -17, 52, -25, -77, 31, -56, 99, -151, -177, -16, -27, 10, -108, 54}
, {71, 15, -61, 37, -59, 2, -108, 31, -33, 62, -27, -30, -43, -59, -61, 32, 33, 16, -35, -53, 83, -21, -8, 54, 50, -150, -180, 48, 40, 0, -18, 46}
}
}
, {{{-16, -14, 43, -11, -2, -191, -32, -33, -68, 7, 32, -3, 24, -2, -3, -81, 67, -50, -16, 17, -2, -70, 25, -67, 44, -27, -2, 87, 14, 54, 48, -2}
, {2, -15, 70, 44, 40, 46, 41, -43, -12, 32, 31, -23, 72, 52, 41, -59, 10, 13, 10, 68, -2, -29, 29, 15, 16, 40, 41, 25, 92, 34, 48, 47}
, {58, 55, 10, 49, -4, 86, -59, -82, 34, -14, -24, -99, -9, 75, 54, 1, -27, 58, 33, 33, 32, -19, 145, 52, -10, 14, -4, 32, 36, -17, 48, -50}
}
, {{54, -6, 11, 37, -67, 24, -100, 66, -54, 33, 13, 51, 29, 9, 23, 53, -89, 26, 34, -7, -1, -27, 20, 44, 55, -130, -11, 76, -7, 13, -3, 40}
, {79, -251, 26, -20, 43, 62, 101, -58, -109, -203, -21, -197, -31, -17, -107, -80, -37, 16, 75, 47, 29, 74, 27, -9, -127, -16, -65, -142, 37, -6, -119, -179}
, {4, -16, 3, -75, -9, -3, 18, -29, 12, -204, -56, -69, 0, -15, -12, -70, -27, 9, -44, 47, -13, -33, 11, 49, -134, 64, 2, 32, 2, 30, -86, -79}
}
, {{-17, 60, -18, 35, 27, 18, 71, 8, 125, -50, -84, -115, 92, -55, -81, -61, -78, 24, -51, 63, -62, 31, 59, 8, 40, -11, 5, -75, -48, -46, -20, -132}
, {15, 16, 0, 31, -15, -41, 11, -62, 12, -96, 33, 15, -21, -11, -37, -60, -107, -97, -94, -17, -37, -87, 35, -1, -60, -42, -22, -99, -29, -76, -43, 36}
, {34, 23, 21, 43, 41, 63, -33, 29, 55, 13, -18, -15, 31, -14, 2, -22, 14, -8, -50, -34, -23, 39, 35, 51, -8, -116, 20, -7, 1, -20, -21, 53}
}
}
, {{{-72, 28, -120, 12, -120, -27, -7, 30, 27, 20, 7, 42, -80, -51, -10, -23, -15, -37, -20, -114, -29, -31, 10, -85, 22, -27, -120, -32, -98, -29, -91, -41}
, {6, -28, -53, -3, -138, 39, 12, -36, 14, 0, -77, -115, -158, -142, -22, 5, -101, -80, -49, -24, -28, -19, -39, -139, -115, -53, -62, -52, -21, -14, -19, -67}
, {45, -143, -44, 44, -118, 161, 11, -58, -41, -147, -76, -230, 36, -171, -164, -8, -195, 46, 20, 46, 64, 79, 14, -25, -93, 31, -55, -245, -2, -64, -14, -40}
}
, {{-62, -215, 27, -34, 50, 70, 58, 12, -207, 22, 72, 30, -14, 48, -7, -28, -1, 60, 77, -8, 98, 23, 49, 21, -90, 40, -18, -18, -3, 8, -29, -11}
, {-7, -149, 19, -14, 41, 36, 137, -7, 35, 5, -21, -109, -6, 16, -7, -74, 54, 49, 0, 122, 67, 174, 6, 13, -64, -16, 56, 28, 46, 50, 3, -148}
, {-170, 0, 25, 60, 28, -78, -78, -17, 69, -144, 36, -326, -137, -4, -156, -223, -65, 73, -128, 12, -71, -157, -3, 20, 2, -27, 45, 11, -26, -87, -48, -194}
}
, {{-26, 112, -74, -25, -47, -9, -14, -12, -5, 72, -41, -15, 36, -12, 62, 35, 3, 26, 49, 27, -10, 74, -6, 42, -20, -34, -33, 14, -1, -13, -63, 25}
, {-130, -151, 18, -51, -79, 99, 68, -6, -194, 54, 36, -6, -65, 59, 7, 97, 78, 20, 93, -64, 19, 86, -21, -53, -9, -1, 31, 43, -13, 20, 50, -69}
, {-12, -1, -48, -23, -38, 18, -23, -3, -7, 8, 44, 7, 8, 51, -10, 42, -2, -47, 25, 31, 28, 65, -19, -37, 68, -112, -28, 73, 21, -8, -67, -20}
}
}
, {{{71, -63, 18, 23, 25, 65, 75, 24, 48, -29, 128, -45, -41, 0, -98, 14, -28, -32, 124, 3, -27, 111, -38, -73, 90, -57, 19, -17, -33, -128, -17, -29}
, {24, -86, -22, -33, -71, 95, 86, -153, -8, -77, -184, 3, -14, -72, -102, 82, 22, -6, -7, 45, 10, 65, 92, -41, -164, 36, 72, -86, -54, -174, 96, -79}
, {-44, -143, 28, 14, 19, -11, 43, 7, 48, -156, -175, -2, -10, -62, -57, -103, -140, -44, -91, 21, -69, -26, 9, -31, -76, 75, 38, -39, -103, -69, 38, -67}
}
, {{-38, -12, -41, -59, -13, -155, -114, 86, -41, 2, -5, 42, 8, -6, -15, -31, -34, 41, -97, -101, -11, -97, -40, -25, -51, 54, 16, 51, -62, -3, -1, 35}
, {-68, -204, 1, 0, -58, -13, -114, -22, -170, 38, 35, 36, -11, 45, 7, -100, 44, 73, -12, -61, 42, -9, 62, -3, -96, 21, 3, -16, 10, 46, -21, 16}
, {-72, -104, 35, -1, 79, 44, 86, -52, 0, -4, 8, -212, -19, 14, -18, -50, -59, 76, -16, 51, 29, -9, 40, 11, -58, 29, 67, 7, 33, 27, -23, -93}
}
, {{-56, -106, 14, -4, 1, 5, -65, 43, -46, 49, -22, -31, 41, 26, -6, -83, -42, 76, -64, -19, -103, -34, -68, 32, 44, -86, 35, -17, -3, -29, -3, -16}
, {-98, -80, -12, -3, 17, 38, 29, 34, 0, 11, -10, -90, 50, 34, -88, -66, -16, 10, -46, 38, 18, 58, -22, 42, -51, 38, 13, 46, 65, 56, 23, 48}
, {-44, -112, 8, -37, -13, 23, -4, -8, -78, -71, 11, -11, -18, 11, 15, -37, 35, 10, 46, 60, 36, 19, -5, -27, -40, 25, 30, 58, 18, 84, 28, -7}
}
}
, {{{-36, -9, 55, 52, 64, 69, -10, 58, 16, -37, -11, -16, -30, -49, -10, 64, -1, 23, 11, 7, -35, -40, 56, 31, 65, 57, 80, 6, 46, 4, -4, -34}
, {55, -2, -57, 25, -30, -44, -30, 8, -50, 5, -5, -11, 8, 59, -59, -77, -83, -71, -70, -38, -39, 26, -33, -49, 2, -22, -67, -4, -33, 6, -33, 116}
, {-47, -33, 110, 62, 17, -19, -11, -30, 30, -44, 11, 64, -31, -22, -1, -6, -39, -25, -47, -25, -19, -133, 152, -45, 13, 63, 36, -3, 59, -59, 48, 33}
}
, {{50, 55, -16, -47, -37, 50, 48, 34, -2, 40, 20, -17, -33, 15, -4, 23, 12, 24, -51, 49, 41, 8, 2, -45, 31, 64, 24, 31, 2, 10, 8, -35}
, {-123, -173, 8, -75, -111, -67, -294, 25, -247, 63, 69, -2, -114, 45, 23, -90, 44, 45, -13, -86, 38, -135, -12, 8, -10, -41, 26, 25, 61, 31, 50, 45}
, {28, -88, 26, -5, 29, 65, 34, -70, -110, 9, 24, -64, -95, -36, -55, 12, 72, 57, 29, 14, 124, 44, 53, -16, -96, -182, -54, -39, 54, 19, -37, -142}
}
, {{26, -61, 21, -84, -52, -12, 24, 0, -17, 35, 54, 54, -2, 26, 10, -17, 15, 14, -29, -34, -28, -10, 5, 60, 82, -90, -79, 6, -18, 27, -8, 2}
, {-1, 44, -5, 46, 22, 25, -109, 5, -14, 19, -76, -89, 40, -49, -53, -55, -100, 55, -46, -13, 16, -44, -50, 99, 50, -172, -99, -38, 22, 24, -95, -5}
, {8, -13, 16, 48, 94, 69, 17, 41, 64, -187, -36, -175, 87, -12, -66, -86, -152, 38, -63, 97, -4, 13, 44, 64, -65, -5, 2, -49, -20, -29, 8, -57}
}
}
, {{{-57, 11, -36, 71, -43, 52, 26, 3, 68, 6, 62, 1, 31, -40, -15, -56, 18, 4, -55, -10, 8, -17, -69, -16, 48, 6, 13, 33, 27, 28, 12, 43}
, {-56, 78, -154, -57, -153, 63, -91, 70, -24, -74, -51, 9, -7, -118, -9, 28, -83, 0, 61, 35, -8, -52, -97, 24, -13, -55, -109, -94, -55, 35, -137, 31}
, {-111, 34, -23, 56, 25, -87, 38, 10, 5, 17, 30, 61, -35, -17, -62, -64, -24, 60, -31, 68, -108, -101, -162, -19, 83, -1, -60, 25, -142, -34, 25, 72}
}
, {{-10, 65, 24, 57, 58, -78, -105, 49, 44, 36, 124, -41, -81, 9, -37, 49, 6, -13, 65, -120, 21, -79, -104, -47, 45, -129, -15, 26, -30, -144, -31, -35}
, {-54, -49, -74, -20, -88, -141, -32, -33, -127, 7, -70, -37, 8, 13, -58, -84, -71, -191, -232, -129, -162, -55, -59, -43, 1, -70, -100, -52, -108, 5, -24, 78}
, {1, 23, -12, 38, -17, -2, -68, 93, 13, 125, 44, 130, 76, -18, 144, 86, 50, -9, 66, -5, 51, -35, -41, 2, 111, -109, -2, 45, 18, 23, 34, 99}
}
, {{-144, 71, -17, 14, -52, 26, -116, 33, 31, -18, 35, 59, -15, 79, -83, 17, -3, -16, 38, -92, 53, -145, -19, 28, 105, 16, 47, -10, 6, -57, 13, -5}
, {-163, -119, 62, 133, -4, 51, -63, -51, 12, -6, 17, -71, -14, 75, -15, 61, 37, 3, 21, -55, -25, 58, 33, -17, -32, 104, 100, 11, -10, 8, 81, -24}
, {43, 20, 18, 6, 45, 38, 88, -53, 41, -80, 24, -101, 17, 3, -114, 23, 18, -2, 25, 37, -51, 90, 32, -26, 52, 30, 26, -2, -52, -117, 27, -131}
}
}
, {{{70, 129, -87, 49, -14, 18, -10, 12, 38, 23, -56, 46, 6, -6, 6, 17, -48, 0, 16, -24, 14, -46, -135, -24, 39, 53, 17, 8, -76, 0, -17, -10}
, {-11, -36, -93, -32, -133, -74, -89, -35, 25, -57, -83, -40, -12, -143, -63, -34, -18, -34, -17, -85, -68, -70, -119, -148, 26, 7, -86, -80, -168, -20, -84, 45}
, {-58, -36, -82, 15, -7, -159, 21, -12, -61, 3, 11, 139, -3, -26, 28, -36, -47, -73, -126, -98, -76, -95, -218, -170, -38, 65, -22, -12, -161, 12, 7, 74}
}
, {{-16, 101, 18, 28, 57, -62, -8, 45, 50, 75, 49, 101, 113, 34, 80, 66, 39, 27, 2, -17, -21, 46, -19, 53, 24, 40, -24, 15, 53, 17, 4, 107}
, {-27, 115, 5, -71, -51, -71, -67, -200, 20, -112, -59, 77, 49, -21, -33, 104, -110, -216, -46, -145, -205, -28, -71, -172, 49, -87, -24, -146, -148, -97, -26, 113}
, {-140, -9, -81, 11, -58, -295, -124, 52, -40, 44, 75, 106, -26, -23, 59, 89, 75, -91, 11, -111, 28, -261, -190, -102, 65, -50, -75, 32, 14, 55, 0, 69}
}
, {{41, 66, 41, 63, 17, -9, -14, 45, 42, 23, 68, -63, 53, 32, 18, -15, 77, 43, 38, -3, -1, -67, -31, -29, -6, -14, -41, 24, 50, 34, 69, -48}
, {56, 10, 99, 11, -31, -147, 35, -108, 39, 21, 5, -26, -13, -61, 8, -106, 17, -35, -115, -41, 15, -117, -55, -198, -105, 30, -30, -38, -34, -38, 21, -56}
, {3, 16, -47, -7, -83, 61, -280, 60, -14, 100, 63, 87, 44, 33, 69, 67, 71, -28, 23, -61, 16, -80, -63, 3, 34, -180, -136, 101, 29, 65, 41, 74}
}
}
, {{{-85, -18, -14, -2, -104, 52, -1, -50, -86, 107, 37, 17, 41, -34, 61, 16, 153, -27, 24, -82, 22, -111, 53, 50, -4, -74, -70, 36, 53, 68, 3, -4}
, {-118, 3, -47, -13, 18, 2, -155, -7, -42, -24, 3, 20, -39, -60, 20, -74, -14, 16, -21, -85, 40, -151, -32, -22, 6, -66, -37, -11, -43, 43, -56, 27}
, {33, -16, -30, 6, -66, -25, -35, 7, -14, -42, 4, -62, -145, -40, -62, 72, -48, -53, 60, -38, 20, -28, -34, -35, 53, 26, 19, -57, -39, -86, 52, -53}
}
, {{-20, -130, -17, -74, 8, 63, -1, -43, -101, 19, -76, 3, 60, -23, 33, -104, -67, 34, -99, 85, -2, 57, 34, 55, -172, 40, -51, -1, 28, 41, -19, -14}
, {2, -92, 26, 64, 28, 58, -15, 17, 6, -27, 38, -140, 68, 28, -1, -156, 23, 35, -71, 33, -9, 16, 50, 48, 22, -40, -50, 60, 0, 32, 39, 29}
, {-128, -23, 6, -20, 7, -13, -16, 13, 38, -72, -7, -188, -79, 29, -101, -123, -60, 31, -50, 33, -35, -18, 23, 67, 78, -59, 61, 42, -33, -48, 7, -88}
}
, {{39, 33, 18, -57, -101, 27, 61, -198, -118, -62, -23, 36, -77, 21, -46, -1, 50, -7, 31, -12, 33, 39, 27, -96, -84, 42, 73, -25, -2, 26, 5, -35}
, {-60, -120, 19, -102, -39, -17, 47, -25, -215, 22, -9, -47, -34, 16, 43, -50, 53, 59, 36, 45, 47, -17, 7, 17, -44, -9, 48, 49, 47, 83, 57, -57}
, {-47, -42, -13, 17, 62, -16, 12, -6, 45, 60, 44, -67, -10, 2, 8, -19, 11, 29, -44, -5, 41, -53, -48, -8, 13, -22, 22, 20, -5, 1, -39, -69}
}
}
, {{{71, -60, 15, 32, 91, -63, 2, -33, -48, 5, -5, -18, -137, 18, -79, 67, -56, 82, 23, -20, 24, -62, 30, 49, -50, 73, 63, -4, 10, -19, 0, -12}
, {-13, -38, -38, -84, -190, 18, 27, -42, -77, -62, -24, -7, 35, -90, 25, -22, -63, -132, -1, 67, -33, 55, -54, -176, -66, 32, -54, -6, 15, -4, -4, 44}
, {9, -69, -61, -105, -149, 109, 48, 23, -125, 4, -9, 129, -49, -38, -37, 41, 83, 34, 83, 48, 98, 100, 40, -33, 8, 29, -128, 20, 74, 65, -26, -12}
}
, {{-37, -74, 20, 31, 78, -15, -37, 18, 44, 33, 13, -12, 54, 57, 63, -70, 33, 1, 107, -3, -39, -58, 47, 7, 5, -40, 48, 51, -7, 48, 14, 25}
, {4, 7, 64, 70, 49, -109, -133, 25, 63, -30, 15, 49, -29, -3, -107, -54, -51, 69, -79, -103, -10, -258, -17, 102, 9, 65, 45, 3, -11, -34, 64, -89}
, {73, -49, -21, -33, -121, -128, -12, -84, 6, -157, -202, -37, 22, -215, -73, -45, -115, -149, -86, -5, -60, -16, -150, -92, -76, -2, 10, -218, -62, -36, -36, -123}
}
, {{-70, -43, 21, -14, -52, 57, 35, -14, 22, -16, -2, -47, -10, -65, 14, -167, 24, -38, -62, -10, 18, -55, -6, 23, 50, 71, -12, 15, 18, -11, 78, -37}
, {-94, 45, -46, 5, 68, -130, -116, 36, 70, -14, 36, -71, 50, 29, 18, 1, -12, 8, -54, -4, -19, -108, -80, -16, 29, -134, -91, 67, 15, -8, -83, 60}
, {-65, 3, 50, 72, 75, -157, -76, 77, 61, -98, 77, -4, 34, 63, -101, -60, -115, -62, -67, -286, -88, -162, -98, 50, 47, -41, -46, 79, -67, -64, -48, 80}
}
}
, {{{-51, 11, -41, -10, -16, -10, -12, 96, 47, -52, -25, -156, 66, -36, -126, 5, -25, -17, -42, -53, 39, 13, 35, 18, 9, -94, -31, -16, -13, -32, 21, -2}
, {9, 22, -83, -35, -55, -9, 36, 8, -111, -11, 16, 85, -44, 29, 27, 13, -44, -3, -129, -1, 36, 5, 32, 48, 80, -19, -100, 17, 59, 20, -52, 45}
, {-31, 31, 45, 41, 84, -91, 36, 65, 57, 5, -10, -46, 47, 3, -33, -14, -19, -8, -12, 19, 36, -77, -5, -3, -14, 24, 31, 4, 28, -18, 35, 28}
}
, {{-45, 33, 33, 61, 58, 45, 102, 61, 60, -155, -17, -193, 25, 47, -131, -101, -38, -20, -15, 40, -11, 38, 17, -21, 98, 20, 11, -44, 3, -60, -37, -107}
, {-29, -37, -21, -100, -47, -76, -13, 102, -146, 80, -39, -35, 21, 5, 51, -99, 25, 29, -90, 26, 18, 29, 39, 23, -7, 30, 18, -12, 11, 28, 35, 1}
, {13, 1, 48, -7, 11, -72, -135, 26, -7, 99, 25, 122, 17, 71, -9, 23, 53, 1, -20, -108, 44, -150, 36, 25, 48, -298, -76, 38, -10, -10, -5, 124}
}
, {{25, 31, -45, -64, 40, 48, 73, -57, 56, -181, -76, -94, -64, -53, -55, 56, -107, 49, 40, 38, -7, 112, 64, -19, -6, -22, -20, -102, -113, -71, -26, -60}
, {-40, 6, -228, 47, -167, -126, -118, 49, -1, 54, 30, 31, 61, -40, 13, -111, -55, -71, -90, 16, 49, -144, -61, -26, 56, -88, -116, -18, -63, 11, -120, 46}
, {47, 53, -74, 87, -60, 9, -258, 79, -17, 30, -10, 10, -106, -126, -114, 1, -166, 96, -5, -111, -13, -163, -194, 19, 32, -291, -286, -68, -100, -42, -224, 54}
}
}
, {{{90, 86, -6, -45, 24, -24, -50, -22, 62, 22, 51, 22, 40, -37, 46, 53, -39, -63, -4, -9, -92, -24, -120, -95, 43, 8, -32, 2, -116, -4, -2, 14}
, {83, 28, 21, 14, -13, 91, 34, 99, 44, 35, 26, 43, -1, 40, -18, 43, 30, 23, -3, 26, -39, 86, -3, -8, 96, 16, -47, 11, -11, -24, -30, -26}
, {-24, 14, -6, 11, 10, 25, -7, 4, -45, 51, -40, 76, -75, -35, -47, 12, 15, -62, -5, -29, 61, -22, 28, -70, 54, 31, 35, -98, -24, -29, -17, -23}
}
, {{-3, 71, 1, 33, 46, -2, -11, 32, 104, 13, 15, 29, 19, 26, 22, -22, 13, 8, -5, -1, 2, -52, 25, 75, 46, -80, -3, 14, 16, 4, -10, 21}
, {98, -35, -32, -81, 18, 67, 5, -16, -97, -37, -1, -85, -6, 28, -46, -43, -21, 50, -12, 54, 58, 80, -22, 26, -23, 26, 2, 7, -8, -2, 0, 30}
, {-8, -24, 14, -136, 10, 43, 65, -11, -106, 21, 3, -46, 23, 50, -10, 17, 47, 46, 47, 89, 81, 115, 40, 7, -102, 193, 46, -53, 26, 79, 138, -3}
}
, {{0, -6, -9, 4, -4, -45, 14, 1, 69, 28, -13, 44, -30, 41, 33, 14, -31, -40, -57, -67, -82, -100, -66, -37, 44, 46, 43, 16, -71, 3, 11, 13}
, {76, -15, 40, -44, 37, -49, -16, -58, 39, -38, -76, -63, -140, -28, -216, -29, -163, -38, -31, -34, -207, 39, 70, 37, -26, 67, 99, -40, -132, -94, 4, -48}
, {58, -46, -76, 1, -75, -53, -10, -242, 35, -281, -267, -121, -67, -260, -260, -46, -342, -2, -74, 15, -243, -11, -153, -172, -301, -33, -43, -357, -378, -359, -86, -199}
}
}
, {{{18, -166, -101, -36, -39, -24, 29, 70, -117, 28, -12, -10, -97, -64, -21, -43, 11, 1, -70, -25, -1, -13, 11, 12, 27, 27, -61, -14, 9, -48, -5, -131}
, {38, 24, -28, 26, 4, -34, -16, -25, 113, -110, -57, 43, -43, -37, -45, 72, 2, -81, -14, 3, -64, 59, 38, -45, -118, -28, -14, -46, -57, -113, 22, -23}
, {-79, 54, -26, 35, -53, 3, 81, 24, -38, -5, 33, -11, 32, -18, 26, -91, 53, -29, -12, 29, 87, -76, 61, 14, -21, 55, -25, -1, 26, 6, -11, -48}
}
, {{-57, -60, -10, 9, 23, -55, -86, 38, 2, 72, 56, 14, -87, 47, -37, -5, 8, -48, -11, -75, 59, -109, -12, 8, 3, -9, -3, -12, 5, -26, 1, -26}
, {-52, -100, -15, 25, 58, 18, -24, -8, 101, 43, 11, -53, 15, 22, -2, -105, -1, 18, -42, -8, -8, 77, 26, 30, -76, -17, 23, 42, 18, 52, 59, 53}
, {-131, -52, 35, 9, -21, 115, 70, -118, -28, 6, -22, -38, 49, 5, 13, -36, 0, 27, 26, -30, 49, 107, -8, 10, -14, 63, 14, -34, 79, 53, 14, -82}
}
, {{52, -7, 41, 26, 67, 51, 108, -13, 105, -33, -5, -221, -35, 59, -156, -11, -24, 83, -35, -22, -53, 103, -7, -15, 0, -114, -45, 3, -12, -16, -47, -170}
, {51, -172, 30, -23, -18, 27, 92, -113, -31, -149, -62, -92, 43, -88, -49, 34, -34, 53, 1, 51, 15, 77, 52, -37, -105, 112, 30, -79, -20, 13, -34, -36}
, {75, -88, -33, -114, -40, 19, 138, -131, -24, -136, -44, 46, -24, -41, -3, -44, -57, 48, -10, 42, -11, 50, -25, -10, -178, 61, 59, -45, -19, 18, -8, 21}
}
}
, {{{9, 22, -134, -33, -268, -12, 40, 1, -9, -46, 36, -19, -155, -139, -207, 84, -126, -53, 103, -43, 45, 42, -4, -128, 81, -51, -106, -41, -69, -155, -37, -92}
, {71, -3, -168, 12, -216, 56, 44, -52, -31, -125, -215, -115, -40, -246, -122, 48, -81, -11, -69, -24, -34, 79, 41, -56, -149, -39, -96, -216, -76, -94, -80, -41}
, {-10, -149, -65, 32, -115, 22, 56, 83, -68, -11, -105, -75, 5, -80, -41, 106, -126, -3, -52, -58, -63, 66, -96, -117, 10, -27, -41, -41, -118, -162, -46, -58}
}
, {{-52, 29, 37, 99, 39, 0, -193, 23, 94, -29, 111, 62, -48, 23, 46, -36, -8, 46, -52, -145, 54, -62, 47, -19, 7, -64, -33, 15, 30, 33, -26, 20}
, {-226, -188, 21, 41, -16, 1, -80, -64, -46, -5, -28, -3, -70, 27, 27, -139, 20, 19, 25, -26, 54, 4, 61, 18, -78, 8, 25, 28, 53, 34, 14, 27}
, {-100, -74, 21, 5, 36, -4, 47, -21, -47, -51, -14, -86, -116, 1, -65, 19, 6, 19, 89, 73, 49, 41, 38, 35, -70, 115, 72, 37, 63, 24, 64, -132}
}
, {{55, 22, -76, 18, -61, 4, 14, 33, -24, 37, 29, -22, 56, -3, 43, 1, -46, -93, -48, -16, -46, -24, -28, 5, 34, -96, -151, 23, -45, 26, -51, -29}
, {22, 22, 43, 10, 53, -1, -93, -1, -31, 30, 1, -2, 51, -4, -15, -63, -32, -22, 2, -59, -5, -59, 73, 31, 52, -109, -93, 31, -9, 11, -39, 37}
, {-12, 14, -7, 38, -41, 24, -43, 34, 32, -28, 45, 11, -63, 8, 14, 10, -23, 20, 31, -70, 14, -12, 43, -33, 41, 8, -63, -16, -26, -28, -23, 9}
}
}
, {{{79, 55, 58, 16, 47, -104, -96, 36, 133, -22, 51, -76, 65, 23, -69, -55, -105, 23, -62, -27, -41, -107, -99, 32, 5, -54, -4, -2, -30, -64, -35, 94}
, {-80, -97, 71, 58, 32, -86, 44, 15, -32, -68, 22, 5, -2, 67, -15, -106, 19, -24, -152, -23, 27, -20, 73, -13, 7, -61, -41, 30, 50, 30, 37, -8}
, {-124, 68, 39, -5, 15, 82, -21, 37, -6, 15, -7, 43, 87, 61, 148, 3, 31, 51, 57, 31, 58, 31, 68, 5, -2, -17, 22, 27, 44, 35, 67, 27}
}
, {{33, -143, 42, 30, 88, 24, 12, 30, 73, -74, -9, -53, 37, 47, -58, -54, -86, -10, -21, -5, 30, 28, -18, 56, -10, 23, 48, -26, 0, -7, 6, -17}
, {42, -53, 45, -28, -123, 97, 35, 52, -59, -86, -55, -105, 45, -13, 2, 33, 13, 70, 34, 43, 31, 69, 59, 15, 11, 127, 70, -77, -7, 17, 61, -50}
, {94, -72, -118, -184, -137, 42, -25, -49, -78, -160, -69, 28, -14, -180, -62, 57, -90, 31, 76, 37, -7, 60, 18, -38, -87, -118, -186, -140, -12, -66, -202, -52}
}
, {{-73, -56, 4, 32, -28, -11, 43, -18, 5, -9, 13, 19, -80, -22, -73, -147, -21, -84, -19, 28, -69, -3, -7, -78, -29, -3, 14, 7, -29, -46, 24, -40}
, {13, 108, 3, 0, 36, 4, 157, -14, 42, -62, 36, 82, -42, -24, -28, 46, 49, -55, -6, 15, 1, -8, 174, -34, 21, 57, 27, 32, 15, -51, -40, -17}
, {48, 11, 29, 5, 74, -57, 96, -24, 10, -89, -62, 23, -101, -2, -66, -57, -118, -43, -210, -166, -167, -15, 86, 47, -139, -58, 1, -15, -189, -73, -46, 43}
}
}
, {{{28, 61, 4, -87, -102, -48, -11, 11, 16, 24, -6, 55, -13, -26, 9, 41, 132, 10, -8, -30, 21, -10, -110, 3, -6, -30, -57, -15, 4, 30, -104, 61}
, {7, 117, -45, 7, -39, -45, 10, 51, 73, 50, -19, 94, 56, 5, 31, 64, 18, -62, 27, 4, -51, 15, -31, 57, 29, -50, -61, 37, -37, 72, 2, 2}
, {15, 54, -87, -25, -32, -49, 49, 7, 14, -1, 42, 53, -155, -146, -100, 5, -11, -89, -20, -5, 1, -64, -118, -186, 50, -142, -70, -18, -90, -144, -140, -43}
}
, {{7, 121, -1, 83, -1, -74, -38, 31, 73, 41, 43, -9, -25, 4, 29, 115, 37, -75, -68, -57, -56, -3, -84, -104, 77, 41, -26, 45, -33, 4, -1, 19}
, {-10, 138, -57, 8, -48, -75, -118, 10, -33, 59, 6, 109, -80, -3, 71, 103, 39, -110, 19, -192, -55, -82, -136, -160, 25, -101, -95, 32, -43, -50, -16, 45}
, {-25, -9, -52, -46, -18, -52, -106, -64, -34, 98, -76, 39, -93, -39, 38, 82, -69, -309, 36, -298, -91, -123, -408, -195, 80, 39, -17, -35, -174, -19, 23, 9}
}
, {{-76, 51, 73, 13, -21, -39, -85, -11, 17, 2, 0, 83, 5, 79, 46, 72, 14, 14, 1, -73, -37, -28, -95, 42, 41, -23, -6, 42, 8, 18, 15, 17}
, {-55, -18, 78, 89, -14, 26, -66, -17, 14, 40, -29, 49, -71, 40, -5, -78, 25, 40, -99, -139, -63, -109, 1, 14, 5, 20, 22, -7, -38, 6, 78, 5}
, {-99, -93, 53, 91, 25, -106, -84, -52, -2, 45, 23, -28, -21, 51, 115, -207, 40, 31, 23, -146, -13, -180, 26, 34, 22, 50, 88, 49, 73, 85, 159, -20}
}
}
, {{{-67, -10, 17, -9, -7, -22, -2, -15, 41, 8, 51, -101, -39, -10, -62, -85, -61, 64, -15, -30, 11, -62, 16, -32, 57, -58, -94, -23, -54, -52, -43, -85}
, {-159, -11, 42, 2, 38, -38, 8, -9, 61, -46, -43, -61, -20, 29, -23, -34, 21, -47, -60, -79, -24, -62, -13, 17, -5, -18, 37, 17, -37, -28, 26, 26}
, {11, -73, 64, 16, -20, -53, -87, -30, 16, -57, -19, -37, -1, -37, 44, 6, 38, -21, -10, -9, 43, -20, 71, -33, -84, 31, 56, -29, 54, 36, 66, -2}
}
, {{-22, -201, 37, -103, 17, 44, 34, 43, -46, -38, 19, -53, 17, 11, 30, -80, 36, 48, -72, 54, 76, 27, 66, 89, -68, 109, 2, 36, 62, 80, 52, -9}
, {-146, 10, 29, -24, -47, -22, -208, 9, -116, 68, 52, 35, -161, 27, 38, 99, 72, 82, 54, -116, 78, -141, -21, -38, 9, -26, -44, 36, 85, 95, 49, 29}
, {52, 64, -83, -44, 9, 42, -12, -67, 16, 90, -64, -21, -119, -47, -90, 33, -23, 20, 62, 53, 23, 24, 17, 14, 15, -219, -181, -66, 5, -16, -164, -12}
}
, {{-54, -64, -91, 7, -50, -155, -88, 7, -22, 34, -10, 46, 5, -110, 30, 4, -30, -43, 12, 24, -78, -112, -124, -47, -44, -77, -115, -30, -16, -27, -128, 35}
, {53, -15, -69, 44, 29, 55, -90, 57, 66, 8, -5, -62, 34, -40, -1, -51, -81, -47, -24, 8, -3, -13, -36, 22, 13, -244, -197, 54, -40, -28, -83, -9}
, {114, 0, 35, 5, -12, 30, 110, -156, 47, -217, -52, -62, -12, -56, -174, -55, 43, -35, -116, 81, -10, 131, 107, -47, -153, 1, 88, -106, -32, 7, 32, -39}
}
}
, {{{-101, -153, 26, 36, 26, 47, -5, -1, -44, 25, 15, -32, -47, -58, 60, -28, 79, -54, 25, -48, 7, -27, -38, 0, -38, 7, -21, 33, 52, 25, 63, -45}
, {-41, -15, -51, -1, -26, -84, -168, 28, -44, -20, -12, 11, 33, 24, -40, -9, -53, -72, -40, -19, -30, -74, -83, -2, -36, -16, 22, -44, -28, -59, -14, 15}
, {-60, 25, -33, 90, -45, -40, -23, 52, 17, 80, -2, 6, -122, -23, -81, 49, 6, -65, 51, -29, 57, 6, 54, 0, 62, -46, -82, -50, 2, -92, 3, 57}
}
, {{6, -96, 23, 11, 97, 14, -77, -57, 0, 22, 61, 8, 5, 66, 25, -97, -15, -7, -12, 2, 20, -21, -51, 31, 26, -80, 29, 11, 16, 10, 6, 30}
, {-77, 9, 78, 16, -19, 5, -59, 25, 83, -23, 8, -220, 24, 12, -24, -32, 17, -2, -65, 2, -13, -70, 11, 58, -19, -64, 0, 38, -46, 20, 39, 24}
, {-132, -18, 24, -45, 30, 60, 162, -16, -34, -4, 14, -174, 11, 42, -44, -69, 23, 53, -61, 14, 8, -6, 100, 0, 31, -72, -51, 1, 5, 60, 50, -76}
}
, {{24, -6, -28, 21, -123, 32, 3, -1, -27, 5, 13, -88, -85, -25, -16, 32, 55, -41, 52, -62, -2, 15, -126, -89, -38, -65, -55, -37, 2, -20, -41, -12}
, {-9, -70, -4, -134, 47, 96, -10, 17, -115, -59, -33, -123, 12, -26, 2, -14, -54, 35, 39, 44, 21, 29, 4, 71, -46, -30, -67, -40, 12, 32, -116, -83}
, {18, -65, 77, -27, -4, 72, 32, -19, -45, -18, -38, -12, -24, -49, -7, -56, -10, 60, 75, 66, 115, 73, 24, 17, -59, 54, 106, -46, 17, 6, 53, -67}
}
}
, {{{37, -31, -9, 14, 5, -30, -16, 32, 6, 20, -7, -53, -8, -12, -4, 62, 16, 86, 25, -15, -37, 23, -1, 46, 2, 32, 17, 3, -5, 60, -5, -8}
, {-47, -21, -55, -88, -63, -78, -24, -12, -94, -26, 32, -18, 82, -13, 23, 0, -4, -49, -47, 78, -29, -125, -58, -35, 1, 2, -36, -39, -26, -16, -47, 1}
, {-19, 34, 10, -47, -29, 40, -135, 85, 40, 87, -2, 116, 57, -4, 96, 39, 69, 9, 38, -46, 63, -29, 45, -6, 75, -127, -26, 30, 5, 34, -44, 176}
}
, {{-105, -14, -7, -32, -11, -5, 22, -77, 8, -13, -15, 61, 4, 5, 26, -24, 17, -3, 55, 47, 16, -44, 34, 68, 42, 32, 43, 23, 27, 1, 30, -21}
, {54, 23, 89, 51, 10, -11, 43, -145, 6, -43, -31, -17, -119, -17, -44, 42, 31, -6, 14, -42, -71, -20, 32, -6, -69, 100, 86, -53, -22, -89, 97, -47}
, {29, -20, 0, -52, -27, -22, 72, -56, -16, -146, -135, -145, 30, -165, -190, -40, -172, -107, -85, 29, -134, 68, -255, -27, 7, -10, -29, -159, -164, -213, -34, -136}
}
, {{-78, 18, -33, 32, -17, -36, -83, 49, 42, 52, 16, 5, -33, 54, -9, -58, 18, 13, -13, -62, 5, -86, -67, -13, 16, -101, -131, 2, 2, 17, -25, 56}
, {-163, -187, 7, 58, 96, -92, -140, 41, 63, -22, 4, -78, -2, 56, 20, -186, 17, 49, -86, -61, 14, -156, -2, 65, -8, 33, -36, 52, 50, 56, 34, -56}
, {-147, -36, 41, 73, 63, 7, -160, 48, 60, -88, 53, -150, -5, 165, -83, -166, -4, 64, -56, 6, -2, -172, -12, 134, 15, -58, -41, 105, 27, 96, 24, -9}
}
}
, {{{61, 36, -42, -69, -94, -13, -36, -64, 1, -3, 13, 46, -52, -43, -41, 56, 27, -43, 22, 35, 38, 3, -44, 29, 59, -11, -26, 27, 50, -23, 18, -7}
, {36, 124, -96, -16, -21, 35, -15, 66, 27, 124, 13, 48, 33, 3, -13, 86, 36, -28, 44, -40, 24, 0, -1, 6, 125, -29, 30, 24, -21, -5, -66, 51}
, {-8, 52, -45, -51, 33, 6, 63, 42, -20, 15, -12, 108, 5, -24, 43, 24, 16, -29, 29, 2, 2, -30, -31, 0, -9, 102, 30, -27, -15, 22, 18, -3}
}
, {{32, -25, 25, 38, 28, 8, 18, -36, 89, 7, 101, -125, -132, -13, -45, 72, 125, -106, 13, -54, 2, 14, -153, -151, 88, -48, 1, 9, -49, -96, 24, -34}
, {44, -30, -103, -147, -134, 6, 5, -190, -57, -240, -210, 12, 10, -152, -90, 34, -162, -162, -23, -21, -132, 13, -114, -167, -83, -4, -86, -174, -144, -91, -47, 15}
, {93, -129, 4, -28, -23, 56, 54, -20, -93, -15, -116, 3, 9, -27, -32, 36, -18, 11, 71, 18, 0, 75, 36, 51, -69, 15, 77, -43, 28, 35, -19, 48}
}
, {{-16, 62, -21, 99, 7, 85, 50, -14, 14, -53, 43, 2, -98, 7, -132, 118, -22, 93, 62, 11, 13, 45, -11, 9, 116, 22, 2, -43, 19, -64, -3, -124}
, {-29, -87, -18, 17, 17, 73, -64, -99, 33, -58, -130, -57, -58, -9, -38, -31, 32, -5, -12, -12, -37, 73, 95, 39, -166, 53, 51, -28, -29, 30, -1, -14}
, {75, -78, 61, -5, 62, 12, 180, -133, 71, -130, -80, 7, 28, -54, -27, 73, -21, 35, 17, 78, 7, 46, 126, -52, -33, 155, 93, -73, -7, -14, 5, -8}
}
}
, {{{70, 80, -4, 25, -92, -12, 19, -57, -22, 16, 5, 32, -87, -1, -102, 44, 2, 17, -13, 19, -78, 35, -149, -80, -43, 23, -77, -39, 25, -8, -25, -45}
, {16, -51, -18, -17, -13, 31, 21, -48, -51, 50, 13, 43, -23, 4, 33, 19, 88, -17, 23, -18, -40, -3, -25, -6, 4, 27, 5, 13, 49, 41, 30, -25}
, {-79, 3, 30, -1, 15, 13, -40, 30, 22, -18, 19, -7, 52, 63, 70, -54, 27, -1, -62, 8, -16, -89, 51, 23, -18, 35, -7, 37, -2, 11, 7, 52}
}
, {{-59, -53, -65, 6, -112, -44, -41, 57, -88, 43, 85, 102, -7, -8, 63, -19, -46, -19, -10, -64, 7, -75, -49, 14, -42, 35, -64, 9, 0, -12, 8, 73}
, {-3, -74, 15, -45, -45, -3, 28, -153, -131, -43, -56, -158, -17, -47, -113, -19, 9, -6, 31, 13, 20, 13, -14, 19, -36, -127, -37, -131, 58, -81, -100, -126}
, {11, -11, 44, -72, 2, 34, 72, 33, -30, 12, -28, -85, 69, 9, 10, 60, 1, 66, 45, 18, -1, 63, 88, 25, -22, -8, -29, -15, 33, 46, -108, -8}
}
, {{-25, 16, -16, 37, 35, 81, 85, 66, -37, -9, -53, 4, 113, 12, 16, -61, -67, -24, -6, 98, 8, 37, -44, -5, 44, 6, -54, -2, 2, 43, -66, 51}
, {-131, -87, 14, -15, 38, -4, 5, -3, 39, -70, 17, -4, -71, 57, -47, -113, 12, 18, -63, 27, 82, 58, 51, 56, -45, -3, 83, 91, 44, 28, 45, -99}
, {-94, -40, 47, -26, 28, 29, -72, -82, -18, 1, -8, -31, -203, -4, -51, -49, 56, 45, -7, -101, 32, 48, 22, -20, -42, 67, 52, 13, 48, -7, 41, -126}
}
}
, {{{45, -38, 37, 56, -5, 34, 30, 10, -28, 27, 47, -11, 17, -33, -23, 49, 46, 8, 21, 27, 35, 3, -41, -12, 36, 32, 10, -10, 39, 49, 25, 27}
, {-145, -16, 29, 51, 16, -143, -31, 26, 45, -25, 3, -13, -26, 71, -18, 4, 12, -3, 15, -84, -21, -158, -25, 34, 17, -6, -41, -11, -70, -3, 24, -6}
, {55, -15, 7, 62, -83, 75, 16, 35, -13, 17, 30, -129, -99, -22, -146, 14, -29, -65, 35, -73, 15, 40, -102, -48, 13, -23, -28, -66, -6, -59, 11, -62}
}
, {{-67, -46, 51, 31, 122, 58, -27, 49, 9, -37, -23, -64, 22, 1, 34, -80, -33, 47, -64, 48, 6, -24, 11, 62, -68, -34, 0, 36, 23, 12, -32, -99}
, {-10, -118, 15, -57, 34, 18, -51, -7, -48, 23, 26, -37, 11, 31, -11, -46, 8, 14, -53, -3, 66, -19, 44, 32, -4, -90, -43, 61, 42, 27, 13, 58}
, {-40, -15, 46, -113, 11, 7, 68, -3, -48, 45, 33, -81, -14, 67, 12, 15, 46, 26, 62, 74, 89, 63, 45, 6, -21, -61, 22, 29, 76, 36, 44, -24}
}
, {{-13, 65, -159, -31, -119, -116, 43, 16, -24, -7, -11, 5, -2, -121, 23, -47, -6, -149, 26, 25, 4, 1, -168, -200, 58, -94, -109, -66, -64, -34, -101, -43}
, {-32, 63, -177, -45, -196, -1, 7, 0, -23, -14, 5, 53, -33, -182, 24, 35, -28, -77, 41, 4, -70, -3, -71, -104, 41, -66, -11, -30, -127, -95, -26, -10}
, {-22, 35, -94, -83, -137, -71, -20, -54, -31, 66, -25, 131, -76, -140, 23, 14, 39, -61, 46, -28, 14, -107, -138, -105, -24, 38, -31, -57, 5, -40, -34, 7}
}
}
, {{{37, 84, -110, -123, -95, 19, 23, 11, 44, -23, 9, 47, -57, -162, -41, 4, 82, 32, -57, 2, -34, -40, -169, -46, -59, 41, -87, -76, -46, -82, -91, -39}
, {-86, -13, -86, -30, 53, -21, -19, 48, -60, 75, 5, -4, 24, 8, 72, -66, 3, -63, -30, 47, 17, -21, -198, -84, 54, 20, 7, 32, -28, -13, 16, 34}
, {67, -22, 33, 37, 67, -83, -18, -40, -27, -19, -24, 7, 2, -38, 6, -30, -4, 4, 9, 30, 33, -29, -7, -18, -26, -53, -39, -3, 72, -11, 29, 79}
}
, {{-6, 125, -39, -80, -2, -28, -10, -141, 11, -47, -81, 56, -41, -50, 26, 47, -40, -10, -151, -37, -89, -89, 44, -37, -28, 4, 28, -110, -38, 10, -10, 66}
, {-336, 20, 0, -1, -80, -155, -149, 24, -284, 47, 51, 94, -212, 43, 51, -88, 71, -34, 26, -202, 47, -86, -98, -70, 24, 29, 54, 48, -26, 28, 61, -59}
, {-39, -42, 58, -67, 41, 69, 52, -90, -54, -15, 15, -199, -157, 17, -29, -58, 14, 26, -24, 102, 63, 51, 57, 92, -62, 5, -24, 25, 49, 5, 2, -244}
}
, {{18, 2, -5, -11, -52, -139, 56, 46, -45, 89, 19, 73, 49, 6, 39, -108, -10, 47, -96, 61, 10, -14, 39, 18, -43, 20, 5, 8, -28, 10, 3, 34}
, {-151, -10, -36, 144, -71, 8, -242, 47, -4, 48, 40, -39, -25, 13, 60, -31, -20, -4, 17, -59, 9, -89, -9, -9, 70, -227, -120, 11, 31, -27, 7, 18}
, {6, -46, 19, -117, 6, 54, 34, -115, -23, -149, -48, -160, -48, -46, -7, 7, 34, -111, -18, 65, 42, 46, 114, 7, -126, 34, 87, -78, 72, 43, 108, -76}
}
}
, {{{-153, -103, 67, 63, 26, 6, -18, 42, -106, 60, 39, 2, 51, 36, 4, -65, -39, -50, -1, -14, 7, -68, 73, 47, 46, -66, 69, 12, -18, 22, 26, 24}
, {-159, -25, 40, -1, 58, -103, -153, -12, -25, 38, 58, -24, -79, 40, 10, 5, 33, -55, 51, -169, -17, -107, 30, -45, 6, -2, 1, 24, 25, 5, 73, 37}
, {36, 72, 19, 7, -56, -41, -136, -150, 52, -90, -12, 1, -157, -67, -16, 8, -3, -56, 5, -85, 14, -60, 34, -41, -7, -61, -17, -74, -51, -47, -34, -42}
}
, {{-43, -2, -44, -155, -10, -39, -4, -53, -28, 41, -65, 4, 39, -76, 62, -57, 17, 12, -78, 77, 17, -49, -3, -42, -91, -100, 16, 10, -22, 56, -30, -21}
, {-146, 122, 9, -20, -39, -186, -411, 20, 28, 45, 68, 7, -43, 22, 4, -5, 9, -54, -10, -182, -3, -273, -135, -27, 59, -196, -167, 1, -2, 33, -99, 80}
, {-16, 79, 25, -29, -12, -28, -58, -105, 3, -11, -29, 57, 3, -67, -137, 16, -22, -62, 40, -20, -13, -59, 0, -91, -32, 2, -66, -115, -85, -100, -12, 51}
}
, {{0, -98, -43, -133, -231, -21, -57, 2, -156, 76, -21, -2, 19, -7, -9, 32, 75, 28, 84, 33, -1, -5, -26, 61, -60, -110, -105, 16, 41, 18, -98, -18}
, {3, -92, 45, 38, 74, 30, 2, 30, -30, -55, -43, -101, 15, 10, 20, -90, -94, 56, -95, 34, 33, 38, -48, 23, -65, -176, -109, 14, -10, 39, 36, -6}
, {-119, -65, 46, 64, 50, 8, -17, 3, 14, -65, -17, -149, -123, 47, -31, -109, 50, 125, 13, 6, 23, 8, 73, 15, -61, 71, 65, 6, 21, 39, 58, -102}
}
}
, {{{35, -35, -26, -52, -8, 39, 22, -39, -90, -6, -91, -29, 10, 25, -36, 26, -50, 16, 42, -34, -69, -14, -4, -5, -61, 49, -26, -43, -10, 11, -20, 3}
, {-50, 25, -60, 13, -93, 60, 42, 26, -43, 57, 14, 57, -9, -4, 46, 65, 68, -4, 133, 40, 1, -2, -29, -56, 81, 45, 7, -1, -16, 14, -1, 29}
, {84, 13, -1, 10, -11, 65, -5, -8, -21, 8, 24, 79, -83, 40, -12, 69, 40, 11, 60, -33, 22, -4, -59, 2, 84, -22, -12, 43, 47, -16, 15, -18}
}
, {{108, -23, -49, 1, -22, -56, 28, -11, -96, 12, 25, 37, 16, -87, 51, -4, 24, -41, -127, 26, -51, -88, -29, -31, -44, 35, 27, 20, -15, -36, -54, -21}
, {32, -51, -69, -56, -41, -1, -21, 41, -34, 57, 11, 60, 15, 6, 50, -10, -15, -26, 18, -16, 1, 37, -15, 5, 21, 18, -52, -3, -16, -4, -27, 59}
, {56, 9, -8, -80, -28, 58, -32, 17, -7, 74, 45, 1, 6, 21, 92, 49, 80, -7, 43, -35, 53, 67, -13, 6, 36, 87, 52, 68, -5, 42, 76, 41}
}
, {{-54, 72, -107, 54, 2, -93, -65, 26, 11, 54, 23, 102, -34, -81, 103, 104, -23, -192, 47, -140, -20, -127, -273, -120, 80, 35, 16, 44, -97, -98, -34, -46}
, {47, -32, -82, -96, -88, 101, 110, -208, -14, -231, -156, 45, -33, -201, -158, 62, -165, -36, -6, 37, -59, 156, -3, -147, -33, 121, 196, -237, -89, -244, 34, -72}
, {27, -104, -57, -22, -21, 21, 35, -190, -69, -89, -235, 77, -125, -102, -73, 44, -91, -83, -39, -33, -106, 31, -113, -219, -229, -8, 66, -61, -214, -155, 44, -148}
}
}
, {{{-68, 35, 67, 35, -11, -11, 12, 56, 5, 19, 68, -22, 58, 11, -121, 44, -1, -2, 9, -63, -62, -37, -77, 22, 21, 7, -21, -17, 23, -35, -34, -27}
, {89, 1, 17, -49, 23, 93, 41, -196, 53, -185, -92, -126, -76, -52, -148, 13, -100, 11, -37, 8, -48, 58, 27, -79, -225, 75, 37, -151, -19, -16, 7, -79}
, {46, -185, -44, -38, -71, 80, 67, -143, -34, -307, -171, -132, -78, -117, -207, -78, -138, 72, -29, 8, -54, 10, 78, -33, -303, 58, 83, -240, -47, -65, 9, -235}
}
, {{92, -194, 51, -12, 36, 24, 50, 17, 14, -15, -25, -40, 9, 14, -69, 69, -37, 41, 1, 39, -30, 71, 35, -15, 9, 10, 22, -1, -16, -98, 20, -94}
, {55, -92, 0, -16, 24, 5, 76, -60, 39, -24, -92, 28, 18, 0, 6, -52, -34, -25, -111, 63, -24, 54, 60, 10, -156, 87, 95, 16, -6, -17, 63, 12}
, {2, -21, 9, 6, 33, 13, 128, 29, 1, -38, -12, -136, -1, -33, -132, -175, -1, -12, -147, 21, -12, 73, 104, 32, -81, 99, 50, -23, 0, 18, 10, -174}
}
, {{-101, 87, -49, 6, -43, 31, -19, 52, 51, 65, 47, -88, -38, 8, 9, 71, -41, 2, 12, -39, -6, -12, 4, 33, 38, 31, 72, 36, 21, -15, 24, -138}
, {-54, -12, -39, -43, 27, 84, 30, -26, 21, 58, 10, 33, 3, 57, -21, -26, 1, -3, 17, -27, -31, 48, 4, -11, 25, 14, 1, 54, 11, -3, 22, 12}
, {-11, 41, 20, -27, -87, 39, 40, -34, 18, -10, 44, -54, 2, 61, 11, 38, -8, -44, 32, 5, 35, 30, 36, -72, 36, 55, 56, 33, 14, 10, 3, 33}
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

typedef int16_t flatten_2_output_type[OUTPUT_DIM];

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

#define NUMBER_T int16_t
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

typedef int16_t dense_4_output_type[FC_UNITS];

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
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
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
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
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


const int16_t dense_4_bias[FC_UNITS] = {24, -65, 23, 75, 26, -22, -35, 78, 2, 5, -27, 19, 27, 68, 26, -38, 18, 85, 78, 55, -12, 79, 13, 32, -6, 2, 20, 17, 3, 34, -14, 80, -15, 6, -52, 65, 26, 15, 80, 101, -3, 4, 65, -3, 2, 33, 22, -16, 6, 37, 6, 25, -4, 48, -9, 71, 31, 15, 7, -16, 22, 13, 44, -52, 96, 74, -4, 12, 54, -57, -41, -18, -8, 65, 7, -7, 45, 52, -2, -21, 43, 78, 34, -24, -32, 87, 1, 10, 50, 15, 82, 67, 65, 11, -35, 27, 46, 4, -14, -1, -13, 64, 77, 32, -1, -1, 70, -9, 4, -31, 38, 35, -18, 26, 32, 56, 39, -36, 71, 14, -9, 53, -53, 6, 44, -5, 24, 17}
;

const int16_t dense_4_kernel[FC_UNITS][INPUT_SAMPLES] = {{240, 40, 89, 64, 33, -59, 6, 131, 47, -71, -25, -81, 23, 81, -102, 54, 18, 68, 4, -22, 121, -139, 42, 2, -30, 2, -97, 34, 63, 29, 83, 66, -4, -113, -20, -24, 8, -161, -49, 21, -122, -30, -44, -97, -62, 51, 37, 187, 15, -87, -143, -16, -154, 27, -31, -59, 54, -6, -77, -138, -11, 147, -94, -54, 1, -23, 122, 41, -133, -167, 82, 11, -41, 94, 35, -40, 139, -57, 21, -89, -44, -81, -76, 29, 51, -86, 101, 50, -6, 23, -46, 15, -23, 100, -80, -5, 35, 1, -4, -41, -41, -48, -42, 110, -66, -129, -80, 27, 115, 165, -146, -11, 12, -45, -71, -83, -140, 74, 49, -64, -73, -39, -85, -34, 74, 0, -89, -35, -48, 51, 61, 168, 91, -80, -49, 117, 48, -74, -100, -24, 14, -58, -57, 68, 47, -69, 116, -128, 15, -46, -2, -14, 80, -12, 48, 9, 50, 67, 20, -8, 56, 63, 94, 31, 66, -63, -42, -21, -10, -21, 2, -37, 28, -19, -40, -140, 111, -37, 7, 7, 3, -78, 215, 17, -138, -23, -26, 30, 51, 23, 26, -176, 17, -10, 51, -56, 38, 35, -50, 44, 26, -93, -48, 1, -9, -127, 59, 8, -32, 95, -22, -149, 78, -5, 33, -35, -42, 10, -32, -52, -13, 58, -2, 17, -2, 121, -25, -60, 10, 3, -87, -5, 70, -57, -187, -51, 6, -19, 19, 44, -34, 12, -8, 92, 26, -36, 34, -59, -35, 38, -65, 47, 9, -31, 21, -157}
, {318, 36, -105, 49, -103, 26, 5, 45, 153, -41, 35, -10, -83, -4, -50, 47, -48, 59, 2, -68, 65, -129, -23, -8, -32, -58, -105, 28, 88, 58, -6, 5, -45, 42, -137, 6, -102, 56, -17, 68, -73, 14, 9, -67, -60, -33, -4, 195, 66, 23, 52, 24, -48, 19, -176, -12, 48, 1, -13, 40, -10, 136, 29, 7, 20, -16, 34, 53, -10, 69, -6, 88, 102, 112, 24, -26, 22, -178, 3, -100, -2, -52, -21, -130, -11, 66, 96, -11, 58, -41, 146, 69, -79, 6, -2, 6, 5, -11, 43, -16, -15, -87, -6, 118, -190, -19, -19, -141, 86, 56, 15, 8, 148, -47, 8, -23, -29, 91, 45, -26, 14, 110, -37, 10, 0, -15, -70, 17, 68, -28, -72, 96, 42, -122, 26, 115, 8, 59, -83, -45, -185, 47, -60, 29, -66, -88, 5, -139, 19, 111, 23, -93, -37, -50, 79, -114, 34, 10, -48, -80, -65, -74, -22, 10, -94, 38, 0, 57, 51, -81, -37, -119, -88, 50, -116, -86, -43, -6, -16, 33, 11, -10, 64, -73, -155, -60, -91, 45, -132, -41, -34, 51, -29, -158, 92, 2, -5, -29, -97, 28, 54, -3, -57, 30, 9, -84, 83, -78, -22, -33, 11, -130, 18, -12, 5, 89, -1, -14, -42, 32, -146, -7, -63, 19, 118, -31, 53, -39, 15, 91, -67, 43, -50, -167, -49, 61, -2, 146, -49, -69, 23, 90, -77, -4, -77, -44, 41, 29, -103, -89, -80, 36, 44, 27, 87, -88}
, {100, -20, 5, -2, -41, -140, 39, 91, -94, -123, 13, -67, -48, -34, -234, 31, -73, 85, 105, -78, 67, -148, -106, 50, -88, -78, -26, 77, -43, -24, -73, -14, -11, -18, -33, -88, -48, -145, -154, -35, -127, -23, -18, 17, -166, 13, 37, 68, 52, -118, -32, -62, -220, -36, -110, -54, -57, 78, -110, -100, 39, 16, -113, 3, -27, -16, 62, 5, -136, -88, -3, -2, 5, -14, -88, 18, 29, -61, -56, -170, -11, -19, -57, -31, 150, -30, 66, -1, 12, 88, -74, 11, 48, 47, -40, 43, -3, 28, 36, -56, 5, -101, -60, -51, 42, -21, 26, -97, 75, 148, -110, -8, 78, -88, 36, -25, -48, -75, -8, 87, 35, 61, 27, -62, 37, -58, -60, 52, -76, -62, 18, -26, 38, -125, -87, 107, -76, 16, -40, -16, -82, -72, 35, 57, -47, -85, -130, 11, -1, -15, 54, -103, 34, 5, -134, -12, -18, -21, -98, -103, 74, 138, -1, -8, 40, 46, -100, 67, -104, -132, 44, 48, 9, -2, -28, -195, -43, -14, -72, 112, 6, 28, -3, -8, -129, -3, -64, -28, 87, -73, -75, -33, 42, 42, 55, -39, 24, -46, -122, 44, 49, -43, -70, 92, -118, -54, 46, 40, 56, 2, -44, -124, -6, 20, -4, -194, -1, 56, 94, -76, -13, 20, -7, 0, -54, 41, -83, 38, -57, 35, -112, -15, -34, -195, 33, -73, -93, 58, -107, -39, 0, 53, -7, 124, 69, -56, 40, -110, -103, 73, -206, -47, -51, 48, 32, -130}
, {-26, 48, 4, -34, -86, -18, 22, 12, -1, -51, 37, 49, 58, 31, 26, -41, -47, 48, 23, 10, 12, -37, -30, 18, -3, 16, -135, 113, 60, -42, -32, 32, 75, 19, 8, -35, 34, -15, 17, 29, -24, 14, 49, 43, -22, -68, 20, 156, 50, -44, 21, 73, -6, -21, -129, -28, -15, -48, 31, 31, -25, -18, -33, 35, -92, -29, -132, 87, -45, -5, 84, -98, -55, 50, -69, -54, -114, -10, 68, 1, -68, -76, -132, 47, -66, 9, 52, 9, -103, 16, -65, -34, -22, 5, 23, -4, 24, 47, -48, 6, 72, 62, 78, 47, -2, -26, -13, 17, -43, 22, -46, 0, 135, -108, 62, 24, -71, 6, 9, 22, -13, -38, -32, 55, -20, -2, -12, 57, -13, 285, -16, 36, 16, -2, 21, 10, 4, -99, 93, 208, -53, -136, -14, 46, 25, -36, 5, -144, 84, 126, 10, -15, -44, 6, -120, -113, -2, 88, 59, -111, -54, -21, 11, 102, -53, 76, -53, -4, 161, -98, 28, -13, 42, -45, 26, 23, -24, 179, -29, -104, 37, -95, -16, -75, -36, 23, -35, 105, -135, -91, 39, -4, 35, 102, -16, 130, 78, -6, 27, 44, -15, -156, -135, 49, -46, -94, -52, 101, -13, 19, -21, -51, -30, 39, -17, 16, -73, 52, -48, -78, 93, -43, 12, -106, -6, 100, 59, -92, 26, 34, -82, 45, 82, -33, -69, -25, -144, -55, -31, 69, 15, -29, -134, -8, 79, -161, 9, -27, -67, -130, -15, 97, 91, -87, -72, -100}
, {132, -63, 42, -53, 11, -84, 45, 86, 32, 56, -6, -129, 30, -66, -77, -27, -66, 11, 18, -75, 2, -47, 48, 82, 64, -67, 5, -63, 30, 19, 43, -51, -10, 12, -43, -28, 29, -86, -20, 38, -99, -71, 0, 1, -45, 46, -48, 26, 43, -21, -126, -47, -181, 21, 39, -15, 63, -21, -72, -59, 20, 63, -74, -25, 13, -31, 106, -25, 47, -1, 97, 56, 32, 0, -169, -33, 57, 30, -48, -62, -24, 15, 2, -22, 4, 38, 9, 66, 114, 0, 50, 18, 12, 37, 95, -40, 42, 32, 36, 20, 78, 59, -22, -74, -3, -24, -11, 11, 49, -50, -37, -4, -49, -19, 46, -7, -42, -23, 20, -3, 96, 41, -3, 21, 131, -21, -19, 67, 108, -134, 114, 81, 61, 62, -23, -14, 24, -1, -159, -131, 37, -1, -2, -117, 65, 81, 121, 77, -44, -150, -63, -44, 4, 4, 78, 21, 18, 54, 34, -12, 41, 114, -83, -177, 126, -215, -134, 66, -96, -49, 6, 23, -2, 62, -15, -15, 91, -76, -157, 12, -40, 93, 67, -41, 0, -69, 47, -129, 74, 51, -176, -123, -89, 0, 60, 20, 3, -13, -5, -32, -88, -61, -14, 71, 64, 9, -40, 39, -124, 47, -130, 83, 5, -9, 94, 5, 23, 125, -14, -67, 51, 85, -18, -102, 5, 63, -38, -35, -218, -30, -63, -66, 51, -42, 117, 5, -29, -24, -117, 45, -70, 26, 80, 31, 7, -109, 71, -46, -37, -23, -22, -41, 38, -116, -47, 41}
, {75, 18, -74, -22, -19, 62, 6, -30, 137, -27, 29, -6, -138, -40, -52, -44, -74, 37, -101, -8, 77, -222, -36, -13, -30, -72, 8, -17, 88, 26, 18, -106, -33, 39, -5, 23, -65, 55, -28, -48, -97, 66, 19, -57, -58, 4, 25, -80, 10, 0, 57, 16, -173, 31, -159, -30, 41, 43, -3, 1, -49, -30, -3, 29, -19, -14, 56, 88, 47, 0, -66, 113, 32, 136, 135, -102, -8, -56, 60, -136, 2, -81, -43, -70, -133, 14, 29, 5, 64, 77, 52, 31, -125, -28, -17, 30, -52, 46, -100, -9, 46, -12, 82, 62, -136, -15, -57, -6, -34, 33, -33, -18, 148, -61, 75, -8, -27, -9, 96, 68, -22, 161, -155, 45, 64, 21, -70, -20, -45, 24, -41, 112, 7, -9, -11, 50, -14, 73, -52, 24, -98, -4, -54, 59, 60, -97, 42, -87, 88, 109, -11, 4, -4, -68, 20, -73, 38, 36, 20, -172, -36, -89, 26, -106, -31, 44, 56, 29, -16, -97, 95, -208, -65, 92, -121, -101, -11, -84, -23, 31, -31, -20, -36, -146, -100, -13, -169, -1, -191, -64, -16, -18, 74, -79, 27, -3, 54, 42, -125, 75, 68, -42, -20, -83, 14, -111, 58, -55, 24, -33, -52, -152, 64, 20, 131, -2, -4, -64, -21, 99, -247, 69, -69, -45, -22, -9, 52, -70, -62, 38, -11, 60, -19, -137, -88, -104, -54, -2, -38, -8, 50, 26, -43, -87, -71, 37, -66, -175, -38, 24, -150, -80, -28, 11, 76, -71}
, {-71, 43, 4, -15, -63, 71, -32, 58, -100, -42, 14, 51, 20, -24, 109, 117, -50, -83, -12, -28, -49, 114, 30, -74, -51, -38, 141, -102, -112, -80, 2, -58, 10, -23, 68, 30, -107, -40, 20, 9, 6, -12, 8, -52, 86, -81, -85, -76, -32, 3, 32, -11, 46, -113, 12, -42, 68, 18, -2, -57, 58, -95, 1, -51, 132, 11, -32, 22, -30, 3, -57, 14, -13, -127, -13, 1, -20, 34, -13, 109, -7, 23, 50, 47, 108, 117, -85, -108, 11, 29, -79, 71, 16, 21, 79, -60, -33, -53, 92, 83, -9, 11, -84, -86, 48, -49, -39, -127, -29, -122, -6, 33, -5, -35, -59, -30, 35, -60, -52, -17, 75, -46, -19, -71, -136, -88, 64, 12, -84, -148, -5, -85, -31, -118, 61, -38, -92, 37, -94, -89, 60, 51, 18, -22, -1, -118, -119, 53, -128, -54, 101, 85, -4, -26, -37, 28, 9, 46, 56, 20, 54, 70, 15, -122, 98, -63, 62, 22, 1, 58, 56, -4, -70, 7, -83, -52, -83, -154, 45, 110, -138, -57, 10, 43, -100, -152, -79, -30, 40, -81, -203, -5, -23, -185, -54, 0, -47, 24, -9, -139, 2, 33, -76, -55, 112, 155, -14, -123, -60, -51, 51, 20, -46, -131, -42, 82, 55, -96, 5, -62, 38, -3, -107, -71, 28, -58, -4, 19, 82, -112, -11, -42, -63, 84, 39, 52, 26, 79, -43, -41, -125, -4, 68, 49, -8, 101, -75, -33, 4, 19, 30, -61, 36, 110, -37, 13}
, {-12, -92, -29, 39, -45, -30, -60, 12, -135, -49, -107, -105, 80, 13, 71, 24, -14, 42, 34, 101, -16, 17, -14, 51, 46, 68, -47, -7, -46, -40, -68, 126, -66, -31, 11, -134, 108, -168, 44, 22, -2, -67, 30, 25, -3, 25, 2, 5, -33, -104, -29, 116, -49, -53, 70, -3, -118, -72, 73, -50, 9, 110, -173, 20, -156, 7, -16, -88, 62, 17, -15, -95, -61, 24, -94, 31, -47, -43, 36, -76, -11, -29, -128, 51, -96, -177, 42, 38, -46, -2, -13, -108, -67, 110, -15, -36, 19, 55, -67, -92, -70, 73, -25, -98, 17, -22, 19, -29, 66, -43, -28, -66, -70, 23, 106, 25, 19, -67, -26, -17, 3, -33, 28, -43, 23, -86, 106, 37, -75, 164, 50, 1, -23, -11, -44, -76, -45, -213, 37, 89, -6, -167, 68, 63, 18, -89, -83, -69, -2, 87, -21, 2, 68, -66, -66, -24, -102, 132, -127, -82, -13, 64, 68, -6, 31, 4, -112, 9, 73, 11, -4, 32, 1, -13, 66, -52, -17, 79, 60, -26, 45, -91, 69, -28, 23, -2, 74, 108, -13, -288, 11, -85, 54, 41, -179, 47, -30, -2, -40, 25, -27, -45, -3, -63, -133, 20, -147, -63, -25, 20, 27, -50, 25, -23, -63, -28, -18, 98, 45, 7, 92, -68, 19, -20, -71, 13, 21, 57, -15, -65, -122, -58, -6, 9, -26, -47, -65, -87, -73, 72, 84, 62, -46, -4, 93, -80, 95, -81, -32, 2, -46, 26, -105, 36, -21, -17}
, {181, 60, -9, 139, 234, -33, 49, 20, -71, -35, 96, 99, -14, -16, -90, 142, 98, -26, -56, -13, 36, -27, 12, -210, -55, -56, 163, -39, -18, -18, 74, -14, 7, -103, 21, 17, -54, 34, -114, -52, 226, -15, -132, -185, 12, -33, -35, 120, -61, -93, -115, -116, 7, 16, -82, -97, 13, -33, -38, -83, -121, -13, 21, -54, 102, -19, -19, 125, 20, 67, -29, 22, -15, -37, -43, -76, 41, 51, 34, 34, 90, 78, 15, -108, 85, 103, 86, -34, 57, -7, 26, 92, -52, 113, 72, 32, -51, 6, 77, -12, 8, -4, -20, -16, -87, -56, -98, -47, 27, 51, 6, 30, 71, -135, -85, -34, 55, 66, 32, -63, 0, 85, -24, -139, -41, 85, 37, -111, 23, 3, 30, 58, 48, 2, 34, -63, -9, -56, -64, 81, 70, 7, -70, 65, -9, 7, -38, 3, 12, -51, 17, 81, 61, 7, 41, -55, 19, -10, 13, 53, 16, 3, 50, 57, 91, 57, -13, 12, 7, 140, -13, 13, -83, -6, 52, 17, -42, 47, 28, 18, 39, 32, 27, -22, -55, -128, -43, 17, 68, 25, 11, -20, -97, -25, -3, 9, -77, -139, -15, -64, -23, -106, 82, -32, 43, 9, 122, -9, -88, -59, -36, 51, -19, -7, -14, 35, 56, 70, -60, -60, 55, -9, -21, -78, 1, 4, 84, 66, 7, 5, -41, -6, -43, 45, 36, 74, -11, -43, 103, 17, 37, -17, 102, -31, -14, -184, 19, 136, -33, -32, 145, 41, -13, -36, 6, 38}
, {134, 24, 13, 156, 80, -51, 58, -29, 199, 24, 37, -75, 28, -59, -25, -38, 5, 27, -55, -39, -3, 88, -7, -24, 8, 36, -20, 12, 91, 18, 109, -18, 0, 75, 44, -65, 11, -53, 40, 61, -107, 69, -16, -11, 18, 59, -20, 224, 86, 22, -28, -22, -14, 125, 56, -9, 79, -78, -48, -8, 80, 108, -47, 48, -47, -81, 26, -76, -12, 52, 18, -61, -10, 82, 20, -25, 228, -14, 42, -151, -2, 32, 29, -92, 50, -8, 53, -5, 41, -49, 18, 99, -14, 104, -108, -44, -92, 13, 82, 65, -35, -8, 21, -2, -52, -3, -35, -5, 18, 135, -27, -14, 81, -27, -25, -32, -53, 44, 75, -3, -25, 72, -32, -52, -38, -2, 47, 75, 174, -30, -20, 158, 93, -12, -41, 91, -31, 12, -246, -46, -32, -58, -103, 4, 70, 81, 40, 42, 32, 66, -25, 21, -119, 43, 34, 48, -27, -96, 41, -24, -15, 16, -121, -7, 32, -91, -42, 10, -25, 75, -58, 5, -77, -29, -3, 35, -15, -45, -29, -151, -63, 58, 33, -15, -33, 40, -4, 27, 118, 82, -15, -101, -152, -49, 21, -16, -101, -83, 20, -20, 39, -13, 35, -13, -17, -10, 114, 14, -78, 3, -144, 20, 21, 92, 66, 28, 34, 34, -101, -94, 2, 82, -169, -120, 30, 166, 54, -30, -40, 35, 4, 62, -10, -28, 34, -15, -5, -12, -60, -160, 39, 6, -5, -61, -51, -4, 3, 51, 14, -53, 87, 89, 52, -76, -36, -56}
, {-28, -3, 71, -17, -1, -2, -64, -8, -4, 128, 19, -35, 33, -35, 13, -8, 18, -102, -62, 31, -21, 72, 40, -3, 23, 2, 57, -34, -40, 48, 122, -83, -9, -45, -25, 40, -62, 44, 75, -45, 40, 14, -10, 32, 8, 11, -48, -36, -17, 24, 20, -43, 47, 4, -7, -16, 19, -51, 9, 29, -3, -96, 47, 14, 129, 42, 131, 5, 27, -1, 27, 131, 75, -23, -58, -15, 78, 91, -77, 39, 33, 12, 168, -13, 123, 154, 74, -39, 19, -58, 21, -103, 36, 53, 35, -75, -49, -6, 80, 71, -40, 25, -29, -90, 14, 1, -41, 18, -152, -87, -52, 1, -62, -86, -73, -79, 46, 2, 20, 24, 65, -98, -45, -37, -8, -47, 46, -61, -34, -123, 47, -63, 10, -9, 64, -22, 56, 57, -103, -98, 75, 39, -29, -33, 16, -68, -50, 128, -104, -102, 64, 10, 78, 19, 50, 2, -91, -2, 24, 78, 35, 81, -5, -132, 90, -56, -18, 51, -21, 60, 20, 58, 11, 12, 20, -11, -17, -107, -111, 63, -27, 87, -16, -38, -32, -28, 50, -111, 93, 105, -108, 29, -94, -168, 80, -15, -41, -7, -44, -160, -4, 153, -32, -109, 42, 94, 20, -134, -56, -90, 83, -12, -9, -169, 94, 61, 61, -105, -27, -24, 38, -7, -14, -34, 33, -7, -70, -27, 63, -144, 47, 3, -126, 64, 39, 60, 42, 168, 20, -45, -30, -122, 31, -73, -126, 69, -70, 49, 63, 92, -31, -122, 2, 115, -68, -20}
, {90, 101, -91, 23, 173, 40, -34, -7, 49, -124, 97, 103, -28, -21, 20, -1, -93, -45, 50, 53, 53, -51, -39, -20, -120, 16, 5, 40, 68, -68, -123, -61, -68, -173, -142, 229, 14, 77, 49, -70, 63, 164, -10, -111, -47, -67, 156, 94, -79, -16, 46, -19, 82, -91, -63, 76, 144, 17, 22, 71, -129, 27, 88, 27, -28, -38, -186, 130, -100, -56, 4, -106, -73, -48, 74, -52, 23, 98, -56, 63, -2, -12, -75, 34, 171, -95, 48, -48, -88, 18, -122, 120, -34, 11, -53, 64, 39, -40, -31, 8, -15, -73, -15, 42, -24, -22, 27, 12, 85, 76, 26, 21, 102, -69, -63, -21, -120, -9, -8, -35, -8, 55, 29, -21, -29, 3, -66, -43, -41, -39, 36, 58, 9, 10, 154, -67, -52, -27, -55, 36, 3, 26, 53, -48, -118, -147, 28, -82, -21, 88, -23, 37, 48, 79, 32, 27, -78, -20, -30, -2, 33, 65, 64, -30, 48, 80, 73, -34, 18, -35, 88, 121, -5, -113, -61, -26, -42, -1, 15, 147, 28, -40, 100, 19, 7, -42, 54, -41, 83, 58, 20, 3, 18, 28, -76, 67, 21, -24, 52, -5, 42, -184, 51, 147, -60, 15, -20, -34, 11, 50, 14, -56, -39, 15, -96, 34, 67, 64, -15, -51, 62, -12, 78, -62, 49, 84, -5, -5, 9, -27, -82, -36, 17, -30, -39, 44, -85, -41, -11, 56, -24, 145, 16, 24, 46, -127, 97, 34, 30, -43, 19, 108, 49, -47, -45, 68}
, {78, -98, 99, 42, 4, -39, -18, 176, 67, -14, 100, -52, 57, -39, 10, 50, -56, 38, 18, -24, -1, -11, 46, -16, 56, -3, 150, -35, -81, 11, 22, -82, -74, 19, 70, -142, 9, -22, 36, 5, 54, 46, -44, 3, -6, 47, -78, -141, 24, 0, -88, -20, 78, 21, 23, -83, -51, -12, -49, -152, 16, -42, -12, -36, 78, -33, 107, -6, 10, -32, -24, 98, 37, 108, 20, 22, 92, -63, -72, 23, 29, -56, -2, 17, 48, 144, 152, -74, 96, 46, 74, 80, -6, 71, 58, 53, -40, -28, 58, -28, 12, -45, -134, -52, -15, -112, -47, -130, 95, 41, -85, 38, 47, -62, -82, -80, -18, 76, -11, -72, 54, -87, 0, -76, 32, 41, -26, -87, -107, 101, 42, -18, -5, -109, 82, -6, -70, -39, -33, 81, -42, 20, 65, 24, -20, 28, -65, 60, -5, -105, 102, 72, 112, -131, -4, -53, 62, 50, 14, -65, 128, 27, 23, -12, 46, -34, -66, 61, 1, 55, -67, 66, 17, -21, -7, -108, 61, -200, -10, 42, -77, 41, 43, 6, -110, -103, -71, 87, 148, -39, -118, 11, -13, 5, -59, -16, 49, 83, -19, -98, 20, 42, -62, 23, 26, 43, -18, 0, -47, 90, -78, 119, -160, -110, -14, 85, -55, 73, -16, 32, 46, -92, -2, -36, -68, 51, -100, -15, -18, -165, 41, -44, -5, 7, -11, 68, -99, -57, 55, 152, -107, -14, 46, 76, 66, 55, 12, 67, 87, 60, 69, -58, -1, 77, -85, -116}
, {66, -116, 44, -15, 4, -9, 3, 51, 85, 13, -49, -136, 54, -22, 31, 1, 4, 3, -4, -52, 71, -93, 1, 74, 22, 23, 65, 40, -6, -52, 127, 87, -32, -66, 35, -86, 58, -131, -13, 59, -87, -49, -21, -8, 21, 96, 10, -47, 56, -51, -72, 10, -68, -25, 0, 31, 10, 18, 27, -51, 74, 104, -52, -49, -44, -25, 149, -30, 45, 37, 2, 6, 47, 43, 5, 14, 98, 15, 112, -46, -62, 15, 0, -4, -99, -70, -15, 22, 83, 79, 47, -46, -46, -9, -17, -40, -6, 0, 42, 51, -35, 97, 7, -12, 44, 30, 51, 1, -7, -21, -49, -6, -126, 94, -27, -32, 47, 29, 66, 41, 37, 18, 45, 77, 28, -29, 83, 46, 24, -14, 45, 136, 19, 19, 35, -114, -32, -94, -102, -15, -29, -51, -51, 73, 35, 8, 92, -78, 57, -125, -5, -19, 60, -89, 13, -60, 33, 96, -22, 62, -4, -2, 106, -42, 77, -176, -190, 110, -18, -31, -153, -12, 27, 91, 55, -85, 134, -87, -149, -31, -28, 11, 23, -22, -63, -43, 23, -48, -108, -7, -127, 27, 35, 41, -20, -59, -25, -77, 28, -41, -14, -142, 55, 33, 27, -75, -155, 2, -87, 54, -134, 68, 93, -223, 70, -59, 14, 55, -47, -12, 60, 138, 77, -71, -126, 75, -60, -115, -117, -171, -149, -124, 54, -67, 47, -49, -3, -42, 59, 61, -93, -48, 4, 48, 6, -75, -11, -83, 30, -10, -1, -89, 9, 12, -196, 94}
, {-124, -54, 46, -69, -144, 19, -9, -91, 43, -12, -4, -47, -3, -35, -16, 30, -103, -25, 12, -102, 21, -122, -14, 40, 24, -18, 5, 9, 42, 28, 84, -57, 27, 72, 22, -67, -9, -86, 23, 52, -71, 46, 35, 53, -55, -18, -29, -57, 44, -38, -41, 37, -14, -3, -54, 27, -53, 62, -22, -19, 51, -123, -6, 75, -5, 60, -38, -25, -26, 15, 100, 105, 7, -184, 67, 42, -44, -43, -11, -56, 6, -8, 33, 7, -132, -46, -222, 4, -43, 16, -95, -30, 11, -105, 34, -5, 23, 51, -44, 67, 65, -111, 11, 13, 72, -9, 35, -50, -196, -133, 11, 28, -44, -75, 3, 23, 21, -174, 26, 29, 40, -72, -7, -2, 86, -20, -92, 23, 20, 15, 63, 27, 45, -65, -27, 119, 75, -91, -190, -8, -31, 28, -26, 71, 70, 34, 4, -59, 39, 4, 64, -114, 38, 44, -67, -20, 9, 56, 9, -27, 24, -2, 7, 22, 28, -31, -82, 67, 42, -103, -21, -7, -44, 69, 42, -19, 8, 1, -52, 35, 0, 81, -1, 11, -125, -102, -47, -15, 25, 19, -27, -53, -21, -126, 125, -12, 33, -27, -216, 51, -6, -64, -91, -5, -18, -21, 53, -82, -12, -11, 15, -109, 32, 46, 29, -77, 29, -32, -2, -80, -83, 42, -99, 82, 1, -13, -73, -77, -49, 35, -77, 74, -122, -152, -109, -28, 12, 174, -16, -98, 22, 7, 138, -113, -2, -24, 9, -90, -131, 19, -184, -41, -142, 38, 102, -240}
, {67, 7, 12, -9, 41, -39, 29, 72, -16, -58, -78, -4, -47, -11, 12, -29, -12, 0, 91, 23, -23, 17, 2, 24, 40, -74, 182, -29, -63, 12, 100, -100, 55, -37, 21, 9, -62, 32, -44, -102, 125, -11, 5, -17, -52, -48, 65, -98, 3, 6, 1, -7, 36, -31, 90, 13, -103, -58, -1, 63, -24, -97, 1, -77, 114, -29, 48, 58, 14, -19, -127, 60, 39, -9, -1, 56, 21, 38, -155, -6, -18, -44, -8, 29, 130, 47, 33, -79, 27, 21, -7, 41, -28, -58, 18, 78, -15, -85, -98, 8, -7, 27, -64, -87, 23, -77, -76, -98, 25, 63, -12, 8, 53, -12, -52, -171, 33, -42, -21, -58, 73, 12, 18, -10, 14, -19, -109, -72, -134, -17, 50, -55, -30, -175, 6, 128, -161, 127, 11, -34, -41, -82, -72, 75, 25, -94, -111, -87, -3, -18, 3, 70, 113, -177, 12, 108, -3, 77, 70, -48, 49, 22, 78, -185, 96, -125, -29, 47, 8, 40, 22, 25, -41, -6, -60, -61, 109, -151, -93, 127, 45, 61, -77, -108, -102, 65, -94, -76, 175, -197, -135, 97, 51, -88, 65, -50, 2, 105, -88, -24, -5, 43, -29, -30, 1, 46, 17, -57, -23, -12, -25, 62, -32, -171, 2, -28, -17, -18, 61, 116, -107, -36, -91, 32, -49, -109, -101, -115, 72, -38, 153, 53, -65, -38, -22, 58, -5, 40, -62, -32, -4, 41, 92, 15, 17, 42, -8, 12, 18, 133, -57, -139, 58, 6, 9, -29}
, {82, -97, 29, 149, 43, -48, 10, 90, 123, 118, -77, -59, -13, -118, -74, 60, 94, -85, -130, -26, -24, 176, 49, -23, 0, 49, -21, -53, -61, 40, 30, -34, 78, -25, 30, -103, 61, 175, -138, 73, -124, -96, -120, 65, 14, 42, -146, -77, 19, 121, -91, -33, -4, 94, 59, -50, -50, -214, -83, -100, 60, 72, -53, -24, 5, -7, 178, -114, 36, 54, -51, 3, -2, 51, -11, -28, 64, 78, 56, -13, -36, -25, 33, -28, -96, 2, -61, -88, 116, -3, 18, 77, 14, -23, -46, 13, -55, 50, 102, 40, 28, 75, -92, -49, 7, -105, -81, -34, 16, -52, -36, -42, -131, 18, -27, -93, 26, 92, -12, -72, 41, 35, 32, 22, 36, 27, 55, 7, 62, -175, 28, -28, 54, -27, 67, -44, 3, -2, -254, -202, 10, 15, 34, -140, 24, -5, 38, 126, 8, -114, -4, -32, -10, -125, 110, -22, 31, -135, -20, 117, -18, 8, -210, -123, -7, -34, -46, 40, -150, -105, -168, 7, -135, 52, -3, 14, -3, -158, -34, -167, -195, 87, 25, 56, -51, -73, -54, -118, 5, 99, -100, -83, -69, -14, 161, -5, -147, -132, -7, -29, -23, -55, -28, -14, 68, 32, -5, -42, -28, -8, -35, 67, 18, -111, 33, 8, 10, 8, -137, -4, 12, -27, -31, -124, 26, 89, -24, 11, -83, -25, -37, 58, 18, 34, 63, 79, 24, 49, 56, 21, 92, -134, 61, 86, -98, -52, -44, -25, -55, -20, -1, -40, 49, -48, -175, 47}
, {-93, -49, 33, -17, -1, -35, -25, -30, -118, 65, -43, -128, 116, -111, 106, 85, -63, -63, 1, -57, -128, -114, 85, 49, 40, 21, -84, 22, -106, 44, 31, -20, 2, 28, 119, -238, 88, -82, 23, -52, 4, 44, 47, 41, -56, 10, -89, 38, 11, 3, -63, 26, -87, 23, 47, -18, -17, -114, 26, 23, 2, 31, -89, 12, 31, 86, -37, -6, 2, -114, 54, -39, -84, -80, -66, -47, 74, 18, 40, -6, 7, 89, 74, -79, 107, 0, 4, -74, -30, -7, 25, -15, 23, 112, -20, -44, -67, 3, -32, -28, -21, -112, -2, -29, -26, -16, 5, -24, -21, 67, 71, 13, 43, -227, -30, -30, -100, -79, 27, -19, 34, -134, -71, -44, -63, -38, -79, -15, -99, 111, 1, -130, -53, 18, 0, -116, 58, -243, 33, 31, -63, 9, 23, 19, 25, 77, -1, -26, 66, -117, -118, -62, -47, -61, -173, -144, 10, 49, 165, 36, -9, 33, -36, 54, -114, 1, -171, -13, 45, 93, -192, 60, 18, -122, 59, 9, 12, -3, 73, -236, -21, -120, -5, 12, -51, -13, 50, 58, 1, -19, -34, 76, 20, 19, -194, 35, -7, 60, -46, -30, -37, 12, -107, 2, -38, 38, -50, 22, -8, 123, 71, -155, -31, -84, -46, -11, -39, 44, -6, -159, 25, -84, -41, -19, -47, 100, -122, -54, 127, 26, 14, 79, 7, -125, -74, -14, -158, -98, -13, -12, 55, -12, -26, -3, 66, -46, 65, -14, -6, 2, -32, -2, -175, 46, -32, -268}
, {63, -167, 17, -20, -107, -17, -27, 135, -107, 42, -76, -151, 5, -27, 13, 102, 19, -78, 19, -1, -40, -58, 76, 31, 11, -8, -20, -47, -87, 18, -16, -129, 9, -5, 128, -95, 80, -60, -84, 81, -108, -21, -112, 75, -43, 32, -99, -159, -49, -160, -81, -43, -102, -48, 21, -34, -34, -8, -101, -83, 23, -21, -78, -60, 49, 40, 145, 8, 69, 47, 17, 66, -124, 16, -17, -35, 34, -85, -108, -28, 56, -58, -53, -14, 85, 100, 11, -167, 22, 18, 15, 3, 21, -31, 40, 18, 2, 5, 136, -45, -7, -82, -98, 39, -31, 61, 1, -68, 92, -55, -4, -8, 16, -40, -23, -40, -16, -41, -20, -19, -8, 33, 70, -95, 69, -5, 65, -8, -38, -56, 59, -162, -79, -105, -15, -20, 163, -154, 48, 6, -13, -2, 15, 33, 1, 54, 147, -124, 90, -68, 94, -24, -27, -58, -196, -114, 71, 26, 18, 71, -22, 10, 28, 36, 2, -18, -91, 22, -85, -115, -121, 2, 26, -2, 71, -84, 22, 17, -79, -20, -106, -119, -113, 7, -143, 7, -88, 28, 16, 95, -40, -77, -18, 51, -110, -22, 13, 46, -61, 73, -11, -50, -27, 58, 40, 2, -198, 34, 71, 59, -17, -53, -68, -2, 13, -131, -150, 110, 48, -80, 61, -82, 5, -24, 26, 108, -119, -55, 75, 46, -208, -25, 66, -74, -17, -115, -183, -26, -33, 31, 34, -48, -7, -20, 30, -65, 98, -119, -27, -82, -79, -9, -49, 16, -42, -101}
, {19, 30, 49, 36, 60, 72, 1, -120, -43, 30, 70, 78, 16, -41, -7, -59, -117, -39, 6, 19, -20, -32, 14, 40, 18, 45, -90, 65, -32, 40, -11, -22, 2, -56, -9, -9, 80, 15, -33, 21, 52, 28, 42, 31, 54, -13, -33, 42, -38, 50, 20, 1, 69, 19, -28, -27, 42, 73, 40, -24, -18, 11, 87, -1, -31, 57, -68, 151, -23, -83, 47, -28, -91, -156, -67, -34, 25, -11, 17, 103, -41, -33, -16, 24, 98, -102, -60, -74, -146, -29, -124, -13, 69, 18, 82, 20, 37, 43, 38, -17, 0, -135, -3, 80, 2, 66, 8, -89, 40, -6, 79, 26, 19, -178, -35, 58, -60, -13, 43, -52, -40, -162, 26, 8, -52, -58, -78, 66, -68, 61, 37, -92, -51, -42, 134, -77, -87, 10, 11, -20, -101, -63, 69, 125, -88, 12, 51, -14, -5, -150, -40, 8, 35, -59, -110, -35, -19, 74, -26, -50, 76, 18, 112, -59, -79, -19, -10, -74, 27, -11, 28, 6, 80, -152, -3, 12, -15, -34, 5, 22, -74, -98, -51, -1, -11, 38, 0, 51, -88, 9, -33, 16, -77, 6, 32, -8, 70, 42, -77, -41, 87, -131, -77, -32, 65, 25, 41, 51, 36, 49, 26, -19, -170, -17, -127, -23, -52, 114, -60, -132, 16, -124, -53, -112, 67, 44, -40, -8, 113, 31, 33, 20, -14, -60, -93, 153, -157, -176, -12, -50, 40, 89, -65, -4, 113, -95, 163, 49, -28, -121, 35, 132, 76, 34, -7, -196}
, {-55, -31, -100, -27, -115, 32, 68, 51, -7, -9, -27, -17, 24, 26, -80, -28, 24, 46, -37, -6, 49, 75, 46, 34, 55, -2, 72, 43, 33, -44, 14, -39, -36, 67, 26, -8, -2, 9, -75, -34, -52, -43, 21, 24, -10, 94, -14, -10, 70, 8, -12, 8, 55, -13, 37, 7, -77, 19, -28, -34, -19, 40, -3, -48, -92, -51, 55, 55, -51, 162, -78, 12, 31, -4, -5, 12, -12, 0, 66, -84, -8, 2, -27, -22, -81, -40, -61, 87, -130, 125, 109, -36, 25, -50, -115, 54, -62, -17, -27, -28, -39, -105, 96, 50, 49, -38, 33, 43, -111, -83, 2, -86, 41, 45, 25, -29, 73, -36, 7, 31, 9, 81, -8, 44, -13, -10, -6, 37, -102, 38, 71, -88, -40, 38, -71, 18, -76, 47, 116, 100, -136, -121, 126, 74, -105, -118, -123, -98, 0, 46, 67, 26, 67, -65, 5, -19, -15, 46, -14, -184, 77, 76, 184, -71, -70, -67, 144, -62, 40, 157, 81, -14, -76, 74, -65, -78, -43, -65, -21, 281, 42, -58, -103, -44, -17, 20, -51, 7, 46, -166, -55, 42, 53, -61, 2, 12, -11, 73, -48, 40, 74, 15, 30, -5, 91, -66, 1, -88, 33, -58, 56, -36, -25, -75, -25, 34, -26, -108, 41, 132, -211, 25, -8, 20, -82, -70, -41, -22, 38, -60, -7, -10, -19, 20, -68, 37, -28, -84, 62, -17, -31, 109, -36, 47, -64, 115, -54, -44, 24, -5, -93, -59, 47, 8, 108, -91}
, {4, -71, 98, 30, 29, -94, -21, -17, -153, 45, -31, -8, -175, -42, 9, -17, -121, 65, 1, -53, -54, -123, -10, -59, -45, 88, -85, -102, 16, -7, 12, -86, 60, 3, 77, -57, 53, -107, -27, -47, 18, -33, -29, -51, -28, -4, 19, -121, 24, -122, 14, 3, -85, 12, 14, -79, -3, 48, -24, -42, -40, -51, -121, -8, -11, 24, 71, 71, 48, 48, 33, -30, 8, -36, -7, -77, -55, 73, -9, -59, 37, -11, -23, -25, 94, -78, -52, 9, -42, 16, -69, 31, -20, 0, -9, 52, 74, 40, -92, -37, -4, -48, -78, -67, -55, -1, 2, 69, 30, 15, -37, -23, 40, -80, 16, -49, -68, 27, 29, -24, -24, -54, -41, 9, 62, 119, 24, -79, 67, -36, 41, 161, 31, 11, -38, -39, -28, -32, 8, -116, 15, -112, -6, -28, 60, 72, -52, 8, 18, -185, -29, -21, 39, -17, -33, -71, -78, 44, 11, -3, 46, 70, 16, -121, 94, -167, -43, 45, -72, 20, -1, -11, -60, -25, 23, 44, 53, -62, -118, 72, -96, 81, 23, -23, -13, -77, 37, -102, 15, 24, -90, -67, 52, 74, 46, 46, -16, -91, 51, 50, -87, -164, 94, 101, -5, -215, 47, 138, 15, -15, -138, 161, 42, 33, 63, -142, 21, 9, 31, -44, 75, 139, 95, -11, 15, 94, 14, -11, -140, -22, -62, -98, 108, 29, 153, 1, 20, 70, -10, 35, -53, 4, 78, -10, 71, -110, -56, -102, -44, -80, -3, -55, 12, -52, -93, 108}
, {-169, -110, -29, 8, -27, 27, -86, 11, -159, 3, -54, -12, 55, 68, 93, 83, 0, -29, 23, 90, -59, 102, 17, 62, -5, 75, 118, -5, -106, -36, -27, -22, -60, -30, -18, -83, 1, 18, -40, 10, 104, 33, 2, 58, -26, 19, 14, -124, -62, 0, -25, 50, -36, -40, 107, 5, -125, 7, 1, 91, -27, -96, 9, 49, -22, 90, -65, -71, 19, 6, -5, -146, -210, -160, 92, -12, -124, 56, 17, 54, -38, 81, -20, 124, -70, -43, -217, -11, -129, -4, -130, -110, 61, -192, 60, -186, -36, 0, -57, -14, -30, 43, 82, -75, 46, 38, 72, 71, -210, -246, 10, 13, -211, -6, 0, 56, 31, -230, -26, 22, 5, -123, 58, -6, -33, -58, 11, 29, -68, -63, 34, -147, -232, -22, -22, -88, -15, -5, 32, 14, -48, 7, 47, -38, -182, -7, -82, 115, 11, -24, 46, 97, -28, -60, -135, 9, -11, -11, 42, -69, -25, 38, 34, 3, -102, 25, 65, -84, -26, 74, 37, 39, -122, 1, -17, 97, -168, 9, 54, 68, 24, -70, -90, 31, 22, 37, 50, -12, 85, -91, -1, -1, 47, 5, -58, 105, -46, 63, -18, -29, 33, 85, -102, -151, 65, 133, -72, -23, 26, -17, 126, -53, -12, 23, -61, -38, 133, -72, 70, -44, 47, -86, -51, -23, 43, -85, -63, -77, 48, -4, 49, 8, -99, 41, 10, 27, -28, -26, -48, -25, -11, 22, -22, -58, 66, 17, 25, -70, -9, -93, -87, -26, -55, 19, 28, -16}
, {-70, -39, 87, -53, -94, -46, 4, 43, -80, 66, -29, -58, 57, 27, 51, 34, -57, -25, 64, -31, 22, 65, 2, 55, 84, -38, 131, -11, -105, 22, -36, -20, -33, 79, -36, -44, -12, -49, 20, 25, 16, 19, -46, 42, 6, 45, -38, -174, 27, -69, 42, -4, 24, -19, 28, -49, -69, 42, -56, -51, 12, -37, 2, -31, 57, 38, 63, 68, -65, -44, 39, 63, 39, -86, 63, -7, 19, 19, -135, 77, 45, 84, -39, -24, 43, -26, -53, -15, -33, 84, -46, 4, 35, 20, 6, 35, 28, -6, -128, 5, -110, -75, -13, -116, 30, -36, 30, -96, 25, -17, -65, 51, -25, -111, -23, -127, -97, -68, 12, 0, 28, -77, 52, -21, 16, 31, -164, -51, -196, 65, 10, -38, -119, -173, -20, 20, -97, -4, -59, 70, -45, -49, -41, -8, 38, -19, 21, 82, 29, -95, 20, 44, 73, -53, -58, 41, 16, 123, 103, -54, 127, -6, 117, 22, 81, -89, -57, 13, 47, 100, -25, 33, -82, -28, -2, -125, 52, -212, -7, 153, -71, 28, -36, -2, -71, -95, -85, 43, 107, -149, -86, 6, 20, 21, 158, -47, 74, 61, -98, -32, 0, 35, -50, 8, 58, 14, -62, -142, -14, 99, -49, 3, -52, -193, 23, -77, -134, 29, -5, 6, -18, 24, 27, -59, -130, 29, -171, -31, -9, -141, 89, -53, 24, -34, -90, -16, -143, 2, -2, 22, -162, -27, -29, -73, -48, 23, -30, -68, 78, 42, 21, -18, -18, 59, -107, -204}
, {-39, 4, 29, 71, -18, -63, 51, 13, -15, 33, 48, -172, -31, 46, 4, 69, -24, 16, 61, -12, 57, 17, 50, 34, 21, 12, -64, 70, -38, 20, 74, 67, -23, -16, 85, -212, 19, -58, -3, 18, -214, 44, 46, -3, 68, 75, -3, 89, -22, -1, -19, 14, -129, 70, 79, -7, 49, 30, -24, 0, 62, 0, -146, 31, 10, 12, 111, -84, 56, 29, -73, 3, 13, 2, -1, -30, 99, -18, 33, -153, -109, 49, -10, -2, -56, -92, 45, 15, -63, 29, 40, 34, -25, -12, -36, 44, -138, -15, -21, -50, 85, 27, -13, 54, 14, -1, 7, -17, -24, -118, -16, 8, -43, 47, 75, 62, 85, -20, -70, -13, 25, -42, -28, 17, -12, -124, 71, 63, -63, 8, -83, -30, 1, 1, -102, 105, 24, -35, 83, 58, -63, 59, -59, -10, -73, 8, -50, -25, -36, 193, -47, 12, -216, -59, -30, -100, -23, -140, 75, -48, -99, -94, -49, -35, -172, 64, -22, -56, 41, -54, -144, -9, 9, 47, -43, -47, -73, 22, 16, -132, 89, -136, -69, 90, 7, 80, 5, 86, -75, -35, 10, -73, 159, 6, 51, 48, -61, 27, -63, 31, 121, 22, -41, 0, 23, 25, 5, -33, 67, -45, 109, -178, 26, 64, -3, 34, 22, -72, 23, 75, 3, -46, -37, 77, -47, -149, 55, -32, 110, 35, 18, 130, 10, -43, -67, -76, 24, -55, -25, -91, 128, -56, -110, 8, 26, -7, 6, -100, -200, -51, -77, -48, -109, -8, 13, 21}
, {77, -14, 14, 89, -10, -44, 10, -7, 8, 16, 75, -36, 5, -37, 13, 14, -3, -36, -34, 9, -64, -34, 6, 41, 21, -76, -80, 28, -23, 32, 40, 18, -17, -87, 52, -9, 70, -80, -52, -5, -82, 27, -10, -59, 31, 24, 10, 24, -1, 20, 7, 66, -84, 12, -31, 33, 51, 14, -1, -3, -25, -47, 57, 33, 47, 13, 1, 3, 56, -34, -51, -21, -45, 11, -37, -63, 98, -151, -127, 69, -28, -7, -132, -88, 30, -87, 36, -42, -7, -78, 35, 14, -60, 81, -23, -108, -76, 73, -47, -52, 18, -107, -124, 79, -46, 22, -2, 28, -9, 61, -18, 4, -32, -78, -57, 44, 11, 68, -40, -45, 14, -78, -16, -100, -6, -52, 13, -28, -41, 30, -112, -13, -50, -16, 7, -84, 58, -153, -154, 38, -42, 48, 15, 17, -22, 23, 16, 77, 77, -13, -85, -72, -95, -43, -135, -68, 67, -49, 143, -135, -8, -121, -109, 72, -111, 86, -48, -84, 30, 61, -42, -19, -140, 4, 11, 24, -51, 22, 46, -104, -25, -57, 71, 33, -133, -24, 9, 5, -51, -92, 0, -23, -71, -114, -165, 69, 17, 15, -222, -17, 79, -119, -150, -48, -46, 115, 61, -37, 35, 34, 101, -92, -142, 41, -194, 162, 29, 29, -4, -175, -115, -154, -144, -4, 45, -64, -76, -4, 14, 116, 43, 98, -13, -63, -187, 75, -114, 39, -70, -141, 125, 77, -12, -18, 90, -169, 76, 95, -176, -113, -47, 92, 27, -20, 84, -80}
, {-145, 113, -17, -30, 72, -19, 47, -10, 15, 2, 37, 26, -28, -58, 65, -13, 0, -9, 106, -70, 54, 117, -34, 19, -58, 76, 19, 2, -7, -57, 3, -12, -23, 14, -87, 58, -31, 40, 48, 9, 79, 91, 47, 11, 10, 12, 42, 134, 33, -24, 36, 10, 71, -22, 10, -67, 125, -30, 36, 74, -5, -30, 43, 15, 24, -67, -228, 31, -78, -30, 17, -91, -73, 1, 170, 38, -129, -5, -11, 91, -113, 99, 92, -13, 128, -70, -37, 15, -111, -48, -152, -58, 16, -143, 1, 6, -17, 32, -44, 53, 26, -40, 101, 23, 24, 13, -8, -10, -83, -13, -20, 88, -74, 53, 27, 57, -49, -5, -32, 59, 56, 3, -15, 91, -15, 16, -87, 62, 17, -32, 36, 117, 106, -4, 9, 77, -108, 24, -139, -69, -10, -137, 26, 65, 9, -19, 10, -25, -87, 87, -12, -92, 133, 30, -2, 33, -166, 3, -53, -45, 96, 120, 52, -90, 41, -56, -15, -18, -26, -90, -23, -60, -26, 0, -128, -83, 19, -66, -157, 183, -86, 4, 150, -190, 19, -72, -55, -118, 156, -42, -54, -67, -54, -68, 89, -32, -81, 49, -97, 0, 66, 45, -38, 68, 73, 19, 101, -72, -78, -53, -100, 3, 0, -45, 47, 10, 34, 65, 19, -85, 53, 64, -13, -82, -84, -29, 12, -84, -136, -4, 16, 47, 59, -134, -41, 50, 12, 47, -161, -32, -22, 32, 150, -108, 92, 55, 9, 5, 45, -15, -62, -20, 17, 23, -76, 49}
, {87, 82, 45, 115, 146, 73, 69, -29, -41, -45, 42, 34, -31, -37, -88, 80, -11, -131, -145, -23, 4, -201, -55, -138, -55, 70, 29, -100, -20, -68, 20, -76, -57, -79, -30, -50, -19, -75, -87, -160, 43, 76, -22, -126, 0, -61, 8, 51, -83, -176, -31, -55, -48, -12, -131, -54, 73, 67, 45, -26, -106, -114, -1, -24, 75, 15, 53, 110, -17, -102, -4, -4, 15, -59, 11, -93, 16, -62, -71, 148, 109, 103, 64, 90, 226, 18, -58, -94, 36, -106, -89, 34, 32, 142, -60, 8, -28, 27, 62, -6, -99, -1, -133, 22, -86, -71, -45, 27, -25, 53, -61, -29, 84, -182, -11, -118, -114, -45, -42, 49, -171, -11, -87, -38, -111, 27, -97, -118, -46, 38, 99, 2, 6, 13, 45, -82, 29, -55, -15, 15, -50, 43, -103, -89, -83, -53, -209, 101, 147, -62, -41, 47, -64, 33, 19, -13, 17, -90, 22, 6, -35, -14, -75, 65, 25, 78, 15, -22, 61, 44, 36, -39, -64, -18, 68, 42, 3, 85, 25, -145, 23, -6, -21, 4, 44, 29, 19, 80, 43, 52, 46, -125, -29, -67, 61, 39, -50, -143, 21, 1, -106, -102, 88, 4, -23, 5, 152, -41, -67, 64, -80, -60, -47, -57, 45, -41, 22, -62, -81, -110, 70, 66, 41, -34, 0, -52, 28, 62, -34, 27, 36, 50, 23, -87, -47, 56, 105, 69, 42, 13, 63, -19, -15, -24, 39, -73, -10, 25, -23, 10, 45, 70, -67, -2, -119, 58}
, {54, -84, 24, -68, 25, 39, -107, 105, 31, 134, -115, -138, 86, -52, -69, -37, 27, 90, -22, 6, -11, 35, -7, 14, 20, 15, 61, 31, -85, 41, 71, 10, -71, -32, 19, -141, 26, 12, -82, 25, -102, -141, -41, 19, 42, 51, -1, -15, -7, 4, -89, -70, 8, 59, 34, -148, -62, 8, -77, -21, -19, 45, -57, -72, -111, -170, 191, -74, 64, 126, -90, 3, -58, 18, 23, 27, 51, 29, 91, -138, -52, -119, -84, -65, -218, -29, 21, 11, 137, 91, 19, -66, -175, -11, -80, -73, -99, -32, 66, -48, 1, 123, 64, -15, 104, -56, -11, 47, -6, -70, -227, -178, -138, 78, -27, 34, 97, -9, 24, -42, 38, 46, 42, 8, 91, -55, 126, 114, 17, 27, -72, 51, -24, -28, -81, -174, 39, -68, -1, -45, 10, 15, -25, -71, 5, 11, -6, -52, -65, -87, -85, 16, -16, -12, -61, 72, 40, 2, 34, 45, -48, -142, -82, -70, 8, -66, 34, -46, -41, -76, -116, -41, 24, 99, 29, 42, 2, -60, -17, -42, 47, 17, -91, 49, 28, 33, -25, -62, -69, -8, 37, 39, 65, 20, 73, 21, -102, -148, 80, -8, -68, -57, -52, -55, 73, -77, -14, -38, 51, -95, -63, 72, 6, -31, 36, 2, 23, 42, -68, 80, -24, 60, 126, -11, -26, -36, -39, 59, -46, -153, -56, -108, 67, 30, 117, 66, -25, -91, 48, 83, -9, -114, 9, 132, -57, 4, -102, 0, -23, -40, 70, -121, 89, -102, -63, 62}
, {-210, -131, 27, -93, -110, -11, -89, -40, -93, 32, -85, -101, 150, -23, 14, -29, 5, 17, 8, -98, -102, 18, 56, -16, -35, 9, -89, 19, -58, 82, -60, 49, -49, 70, 27, -175, 26, -66, -2, 15, -55, -170, 18, 92, -14, 55, -11, -101, 62, 21, -89, 35, -4, -26, 52, -13, -26, -55, 31, -30, 55, 68, -106, -29, -16, -56, 44, -99, 108, 159, -10, 1, -58, 14, -68, 68, 115, -16, 19, -25, -70, -65, -132, 2, -130, 32, -102, -64, 47, 34, 186, -111, -78, -147, 32, 7, -58, 42, 79, -38, -12, 125, 17, -69, 68, -18, 33, 123, -38, -195, 43, -26, -151, 168, 14, 47, 131, -175, -31, -34, 80, -32, 114, 34, 81, -83, 189, 73, -80, 23, -33, -64, -120, 11, -103, -98, 8, -151, -8, 18, -20, 18, -17, 9, 10, 92, 14, -129, -44, -1, -78, -51, 38, -78, -84, -224, 39, 43, -10, -6, 49, -73, -16, -3, -41, -83, -59, 50, 10, -63, -180, 30, 24, 16, 60, 46, 21, 25, 17, -84, 7, -47, -98, -10, 43, -81, -23, 49, 2, -48, -46, 1, 12, -1, -104, 62, 29, -37, 77, 5, -40, -69, -40, -4, -65, 38, -117, 12, 61, -64, 14, 67, -89, 22, 26, -26, 17, 55, -21, 87, -13, -21, 59, -37, -7, 45, -87, -44, 30, 13, -86, -104, 35, -1, 56, -10, -23, -175, -23, 62, -16, -82, 4, 41, 31, -92, 1, -36, -4, -42, 12, 7, -56, 5, 59, 58}
, {239, -4, -35, 122, 105, 2, -13, 57, 41, -34, -39, -13, -7, -5, -70, 29, 19, -47, -17, -77, 5, 1, 24, -64, -67, -131, 29, -82, 25, 33, 7, 90, -103, -84, 40, -8, -32, 50, -183, -16, -92, -66, -142, -84, 16, 1, -45, 78, -1, 9, -32, -45, 40, 69, 48, -42, -94, 13, -90, -74, -147, 19, 83, -93, 54, 7, 106, 63, -8, 68, -28, 79, 73, 116, -57, -4, 77, -72, 21, 73, 12, 35, -85, -82, 45, 137, 102, -59, 95, 35, 109, 25, -31, 31, -85, -41, -7, -27, 111, -91, -75, 19, -74, -19, -61, -122, -90, -64, 119, 20, -118, -23, 67, -1, -116, -87, 59, 31, -48, -137, 27, -14, -116, -28, -5, 99, 5, -90, -58, -6, -5, -4, 24, -2, 41, 93, 13, 3, -48, 49, 3, 44, -78, 27, 11, -19, -95, 77, -61, 87, 61, 18, 53, 32, 93, 21, 62, 21, -7, -14, 65, -7, 0, 51, 4, 42, 15, 40, 44, -31, 38, 32, -23, 24, 11, -71, -42, -13, 62, -32, 34, 29, -64, 2, -57, -6, 32, 3, 5, -26, 16, -38, -63, -135, 6, -67, -13, 61, -68, -84, 83, 90, -30, -52, 18, 116, -4, -25, -35, -30, 9, 116, -115, -29, 20, 46, -17, 55, -12, 88, -57, -88, -76, 2, 11, -45, -46, -7, -17, 57, 77, 56, -63, 20, -57, 36, -7, 21, 62, -8, 25, -12, 91, 4, -29, 73, -22, 109, 40, 115, 72, 4, -4, 71, 69, -63}
, {11, -199, -14, -137, -104, -43, -65, 146, -78, 26, -41, -195, -15, -36, 55, 70, -35, 52, 116, -101, -45, -102, 33, 22, 4, -23, -44, -15, 6, 51, 30, 18, 25, -50, 21, -49, 42, -127, -2, -35, 21, -80, -26, 71, -102, -40, -60, -96, -28, -127, -48, -5, -86, -9, -87, -115, -141, 14, -156, -62, -18, -53, -137, 15, -140, 69, -22, -177, 105, -36, 51, -97, -94, -93, -117, -18, -93, -23, -1, -52, 62, 26, -20, 91, -55, -71, -150, -44, -136, 48, 43, -62, 6, -186, 19, 36, 55, 40, -155, -133, 76, -37, 8, -27, 82, 47, 55, -89, -242, -221, 32, -44, -43, -61, -25, 27, 8, -163, -88, -26, -57, -106, 81, -149, 27, -54, 37, -32, -134, 49, 69, -116, -143, -50, -95, -110, 117, -108, 30, -37, -17, 25, 90, -15, -122, 37, 5, -64, 14, -95, 24, 22, -34, -32, -146, -86, 40, 40, 25, -84, 43, -92, -21, 9, 0, 13, -12, 40, -55, -116, -9, 61, 34, -19, -14, -102, -11, -112, -14, 34, -139, -86, 22, 78, -13, -43, -10, 8, 68, -8, -8, -130, 41, -11, -128, -29, 47, 18, -59, 17, 8, -50, -148, 77, 56, 5, -141, 41, 28, 55, -117, 6, -178, -2, -4, -247, -60, 105, 51, -214, 89, -130, 5, 9, 16, 49, -63, -83, -3, 13, -228, -13, 39, -87, -25, -20, -161, -39, -157, 45, 30, 32, -112, -26, 58, -15, 71, -156, 40, -143, -87, -15, -48, -17, -59, -199}
, {-42, -77, 37, 66, -3, 66, -29, 63, 173, 58, 16, -97, 13, -9, 18, 23, 14, 102, 16, -30, 42, -36, 28, 30, 38, 10, 29, -1, -41, 74, 100, 17, -55, -5, -33, -105, -39, 21, 38, 56, -87, -11, 23, 58, -29, 69, -20, 76, 72, 97, -55, -43, 38, 76, 58, -15, 24, -48, 30, 29, 26, 54, -19, 18, -22, -112, 119, -135, 98, 89, -108, -44, 54, 2, 76, 26, 21, -24, 175, -182, -164, 33, 15, 22, -138, -21, -69, 10, 61, 75, 90, -36, -101, -35, -81, -139, -199, -70, 121, -2, 39, 117, 87, 40, 87, 46, 14, -66, -46, -123, -28, 9, -110, 29, 75, 103, 90, 9, -74, 39, 118, 29, 32, -37, -51, -77, 118, 94, 93, -55, -113, 60, 46, 54, -110, 12, 25, -4, -45, -69, -19, 57, -35, -101, 36, 6, 25, -47, 0, 113, -12, -36, -104, 29, 56, -32, -25, -19, 13, 73, -124, -72, -40, 59, 63, -3, -35, -10, 19, -75, -118, 9, -86, 25, 7, 70, -12, 58, -27, -7, 21, 75, -45, 39, 14, 57, -30, -33, -53, 20, -7, 60, -5, -13, 7, -6, -140, -166, -6, 61, -12, 17, 55, -7, 61, -39, 80, -29, -31, -92, -39, 37, 101, 111, 141, -49, 9, -131, -107, -33, -24, 40, -27, -109, 11, -20, 91, -59, -119, 111, -45, -10, -24, 24, -14, -107, 33, -19, -95, -8, 17, -113, -112, -31, -126, -80, -61, -30, -93, 19, 13, 12, 4, -64, -17, 31}
, {133, -9, -52, 29, 55, -41, 110, -37, 102, 67, -93, 41, -82, -45, -19, 12, 117, -58, -6, 40, -3, -35, -66, -72, -112, 5, 16, -125, -15, -101, 52, -99, 60, 57, -21, -66, -9, -16, -75, -99, -88, -8, -51, -106, -15, -38, -32, -121, -1, -91, -38, -77, -13, 21, -127, -61, -69, 52, -6, -44, -43, -45, 33, -43, -8, -4, 120, -11, -3, 90, -27, -3, 76, 85, -33, 24, 49, -32, -9, -52, 73, -128, -130, -47, -41, 18, 57, -69, 120, 40, 95, 54, -142, 11, -92, 47, 93, -17, 149, -157, -9, 22, -58, 27, -34, -109, 42, 93, 123, 2, 11, -27, 66, 44, -83, -35, 38, 79, 48, 16, 48, 96, -53, -113, 49, 88, 61, -30, 149, 23, -23, -76, -13, 70, 36, -109, -10, -18, 57, -44, -73, 12, -12, -112, 32, -14, -92, 19, 63, -168, -3, 18, 35, 24, 21, 55, 40, 10, 37, -27, -36, 26, -17, -45, -34, -155, -1, -48, -139, 83, 14, -128, 3, -17, 82, 72, -16, -163, -58, -25, 49, 65, -149, 50, 66, 28, 40, -137, -124, 41, -12, 16, 16, 32, 82, 49, -73, -101, 67, 78, -42, -46, 122, -143, -29, 60, -67, 3, 54, -176, -75, 122, -18, 50, 34, 3, 55, -20, 31, 81, 57, 38, 4, 2, 101, 16, 92, -5, -142, -20, -41, -132, -64, 77, 36, -57, -31, 29, 86, 124, 3, 30, -2, -21, -106, 6, -124, -65, -47, 32, -51, -185, -21, 28, 37, 45}
, {-83, 3, -36, 37, -74, -50, -185, -104, 39, 85, 28, 95, -93, 15, 46, -232, -192, -257, -259, 39, -68, -21, -73, -18, -111, -43, 157, -146, -150, -34, -5, -119, 34, -152, -104, 121, -152, 107, 22, -64, 89, 79, 12, -53, 80, -156, 20, -102, -85, 11, -23, -22, 7, -82, -15, -28, 18, 45, 28, 40, -72, -79, 59, 58, 15, 58, 76, 82, 5, -17, -31, 85, -47, 120, 80, -43, 28, 31, -61, 34, 89, -61, -49, 50, -67, 249, -2, -120, 82, -112, -20, 16, -40, 11, -59, -114, 5, -38, 43, 75, -94, 7, -21, 77, 10, 6, -8, 13, 86, -52, -45, 66, 30, -85, -66, -99, 60, 103, -70, -24, -21, -58, 8, 17, -46, 4, -6, 30, -123, -38, -20, -25, -70, 41, 104, -133, -15, 21, 102, 2, -42, 53, 86, -85, -74, -177, -143, 80, 8, 4, -11, 32, -44, -2, 59, 69, -7, -13, -22, -5, 4, -32, 7, -10, -69, 42, 112, -13, -108, 94, 118, -49, 88, -16, -68, 95, -35, 20, 22, -27, -27, -76, -170, 56, 94, 135, 32, -41, -26, 20, 167, 28, -48, -11, -123, -63, 35, 46, 13, -42, -44, 113, 44, -20, 85, 129, -100, 10, -72, -13, 22, 108, -124, 11, 0, -3, 63, 30, 45, 25, 75, -10, -24, 20, -7, -17, 30, 101, -35, 38, -63, -144, 5, 151, 10, 46, -26, 52, 111, 6, -129, -15, 9, -59, -14, 34, -43, 60, 89, 52, 111, -3, -140, 55, 42, 20}
, {155, -20, 37, 114, 86, -80, 57, 204, -68, 24, 0, -186, 47, 5, 45, 103, -12, 48, -11, -55, 15, 45, -21, -25, 28, -4, 13, 35, -15, 44, 30, 15, -48, 33, 38, -67, 122, -141, -122, 20, -119, -111, -77, -34, -43, 5, -2, -47, -46, -78, -116, -48, -223, 16, 92, -50, -93, -55, -125, -92, 67, 98, -121, -72, 58, 9, 140, -33, 39, -62, 47, 69, -26, -14, -87, -36, 122, -36, -110, -68, -32, 7, -26, 62, 146, -57, 77, -119, 37, 15, 2, 31, -46, 74, 35, 27, -11, 60, 54, -61, -73, -92, -96, -26, -25, -41, -61, -36, 104, -1, -26, 39, -17, -100, -70, -99, -14, -55, -68, -129, 45, -15, -40, -145, 1, 16, -86, -74, -67, 110, 43, -73, -44, -56, -68, -43, 50, -115, 60, 13, 46, 56, -96, 45, -2, 79, 0, -31, 64, 22, 23, 30, 116, -27, -81, -67, 82, 116, -31, -19, 62, -18, 11, 60, 81, -41, -128, 127, 26, -7, -87, 56, -22, -8, 112, -141, 5, -13, 3, -39, 51, -10, 46, 13, -91, -19, -32, 61, 129, -29, -10, 4, 9, 17, -94, 25, 32, 114, -87, 36, 0, -27, -13, -6, -58, 35, -36, 31, 7, 78, 13, -132, -31, -37, -33, -31, -44, 57, 30, -30, 66, -97, -28, 44, -4, 66, -53, -15, 84, -20, -72, 32, 40, -151, -67, 4, -42, -81, -24, -26, 34, 37, -64, 32, 27, -72, 19, -56, 24, -90, -110, -7, -35, 49, 23, -195}
, {28, 85, 29, 14, -20, -16, -36, -145, 25, 6, 84, 43, 42, -81, -9, 4, 16, -44, -59, 26, -116, -84, 40, -25, -19, 8, -163, 41, -18, 77, 25, -7, 58, 28, -28, 12, 76, 45, 42, 65, 37, 22, 3, 43, 3, -55, 28, 94, 72, 53, 17, 18, 15, -8, 0, 72, 52, -68, 6, 43, -21, -33, -56, 60, 1, -65, -56, 33, -52, -11, 28, -58, -33, 47, 37, -31, 20, -67, 73, 17, -97, -33, 5, -76, 49, -47, 90, -5, -69, -56, 30, -45, 6, 135, -77, -82, 8, -75, -138, -35, -35, -103, -4, 52, -23, 0, -9, -7, 92, 115, -80, -6, 24, -64, -12, 7, -26, 96, 61, -14, -32, 63, 41, 57, 15, -162, -82, 35, 1, 36, -16, 33, 8, -135, 14, 57, -25, 27, -100, 61, -77, -52, 46, 152, -19, -28, 110, -22, 25, 109, -51, -62, 29, 1, -39, -43, -44, 37, 28, -67, -8, -33, 45, 2, -5, -17, 57, -29, 71, 9, -39, -56, -93, -91, 30, -6, 53, -30, -32, 33, -60, -7, -46, -4, -58, -74, -76, 77, -74, -85, -15, -7, -77, -118, 9, 52, -85, 41, -156, -45, 111, -71, -140, 31, 24, -43, 110, -55, 36, 92, -44, -118, -86, -18, -21, 168, -83, 95, -100, 8, -89, -67, -44, 10, 49, 139, -25, -80, 29, 46, -61, 81, -41, -51, -55, 150, -102, -36, -40, -7, 31, 126, 37, 0, 12, -104, 109, 91, -58, -150, -153, 97, 73, 30, 37, -136}
, {-123, 67, -5, -54, -100, 122, -13, -96, 16, 2, 53, 30, 11, -75, 107, 47, -77, -49, 30, 65, 1, 70, 30, 47, -30, -25, 28, -18, -3, -61, 9, -84, -9, 0, -56, 74, -26, -30, 2, -8, 77, 153, 111, 48, -7, -26, 22, -39, -50, -10, 0, 146, -27, 14, -6, 42, -44, 45, -20, 51, -15, -208, 49, 78, 48, 71, -97, 16, 68, -6, 73, -27, -52, -161, 58, -9, -128, 81, 5, 106, -53, 21, 40, -41, 28, 63, -163, -4, -54, 50, -142, -54, 81, -126, 100, -33, -8, 6, -31, 109, 63, 37, 19, -59, 16, 168, 83, -35, -168, -155, 50, 123, -114, 25, 46, 146, 22, -96, -51, 50, 9, -75, 19, 64, -24, -1, 21, 117, -2, -47, -6, -59, -56, 11, 36, -101, -47, -21, -11, -39, -26, 65, 58, -86, -84, 18, 13, 16, 1, 28, 19, -8, 19, 40, 32, -20, -35, -10, 12, -15, 35, -26, -23, -4, 1, -30, 61, -32, -17, -9, 111, 15, -45, -9, -31, -7, -101, -18, 59, 83, -16, -52, 33, -28, 27, 51, 6, -85, 54, 8, 32, -44, -68, -72, 57, 93, -13, 23, -13, -32, 26, 50, -76, -5, 88, 9, 66, 17, 46, -9, 39, -17, 11, 44, 1, -27, 100, 8, -79, -108, 15, 60, -38, -55, 49, -7, -26, -35, 14, 74, -40, 21, -48, -34, -27, -7, 48, 125, -99, -103, 22, 56, 33, -116, 48, -67, 45, 19, -15, 29, 15, -44, -107, -42, -35, 47}
, {-12, 62, -22, -44, -120, 63, -45, -55, 110, -20, 40, -86, 41, 37, -9, -38, -45, 104, 46, -102, 27, -89, -45, 58, 41, 102, -87, 69, 81, -81, -40, 83, -51, 55, -54, -26, 19, -70, 59, 12, -163, -24, 16, 3, -47, -73, 8, 28, 57, -72, -19, 98, -67, -17, -24, -26, 144, -23, 22, 76, 69, 79, -87, 94, -179, -41, -53, 2, -1, 32, -14, -127, -75, 96, 67, -22, -120, 38, 96, -20, -68, -44, -45, -8, -16, -113, 11, -34, -185, -16, -15, -113, -68, -15, -70, -65, -5, 68, -70, -37, 58, 74, 18, 84, 10, -43, -23, 44, 53, -46, -19, -131, 32, 23, 72, 15, 64, -8, 74, 49, -99, -5, 67, 88, -56, -100, 42, 15, 23, 64, 0, 138, 55, -45, -27, 70, 55, -84, -30, -11, -23, -106, 83, 56, 31, -20, 32, -36, -13, 93, 19, -14, 57, -109, -14, -142, 13, 90, 27, -52, 61, 5, 95, -26, -21, 10, -106, 83, -2, -81, -65, 48, 1, 40, 0, -76, 95, -47, -99, -26, -25, -83, 52, -32, -28, 6, -71, -7, -61, -113, -31, -78, 76, 42, 38, -60, 5, -54, 1, 7, 68, -141, -2, 34, 53, -91, 56, -20, -54, 88, -4, -112, 99, -111, 59, -133, -69, 45, -6, 47, 77, -24, 29, -17, -89, 27, 4, 54, -25, -100, -196, 33, 73, -70, -67, -21, -61, -108, -59, 75, -99, 53, -37, 99, 111, -1, -1, -38, -39, -89, -63, 2, 14, -9, -112, 56}
, {-65, 82, -28, -46, -99, 35, 14, 59, -72, -40, -16, 53, -81, -10, -22, -14, -113, 17, 14, 62, 99, -104, 2, 31, 6, -50, 16, 12, 57, 1, -72, -107, -41, 7, -1, 65, 6, -9, -35, 27, -12, 22, 42, -45, 21, 1, 95, -41, 14, -21, 17, 44, -51, 34, -28, 40, 43, 32, 67, 47, -49, -105, -31, 53, -90, -5, 62, -31, 17, -13, 11, -36, 14, -34, 76, 4, -95, 24, 46, 12, -5, 19, -75, 69, -59, -164, 56, 59, -1, 66, -44, -55, 13, -81, 24, 65, 9, 19, -81, -36, -30, -70, 27, -36, 6, 22, 84, 53, -36, -38, 3, -44, -91, 107, 32, 17, 58, 3, 33, 38, -41, -62, -32, 111, -2, 69, 57, 24, -17, 36, 65, 13, -78, -100, -2, -102, -84, -97, 119, 14, -16, -73, -5, 79, -35, -108, 1, -55, 5, -57, -83, 13, 1, 65, -82, -77, 10, -2, -60, -26, 45, 11, 59, 6, -38, -15, 26, 3, -19, -42, -33, -21, 44, -69, -79, 94, -11, -41, 30, 39, -44, -39, 1, -27, -21, -22, -37, 34, -62, -140, 39, -7, 18, 53, -67, -62, 27, 2, 73, -63, -71, -211, 9, 93, -56, -137, -203, 104, -13, 85, 3, 14, 5, -72, -108, -151, -90, -20, 5, 41, 81, 30, 180, -35, -13, 80, 8, 168, 12, -219, -117, -150, 19, 18, 17, -68, -89, -126, 47, 120, -139, -55, -161, -45, 41, -54, -65, -132, 121, -107, 7, -96, -84, 1, -223, -27}
, {162, 57, -28, 79, 41, 47, -15, 43, -89, 53, 88, 36, 47, 10, -33, 78, -14, -36, 29, -68, 59, 107, 29, -207, -72, -31, -4, 6, -83, -16, 32, 26, -32, -218, 18, 6, 57, -17, -30, -29, -15, -42, -171, -206, 110, -34, -47, 146, -11, 22, -103, -106, 6, 38, -99, -172, 73, 38, -3, -150, -124, 51, 18, 4, 70, 37, 58, 63, 68, -112, 0, -25, 16, -32, -136, 54, 61, -100, -63, 75, 75, 95, 75, -96, 80, -89, -63, -35, -51, -36, 84, 96, -3, 140, 1, -4, -83, 21, 12, -41, -10, -82, -138, 49, 25, -108, -76, 8, -5, -29, 0, 40, 131, -112, -162, -16, 57, 44, 30, -92, -12, -22, -120, -130, -27, -25, -61, -135, -62, 28, 43, -9, 199, -20, 22, -73, -34, -23, -87, 25, 5, 50, 54, 53, -90, -20, -57, 40, -98, 2, 83, 69, 65, -2, -92, 98, -24, 22, -16, -75, 34, 30, -89, 69, -62, 34, 30, 63, 69, 136, 47, -14, -78, -55, -4, -43, -56, 8, 70, 19, 2, -96, 27, 12, -83, -111, 43, 8, 78, 46, 16, -14, -39, -188, -68, -43, -44, -3, -132, -193, 52, -155, -88, -35, -154, 81, 31, -148, -105, 1, 68, -92, -3, -55, -137, 92, 17, -74, -92, -139, 81, -118, -30, 0, -96, -20, -15, 87, 48, 57, 40, 44, -31, -66, -186, 68, -101, -21, 44, 7, 19, 69, -33, -44, -67, -172, 37, 69, 52, -136, 24, 41, -21, 7, -31, -111}
, {-18, 112, -89, 11, -15, 26, -34, -197, 116, 52, -7, 15, -99, -21, -43, -1, -31, -94, 14, -54, 59, -105, -34, 10, 43, -51, -130, 78, 49, -9, 57, 33, -56, -17, 21, 51, -17, 2, -20, 25, -32, -9, -47, -54, 1, 38, -9, 171, 4, 57, 23, -8, -20, 90, -136, 46, -8, 72, 71, -30, -96, 6, 34, 73, -41, 41, -122, 29, -31, -3, -45, -19, 49, 42, 55, -114, 17, -90, 49, -66, -157, -81, -79, -30, -5, -23, -36, -32, 2, 3, -38, 102, -27, 42, 9, -42, 49, 28, -73, -55, 49, -18, -39, 10, -59, 46, 22, -59, 61, 11, 8, -8, 115, -87, 18, 49, -70, 18, 60, -46, 43, -19, 31, -11, -60, -27, 67, 27, 51, 4, -141, 46, -12, 45, 21, 58, 3, 3, -77, 9, -142, 2, -30, 43, -73, -10, -2, -189, 83, 118, -39, 33, -154, -39, 12, -98, -38, -27, 109, -234, -121, -142, -73, 60, -49, 103, 5, -35, 14, -66, -41, 11, -46, -20, -145, 0, -46, 13, 19, 82, 73, -23, 0, -57, -101, 52, -28, 137, -168, -126, 79, -8, 1, -31, -5, 75, -10, 23, -154, 18, 87, -26, -140, 46, -72, -62, 109, 37, 19, 4, 55, -208, -36, 83, 40, 136, 7, 55, -31, -38, -126, -19, -43, -5, 45, -4, 78, -96, 22, 66, 20, 84, 94, -88, -94, 39, -81, -97, -80, -84, 155, 87, -124, -66, 50, -97, 21, 24, -115, -145, -77, 65, 77, -36, 119, -86}
, {149, -48, 76, 47, 116, -101, -2, 169, -38, -13, -17, -162, 61, -35, -49, 66, 42, 52, -5, -58, -2, -70, 5, -66, 31, -103, 15, -2, 52, 30, 49, 60, -60, -86, 49, -88, 108, -182, -49, 38, -184, -152, -90, -31, -6, 46, -30, -33, 59, -106, -165, -11, -144, 35, 41, -162, -1, -123, -176, -147, 74, 14, -168, -124, -12, 26, 138, -29, -6, -23, 60, 27, -3, 13, -59, -85, 168, -14, -122, -68, 22, 24, 25, 4, 84, -42, 31, -72, 111, -7, 15, 87, 31, 81, 36, 6, 81, -19, 84, 21, -62, -72, -109, -58, -92, -100, -83, 6, 149, 87, -33, 10, 21, -78, -89, -83, -45, 27, -8, -48, 27, 27, -46, -107, 0, 32, -35, -97, -61, 25, 63, -3, 100, -82, -63, -38, -24, -128, -134, 20, 98, 18, -32, 20, 79, 95, 85, -60, 46, -11, -7, -60, 79, 57, -43, -79, 29, 22, 12, 36, 46, -37, 14, 22, 64, -56, -86, 47, 33, 32, -130, 12, 29, 64, 49, -24, 71, -9, -30, -128, -45, 24, 55, -8, -108, -48, -35, -2, 24, 102, -42, -104, -14, 29, -27, -19, 52, 4, -35, -17, -36, -121, -5, 15, -2, -53, -31, 61, -44, 101, 3, 2, 66, -8, 10, -43, 11, 106, 54, 0, 4, 21, 5, 36, 36, 72, -84, -86, -16, -37, -148, 46, 29, -114, -71, 28, 22, 26, 27, -25, 81, -16, 46, 71, -18, -109, 57, 21, -14, -6, -95, 41, -58, 7, -23, -72}
, {-81, 161, -51, -4, -13, 8, 57, -116, 81, -1, 9, 97, -96, -37, 40, -72, -83, -10, 21, -30, 90, -3, 35, 10, -7, 41, -13, -2, 28, -53, -30, -141, 20, -78, -28, 175, -65, 90, -17, -73, 95, 145, 57, 27, -84, -51, 115, -29, 9, -7, 25, 40, 99, -35, -40, 16, 75, 19, 8, 34, -80, -95, 135, 115, 42, 43, -24, 82, -12, -70, 7, -38, -16, -125, 110, 16, -41, -10, 39, 79, 111, -18, 42, -44, 130, -1, -69, 10, -54, -48, -172, 102, 62, -76, 39, 66, 57, 33, -7, 120, -25, 11, 97, 53, 15, 47, 6, -33, -46, 32, -24, 95, 135, -20, 24, 14, -100, -65, 61, -15, 33, -16, -33, 117, -8, 26, -136, 7, -82, -99, 72, 4, 93, 5, 189, 36, -81, 47, -4, -61, -10, 2, 47, -81, -47, -64, -15, -30, 8, -120, -51, 47, 110, 50, -10, -23, -35, 62, -141, 4, 40, 40, 20, -166, 33, -52, -18, 69, -31, -17, 72, 45, -6, 20, -130, -5, 14, -32, -9, 83, 4, 16, 47, -64, 66, 35, 13, -110, 67, -14, 11, -4, -117, 22, 100, 36, -3, 12, -18, 42, -44, 43, -3, 43, 79, -52, 47, 18, -64, -109, -135, 36, 8, -22, 43, 111, 33, 13, 46, -47, 12, 122, -158, -52, -10, 86, -56, -16, -112, 31, -54, -10, -56, -94, -68, 3, -35, 78, -32, -23, -37, 5, 142, -73, -79, 41, -8, 38, 7, 65, 46, -3, 47, -11, -17, 129}
, {-125, 70, -37, -50, -181, 54, 74, -71, 44, 39, 0, 58, -48, -34, 2, -48, -63, -15, -61, -22, 0, -72, 29, 54, 48, -112, -192, 91, 21, 11, 50, 28, 14, 14, 0, 44, -20, -62, 49, -15, -80, 24, -53, -24, 7, -1, -43, 54, -64, -9, -4, -8, -22, 48, -211, -10, 34, 44, 0, 11, -21, 23, 49, 3, 45, 2, 99, 50, -34, 40, -20, 97, -36, -68, 0, -47, 106, -78, -55, -69, -6, -5, -22, 13, -131, 60, 11, -35, -4, -45, -71, 28, -92, 15, 41, -28, 38, 15, -95, -19, 19, -66, -6, 59, -85, 71, 21, -42, 50, 27, 25, 73, 57, -92, 17, -4, -44, -30, -16, -43, 3, -68, -16, 23, -25, -67, -62, 1, 48, 8, -6, -53, -43, -62, -23, 109, 49, -31, -104, 4, -91, -43, -37, 61, -62, -38, -23, -55, 51, 101, 40, -46, -63, -61, 29, -91, 15, -30, 80, -119, -106, -83, 19, 14, -135, 55, -30, -87, 81, -38, -28, -108, -25, 76, -8, -149, 2, 38, 21, -69, 9, -94, -120, 22, -200, 3, -75, 103, -260, 105, -22, 36, -16, -156, 127, -38, 0, -16, -221, -22, 63, 104, -146, -39, 34, -25, 73, -52, -20, -51, 105, -153, 13, 130, 47, 19, 47, -16, 15, -72, -173, -37, -117, 63, 115, -87, 85, -151, 21, 110, -79, 151, -71, -108, -80, 39, -59, 50, -44, -174, 64, 32, -69, -28, -37, -93, 139, -39, -248, -75, -156, 91, 52, 30, 118, -147}
, {-168, -44, -21, -135, -82, 42, 7, -55, -48, -54, 6, -104, 18, -80, 49, -53, -69, 52, 73, -62, -43, -73, -7, 105, -55, -61, -98, -43, -5, 0, 22, 44, 4, 60, 39, -207, 21, -188, 90, 0, -77, 59, 37, 40, -55, -25, -110, -118, 48, 44, -130, 60, -36, -17, 3, -79, -51, -17, -4, -34, 55, -81, -84, 42, -49, 0, -82, -72, 42, 45, -8, -158, -10, -189, 89, 36, -85, 94, 61, -65, -94, 57, -51, -88, -140, -187, -212, 11, -109, -27, -75, -50, 42, -140, 51, -31, 16, 1, -28, -19, 95, -25, 158, -92, 81, 56, 74, 106, -160, -226, 43, 7, -108, 90, 107, 16, -16, -217, -73, 79, -6, -119, -20, 73, -16, -2, 39, 67, 78, -13, 27, -133, -103, 25, -200, -27, 37, -108, -48, -12, 64, 38, -6, -88, -32, 57, -12, -40, -4, -46, -90, -21, 101, 42, -22, -48, -21, -23, -66, 35, -7, -105, -165, -8, -8, -76, -51, -3, -43, -133, -202, 30, 25, -97, 9, 89, -89, -45, 35, -74, 8, 15, 13, -30, 30, 13, 26, 27, 44, -29, 15, -51, 51, -33, -2, 72, -5, -28, -40, 52, -72, -8, -8, -84, 39, -26, -97, 44, -15, -22, 86, 60, 50, -20, -7, -149, -29, -60, -10, 1, -43, 94, 8, -9, -106, -46, -18, -46, -84, 17, -44, 10, -38, -35, -80, -49, 23, 14, -64, -47, 39, -1, 59, -70, 70, 45, 22, -40, -57, 83, -56, -27, -101, -11, -20, 61}
, {-128, 13, 18, -13, -44, -16, -11, 61, -40, -28, -92, 54, -61, 8, 39, -9, -95, -25, -2, 0, 7, 20, -48, -25, -18, 68, -34, -50, 19, -43, 35, -32, 104, 5, 40, 72, 20, 88, -37, -16, 79, -38, -4, 1, -68, -59, -59, -74, -5, -19, -30, 80, 37, -135, 34, 5, -112, 53, 21, -2, -14, -75, 0, 16, -27, 17, -32, 47, 0, 33, 17, -59, 101, -109, -43, -16, -96, 22, 23, -23, 71, -121, -103, 7, -152, -60, -169, 74, 9, 7, -58, -58, -51, -99, 20, 12, 99, 45, -79, -56, -83, 11, 62, -71, 12, 29, 35, 100, -117, -172, 19, -28, -99, 113, 88, 27, 19, -86, 23, -17, 6, -85, 17, 20, -5, 36, 53, 9, 127, -1, -44, -127, 0, 39, 11, -30, -3, -126, 43, 19, -125, -69, -49, -63, -144, -10, -111, 4, 117, -26, -182, 16, -111, 18, 3, 39, -101, -111, -50, -79, -49, -99, -93, -27, -97, -81, -51, -161, 11, 16, -56, -118, 32, -73, 31, 121, -151, 44, 29, -15, 20, 14, -153, 23, 96, 63, 38, 15, -150, -72, 85, -58, 45, 87, -129, 20, -30, 12, 122, 84, -123, -5, 127, -2, -179, -94, -122, 134, -26, 30, 57, 68, 75, 53, -5, -191, -2, 34, -13, -60, 53, -50, 37, -13, -19, -121, 60, 73, -12, 15, -56, -119, -39, 46, -17, -189, -86, 31, -33, 75, -81, -94, -55, -15, -11, -65, -104, -133, -71, 14, -1, -32, -56, -22, -14, 91}
, {30, -59, -55, -46, -234, -80, 76, -96, 29, 58, 25, -39, -161, -116, -31, 15, -158, -10, -23, 4, 21, -125, 64, -16, -7, -91, -45, -51, 123, 8, 65, -11, -6, 142, -25, -38, -37, -28, -116, 2, -148, -13, -8, -61, -35, 35, -10, -75, 53, 34, -40, -2, -46, -1, -218, 5, -39, 104, -96, 47, -34, -20, 62, 25, 25, 48, 45, -7, -19, -60, -17, -21, 56, -165, -6, -47, -17, -30, 77, -50, 19, -29, 53, -122, -110, -5, -64, 87, 48, -31, -28, 62, 32, -36, 27, -27, 1, -1, -16, 93, 57, -17, 20, 84, -9, -4, -43, 3, -89, -50, 16, 51, 124, 102, 89, 28, -149, -233, 54, 60, 52, -32, -180, 40, 36, -16, -21, 87, 85, -78, 21, 119, 39, 42, -21, 60, -9, 49, -61, -23, 79, 18, 30, 7, -11, -16, 13, -75, 122, 43, -5, -46, 14, 31, 94, -22, -1, 14, -24, 83, 29, 55, 16, -93, 15, -71, -75, 26, -76, -46, 3, -50, 15, 33, -148, -44, 19, -58, 7, 75, 29, 21, 123, -129, -99, -93, -105, -17, 47, -2, -1, -22, 49, -86, 98, 3, 2, 5, -110, 93, 31, -20, 36, -164, 62, -67, 57, -22, 37, -39, -34, -55, 190, 22, 61, -36, -50, -120, 13, 33, -126, 81, -53, 6, -38, -13, 52, -67, -108, -24, -76, 8, -105, -53, -136, -135, -5, 83, -102, -56, -9, -18, -4, -79, -163, 2, -65, -109, -84, 87, -208, -147, -69, -13, 70, -120}
, {-129, 82, -22, -67, 22, 12, -27, 99, -86, -82, -17, 19, -29, 28, 76, 87, 38, 24, 51, 60, 105, 30, 6, -35, 6, 23, -108, -66, 24, -42, -36, 57, 2, 37, -40, -33, -53, 34, 6, -54, 37, 16, 13, 10, -17, 20, -80, -131, -12, -16, 18, 28, 57, -41, -66, 70, -74, 41, 72, 92, 29, -60, 110, 16, -18, -20, 83, 56, 47, 44, -31, 65, 33, -94, -39, 33, -89, -11, -13, -21, 51, -75, -61, -94, -33, 19, -7, 4, -59, -19, 53, -37, -60, -99, 11, 10, -46, 21, -79, -11, -96, 65, 17, -90, 35, -84, -34, 42, 29, -83, -56, -36, -72, 100, -26, 70, 90, -44, -20, 48, -19, 6, 47, -3, -51, 43, 56, 31, -56, 14, -35, -167, -45, 10, -62, 20, -28, -56, 302, 25, -148, -61, 16, -113, -116, -47, -186, 64, -20, -29, -121, 4, -119, 85, -7, -57, -62, -79, -70, -193, -100, -116, -87, -23, -100, 98, 29, -71, 4, -9, 6, -56, 121, -207, -48, 31, -76, 37, 32, -19, 92, 5, -162, 28, 75, 98, 40, -7, -127, -236, 111, 6, 86, 64, -76, -25, 8, 20, 30, 20, -69, 38, 12, -41, -103, 72, -64, 28, -51, 7, 141, -34, 75, 37, -134, -91, 117, -61, 37, 86, 41, 94, 33, 37, -64, -120, 5, 107, 36, -75, -6, -122, -39, 49, -1, -169, 23, -111, 3, 26, -44, -98, 48, 36, -18, -32, -209, 8, 38, -52, -18, -56, -61, -41, -110, 39}
, {-93, 56, 26, -50, -102, -25, 26, 44, 18, 1, -7, -4, -32, 50, -26, -43, -123, -13, 69, -52, 40, -36, 6, -12, -70, -23, -116, 63, 1, -9, -16, 18, 37, -8, -20, -23, -22, -2, -35, 38, -37, -38, 17, 55, 35, -13, 5, 50, -20, -6, 5, 30, 11, 36, -99, 29, 67, -23, -25, -26, -7, -10, 47, 78, 24, -70, -61, 48, -31, 17, -57, 106, -49, 62, -25, 1, -4, -33, 68, -64, -25, 6, -79, -51, -96, -23, 95, 70, 14, 16, -5, -11, -59, -115, -22, 71, 25, 93, -35, 9, -9, -45, 37, 98, 21, 64, 27, 27, -49, -42, -12, 40, 96, -31, 26, 58, -47, 21, 15, 35, -54, -5, 13, 34, 7, 3, -19, 7, -57, 39, -35, -64, -39, 3, 40, -3, -5, -7, 177, 145, -93, -23, -1, 96, -118, -18, -68, -82, 124, 145, 54, 40, -149, 14, -35, -63, 46, -101, -37, -53, -194, -113, -15, 63, -188, 104, -8, -166, 124, -146, -23, -87, 134, -28, 17, -77, -114, 86, 5, -82, 163, -71, -150, 39, -83, 118, 32, 22, -134, -78, 65, -78, 72, 73, 3, 72, 86, 31, 6, 63, -46, -44, -31, 10, -76, -108, 15, 14, 42, -37, 125, -73, -18, 119, -11, 3, -121, -36, 90, -58, -31, -61, 60, 86, 27, -45, 99, -83, 108, 98, -141, 63, -28, -60, -57, -96, -139, 35, 72, 38, 126, -63, -228, 23, 9, -39, 6, -127, -29, -49, -36, 28, 65, -67, -97, -17}
, {-73, -44, -53, 38, 93, 57, -1, 60, -89, 7, -3, -15, -199, -24, -74, 89, -53, -173, -67, 35, 0, -105, 4, 24, 7, -41, 133, -40, -59, -43, 60, -147, 1, 33, 12, -32, -14, 68, 4, -17, 33, -54, 23, -35, 62, -142, 31, -76, -81, -6, 30, 25, -14, 0, -21, -50, 48, 81, -17, 23, -64, -69, 59, 7, 63, 32, 14, -70, 103, 33, -15, 207, 78, -20, 8, -4, -53, 0, 49, 87, 123, 22, 99, 33, 60, 102, -142, 28, 33, -18, -58, 26, 5, -159, -28, 15, -36, 10, -11, 54, -45, 29, 27, -121, -39, 45, 30, 29, -67, 18, 25, 15, -119, 47, 13, -31, 114, -1, 11, 63, 92, 66, -32, -15, -112, 17, 34, 14, 62, -103, 26, -15, 73, 8, 2, 3, -25, -22, 16, -41, -11, 0, 34, -171, -63, -26, -127, -7, 95, -140, -94, 86, -53, 77, 23, 65, -13, -87, -30, -2, 38, -13, -161, -106, 12, -10, 53, -36, -45, 34, 22, -30, 22, 12, 17, 91, -56, 48, 14, 33, 1, -40, -89, -9, 95, 26, 74, -115, 70, -42, -4, 29, -38, 66, -14, 17, 75, -52, 120, -43, -88, -32, 120, -54, -42, 53, -13, -37, -56, -10, -66, 146, 94, -38, 41, -18, -1, 21, 3, -142, 36, 76, 25, -35, -14, 27, -8, 82, -78, -91, -26, -117, 29, -7, 39, -32, 105, 98, -38, 135, -57, -11, -14, -95, 1, 18, -96, 90, 101, 88, 160, -61, -63, -12, -159, 116}
, {169, 3, -48, 181, 60, -160, -15, 19, -19, -61, -39, 68, -2, -16, -117, 130, 84, -17, -100, 23, 19, -141, 40, -94, 3, -34, -86, -30, 61, 81, -17, 80, 34, -44, 48, -44, 55, -127, -70, -9, 20, -80, -105, -174, -46, -4, -29, 15, -13, -200, -105, -67, -120, 40, 38, -80, -29, -83, 8, -35, -64, -43, -220, 0, 0, -64, 41, 21, 7, 90, -3, -17, 33, 130, -119, -93, 148, -70, 15, -106, 82, 61, -130, 34, -41, -53, 148, -95, 6, 28, 122, 4, -71, 94, -16, -45, 15, -24, 58, -64, -66, 67, -79, 43, -63, -125, -166, 1, 20, 60, -49, -7, 72, -22, -208, -37, -18, 107, -23, -104, 16, -14, -44, -101, -13, 62, 35, -97, 125, 77, 6, -81, 79, 15, 20, -32, -24, -12, -23, 7, -28, 49, -11, 18, 33, -27, -30, -2, 33, -30, -36, 38, -36, 71, -9, -17, 28, -89, 37, -33, 9, 42, 14, 74, -22, -6, -42, -6, 3, 91, -49, 9, 20, 1, 72, 91, -50, -1, -19, -105, 50, -9, -58, 14, 93, -62, 59, 45, -21, 39, 18, -71, -71, 19, 5, 25, -71, -74, 65, -43, 30, -74, 0, -41, 13, 39, 70, 38, 17, -2, 0, 90, -78, 34, -45, 70, 0, 57, -48, 49, -50, -13, 21, -2, 65, 53, 17, 11, 75, -20, 25, 18, 4, 73, 40, 90, -52, -135, 39, -131, 43, -70, -72, 60, 65, -131, 3, 2, -97, -139, 1, 68, 131, 11, 94, -21}
, {-67, -11, -24, 16, -28, -36, 74, -28, 67, -25, 29, 92, -27, -5, -18, -23, -1, 1, 28, -2, -30, -80, 32, -24, 20, -18, -188, 16, -13, -2, 121, -5, 50, 64, 50, -32, 65, 47, -4, -10, -77, 35, 37, 42, 47, 12, 43, 94, -28, -16, -21, -12, -19, 37, -1, 74, 88, -26, 35, 50, -2, -4, 84, 27, 4, -27, -62, -3, 103, 25, -27, -39, 48, -41, -42, -39, 15, 37, -11, -69, -82, 10, 11, -31, -87, -46, -20, 57, -84, 70, -71, 8, -31, 37, 13, -68, -26, 5, -113, -10, 41, -17, 2, 16, 16, 73, -13, -27, 9, 15, 17, -3, -30, 17, 30, 65, 3, -12, -35, -19, 44, 11, -15, -2, 25, 0, 70, 56, 102, 29, -119, -47, 10, 71, -24, -53, 85, 2, -8, 4, -113, 69, -43, -105, -69, -38, -38, 0, -26, 36, -65, 5, -155, 53, 4, -1, -38, -117, -6, -65, -152, -153, -54, 29, -162, 105, 82, -61, 21, -85, -65, -95, -46, -18, 15, 137, -104, 28, 31, -24, 30, 48, -108, 66, -5, 54, 1, 5, -150, 26, 100, 34, -27, 55, -115, 141, -48, -102, 19, 68, -18, -12, 105, 0, -142, -98, 134, 101, 54, -50, 46, -53, 125, 101, -18, 52, -33, -16, 26, -126, -84, -53, -89, -52, 103, -153, 141, -108, 38, 81, -57, 72, 43, -29, -137, -17, 10, 103, 0, -151, 99, -67, -166, -110, 19, -93, -1, -15, -218, 1, -60, 25, -29, -46, -10, 4}
, {1, -16, 93, -9, 34, -78, -46, 51, -25, -22, -113, 22, -63, 52, 51, 69, -14, -32, 19, 35, -22, 68, -14, 6, 0, 56, -32, -130, -62, 139, 47, -11, 26, 67, 38, -28, 86, 40, -43, 6, -13, -107, 35, 43, -23, -3, -7, -113, 4, -34, -65, 46, -1, 11, 71, 6, -54, 38, 5, 55, 5, 18, 1, -69, -26, 6, 104, -1, 68, 133, -39, 49, -138, 68, -110, -41, 63, -5, -25, -30, -7, -147, -156, -13, -112, 72, 10, -77, 135, -1, 55, -84, -5, -104, -29, -67, 24, 42, 76, 4, 18, 102, -48, 3, 59, 71, -2, 48, -100, -64, -35, -89, 35, 34, -1, 12, 67, 60, 47, -45, 8, -68, 58, 35, 76, -4, 88, -18, 41, -65, -42, -62, -131, 36, 37, -225, 74, -36, 60, 40, -4, -41, 22, -166, 15, 54, -85, -20, 88, 40, 5, 11, -47, -7, -124, -33, -22, -38, -29, 16, -31, -128, -106, -54, -12, 6, -83, -16, -88, -106, -43, -1, 133, 33, 47, 40, 12, 57, -19, -68, 16, -28, -89, 10, 25, 110, -9, 33, -78, 14, 31, -49, 8, 174, -89, 76, 73, -82, 89, 66, -72, -176, -37, 69, 32, -63, -111, 148, 61, 37, -23, 68, -65, 110, -82, -59, 33, 109, -58, -26, 97, -62, 121, -70, -42, -15, -3, 67, 42, -28, -163, -56, 26, 105, 28, -119, -80, -152, -3, 42, -24, -189, 4, 5, -91, -63, -113, -81, -6, -167, -36, -61, 18, -63, -69, 36}
, {141, -30, 54, 185, -44, -83, 47, 132, 181, 40, -15, -5, -19, -41, -39, 18, 75, 6, -40, -28, 17, -21, 25, -15, -34, 11, -71, 42, -65, -7, 46, 38, -15, 67, -23, 27, -3, -36, 49, 25, -132, 57, 3, -2, 9, 80, 23, 168, 3, -5, -15, 28, -113, 9, -8, 47, 41, -168, -82, 42, 31, -28, -24, 39, 4, -59, -39, -12, 38, -32, -22, -15, 29, 116, -91, -31, 105, -59, 20, -77, -25, -13, 57, -120, -20, -101, -17, 26, 62, -116, 48, 18, 5, 84, -23, -88, -53, 42, -47, -19, 14, -22, -25, 121, -17, -11, -84, -38, 34, 90, -29, 1, 34, -101, 53, -22, -92, 99, 21, 3, 26, 79, -44, -24, 15, -97, -93, 28, 87, -44, -42, -3, 53, -24, -75, -86, 138, 16, -188, 36, 79, 1, -19, 25, 49, -22, 68, 31, 35, 29, -90, -57, -139, 52, 24, 10, 55, -39, 97, 141, -122, -103, -114, 21, -26, 18, 30, -18, 44, -47, -50, 4, -35, -32, 92, -12, -14, 49, 0, -68, 2, 48, 53, 38, -32, -7, 5, 20, -25, 115, 27, -24, -111, -188, 64, 15, 13, -42, -150, -11, 15, 12, -13, -67, -5, -49, 124, -99, 9, -1, -55, -9, 36, 94, 44, 100, -25, 68, -40, -115, -140, -29, -148, 9, 95, -53, 16, -132, 28, 35, 15, 81, -59, -136, 9, 88, 53, 89, -99, -154, 108, -52, -49, -9, 7, -92, 148, -10, -123, 11, -35, 88, 65, 19, 2, -124}
, {-27, -69, -31, -38, -5, 14, -44, -16, 5, 14, 14, -178, 22, 21, -12, -11, 1, 83, 45, -24, -10, 56, -16, 43, -16, 26, 42, -34, 52, -19, 46, 95, -64, -12, 16, -97, -8, -68, 43, 8, -135, 2, -21, 33, -10, -18, -91, -71, 19, -124, -71, 32, -127, -16, -19, -1, -29, 100, 75, -59, 18, 169, -88, 22, -144, -54, -4, -36, -27, -16, -27, -48, -49, -26, 60, -13, -55, -21, 49, 0, -61, -41, 34, 2, 45, -17, -7, 77, 9, 69, 57, -81, -52, -27, -57, 26, -53, 30, -3, -20, 19, 52, 85, 0, 63, -40, 5, 26, -11, -94, 10, -138, -98, 134, 54, -14, 133, -73, -40, 22, 38, -9, 94, -24, 54, -37, 122, -41, -147, 83, 28, 182, -7, 23, 73, -93, -38, -26, 20, -28, 76, -10, 62, -5, 55, 19, 8, 4, -73, -8, 17, -57, 104, -9, -15, -22, -35, 46, -4, 12, 83, 34, 88, -66, 24, -92, 93, 43, -37, 141, 88, 25, -62, -72, -8, -17, 55, -45, -40, 85, -47, 1, 38, 37, -14, -122, -27, -58, 65, -89, -121, -53, 51, -4, -46, 52, -16, -68, 32, -9, -50, -110, -20, 94, 46, -32, -71, -6, -51, 29, -55, -79, -6, -128, -37, 21, -89, 3, -20, -2, 122, -95, 54, -41, -179, 98, -3, -36, 6, -253, -42, -119, 48, -26, 75, 39, -71, -204, 17, 101, -154, 78, -34, 65, 130, -19, -48, -68, 105, -70, 21, -87, 58, -64, -126, 69}
, {-85, 28, -87, 33, -30, 57, 2, 46, -26, -6, 54, -83, -12, -87, 16, 20, -84, -48, 49, -24, 34, -47, 54, 64, 51, -27, -10, 13, 65, -73, -18, -45, -30, 26, 113, -131, 13, -102, 43, -30, -139, -24, 43, 18, 0, 8, -10, -93, -63, -85, -36, 62, -84, 11, 17, 4, -76, -12, 21, -15, 52, 4, -10, 86, -136, -57, 1, -44, -51, -145, -38, -112, 60, -14, 37, -23, -21, 51, 43, 70, -58, 66, 99, -47, 156, -132, 33, 168, -6, -31, -69, -2, 40, 118, -56, -61, -83, -24, 8, 64, -5, -42, 66, 44, -23, -24, -31, -66, 31, 165, -45, -34, -39, -8, 63, 20, -127, -86, -10, 13, -3, 27, -8, 30, 22, -13, -15, -12, 99, 71, -24, -88, 114, 51, -30, -25, -45, -51, 35, -29, 77, 104, 20, -30, -1, 22, -59, 57, -110, -146, -178, 67, 68, 113, -43, -78, 75, 1, 15, -7, 12, 93, -21, 57, -94, 52, -20, 36, -34, 89, 50, 107, -80, -99, 45, 44, -64, -5, 64, -17, -70, 12, 85, 57, 51, -30, 43, 49, 76, -59, 28, -64, 89, -42, -119, 18, -27, 21, -100, -67, -34, 1, 71, 11, 18, 4, -49, -9, -133, 11, 28, -32, 55, -89, -52, 42, -168, -70, -28, -6, 36, -71, -6, -43, -86, -43, -36, 44, -14, -72, -16, -11, -16, -90, -49, 22, -15, -49, 21, 24, -57, 92, 40, -16, 114, 72, -69, 49, -23, 64, 4, 38, -2, 16, -39, 3}
, {5, -16, -72, 35, 33, 63, 14, 4, 167, 21, -75, -96, -52, 11, -48, 26, -44, 104, -6, -78, 79, -45, -4, 51, 28, -92, 88, 64, 70, 58, 26, 19, -53, 57, 24, -187, 8, 3, -29, 38, -219, 5, -49, 28, 31, 26, 18, 101, 61, 11, -24, -17, -33, 38, -34, -63, 93, 76, -56, 22, 47, 43, 26, 3, -107, -203, 61, -151, -103, 68, -50, -114, 42, 47, 149, 29, 51, 27, 155, -109, -127, 74, 28, -4, -82, -96, -33, 67, 10, 149, 68, -51, -91, -24, -74, -27, -124, 27, 23, -8, 2, 59, 71, 17, 65, -50, -11, 4, -111, -23, -26, -45, -44, 95, 48, 29, 114, -65, -102, 55, 66, 102, 73, 22, 46, -3, -39, 82, 82, -36, -51, 113, 54, 39, -127, 22, -33, 45, -18, -22, -13, -19, -81, 30, 38, -4, -15, -3, -94, 61, 14, -18, -88, 0, 72, 70, 44, -68, -69, -32, -39, -61, -76, 3, -79, 0, 11, -29, -20, -93, 21, 32, -14, -43, -28, -1, -16, 5, -15, 63, -23, -21, 19, -26, 3, 8, 49, 1, -7, 6, 34, -120, 19, -6, 39, 42, -52, -111, -41, -17, -41, -3, 29, -40, 37, -199, 27, 16, -89, -141, -23, 93, 134, 40, 93, -34, 43, -202, -60, -11, -8, 129, 15, -146, -87, -52, 63, 9, -171, -66, -38, -34, -2, 2, -4, -132, 2, 48, 1, 10, -12, -60, -77, -1, -9, 33, -173, -106, -4, 75, 101, -73, -75, -190, 49, 53}
, {-24, 19, 28, 80, -69, 25, -53, -70, 70, -21, -7, -1, 10, 48, -41, -19, -70, 79, -6, 40, -52, -17, 22, -6, 4, 40, -108, 72, 34, 14, -23, 49, -55, 29, 65, -39, 86, 63, 13, 6, -117, -64, 25, -34, 44, 119, -30, 102, 26, 86, 39, -26, 14, 73, 28, 35, 112, -48, 36, 70, 12, 22, 32, -39, -86, 4, -2, -100, 118, 61, -29, -119, -67, 15, 73, -12, -40, -64, 61, -90, -174, 26, -102, 40, -98, -77, 29, 84, 33, 97, -81, -84, -104, 59, -90, -148, -36, 10, -83, -70, 97, 17, 73, 59, 36, -40, -3, -23, -57, -40, -10, -24, 31, 98, 89, 70, -19, -2, 5, -7, 49, -36, 34, 53, 13, -117, 38, 61, -16, -2, -132, 17, -60, 70, -48, 1, 81, 13, -12, 42, -80, 128, -15, -24, -124, -87, -66, -46, 6, 175, -77, -36, -133, -44, 6, -27, -43, -120, 18, -66, -139, -138, 9, 27, -46, 44, 85, -125, 46, -16, -58, -54, -44, 16, -16, 74, -65, 91, 44, 16, 16, -14, -111, 71, 36, 33, 19, 55, -216, -56, 45, 37, 27, -29, -67, 70, -51, -50, -6, 79, 46, 60, 8, 11, -5, -75, 100, -48, 12, -78, 33, -105, 156, 101, -38, 62, -21, -33, 21, -24, -36, -29, -97, -56, 35, -147, 85, -134, 6, 120, 15, 103, 65, -3, -33, -26, -70, 2, -27, -62, 79, 45, -126, -93, -21, -20, 4, -4, -214, 46, -127, 43, 52, -38, 94, -21}
, {29, 11, -69, -2, 29, 16, -79, 59, 55, -20, -17, 71, 98, -55, -81, -54, 41, -73, 34, -40, 5, 17, -85, -79, -54, -47, 39, -149, -56, -9, -19, 12, -106, -5, -30, 42, -46, 27, -34, 46, 0, -16, -80, -53, -77, -29, -19, -91, -8, 8, 38, -79, 82, 10, 9, -19, 58, 93, 70, -49, 8, -57, 106, -3, 81, -40, -10, 71, 42, 53, -3, 81, 70, 92, -80, 11, 15, -36, 72, 18, 1, -117, -85, -122, 26, 146, 13, -81, 128, 25, 130, 32, -64, -7, 99, 112, 20, -151, 89, -3, -6, 65, -38, 3, -30, -23, -122, -67, 71, 5, 21, -20, 21, 64, -94, -105, 66, 69, 74, -38, 104, -3, -32, 11, 31, 2, 60, 21, -41, -99, -15, -17, -68, -142, 22, 5, -32, -15, -57, 25, -2, 32, -3, -24, 53, -39, 31, 32, 4, 74, -1, 57, 77, -75, 78, 50, -56, 131, -33, 43, 62, 83, 70, -33, 66, -67, 14, 73, 55, 2, 90, 65, -45, 105, -92, -76, 75, -97, 34, 101, 27, 29, -31, -14, -100, -46, -53, -61, -18, 31, -94, 75, 50, -92, -28, -51, -68, 15, 7, -44, 19, 78, -63, -27, 18, 188, -84, -146, 46, -122, -53, 133, -63, -39, 68, 107, -103, 22, 18, 137, -51, -2, -12, 18, -95, -30, -8, 25, 23, 72, 74, -90, -68, 80, -2, 9, -43, 10, -8, -35, -151, 78, 99, 75, -60, 11, -9, 60, 40, 88, -89, -33, -6, 99, 81, 32}
, {-97, 98, -59, -8, 126, 38, -8, 12, -60, -52, 33, 93, -26, 19, -16, 9, 22, -18, -12, 17, 14, -42, -33, -29, -48, -7, -37, -49, -5, -23, -124, -90, 23, -53, -29, 152, 35, -2, -59, -72, 71, 88, 7, -59, -58, -96, 25, -2, -105, -58, -3, 44, 43, -120, 25, -31, 54, 70, 14, 95, -79, -79, 98, 40, 66, 37, -216, 90, -41, -54, 15, -40, -88, -8, 4, -87, -148, 5, -12, 132, 52, -91, 53, 24, 86, 82, -2, -6, -97, -29, -144, -31, 64, -152, 66, 12, 44, -11, -22, 58, -21, -61, -17, -131, -14, 49, -54, -46, -69, -1, 1, 51, 9, -120, -12, -12, -145, -45, -32, 41, -10, -83, 27, 6, -67, 24, -70, 64, -24, -73, 47, 13, 118, 21, 39, -150, -80, -72, -17, -115, 69, -21, 96, -91, -10, -40, 52, 16, -86, -159, 20, 43, 40, 160, 7, -50, -97, -23, -95, 45, 114, 117, 20, -106, 39, -83, 19, 90, -91, -52, 131, 48, 36, 20, -91, -93, 20, 36, 94, 163, -23, -26, 64, -71, 37, -11, 69, -68, 39, 0, 0, -35, -38, 39, 15, 9, 12, 41, 34, -24, 10, -62, -26, 35, 12, 61, 55, 40, -78, 12, -117, -8, -84, -70, -30, -10, 18, 115, -3, -186, 8, -35, 26, -95, 6, -31, -42, 69, -28, -67, -112, -111, 72, -83, -17, 55, 14, 89, -15, -9, -75, 63, 196, 39, 72, -28, 88, 47, 64, 42, 61, 36, 58, -13, -108, 28}
, {-242, -144, 52, -89, -84, 38, -51, -103, -62, 30, -66, 1, -14, -63, 55, -115, 6, -68, -51, -62, -95, 82, -27, -2, -37, -4, 49, -32, -52, -14, -64, -29, -25, 4, -27, 91, -36, 132, -15, 16, 67, 77, 24, 33, 61, -28, -75, -213, -1, -18, 52, 61, 54, -34, 115, -36, 18, 98, -16, 92, 15, -60, 19, 46, 95, 151, -218, 66, 66, -86, 16, -32, -115, -124, -7, -11, -109, -17, 3, 94, -80, -32, -28, -5, -57, 18, -172, -60, -137, -160, -156, -91, 45, -182, 119, -111, 32, 11, -69, 21, 100, -76, -48, 25, 55, 37, -5, -106, -234, -170, 61, 45, -94, -151, -103, 61, -33, -104, -53, -42, 86, -168, -30, 29, 35, -106, -167, 48, -9, -51, -104, -51, -97, 11, -22, 24, -42, 76, 26, -25, -138, -31, -30, -60, -192, -33, -31, 67, 62, -13, 27, 100, -89, -13, 6, 29, -31, -85, -108, -105, -102, -61, -83, -51, -93, 35, -6, -58, 3, 21, 98, -17, 14, 50, -82, -54, -126, -10, 22, 86, 23, -69, -122, -26, -11, 69, 10, 4, -106, 31, 31, 72, -24, 27, 44, 28, 20, -59, 16, 52, 79, 37, -123, -83, 138, -4, 25, 66, 73, 34, 65, 5, -109, 92, -80, -46, 85, 49, 13, -68, -49, -101, -31, 15, 72, -65, -88, 1, 18, 68, 22, 56, -46, 26, -77, 28, -40, 20, 8, -38, 7, -62, -86, -173, 9, -67, 18, -44, -97, -73, 21, 22, -26, -31, 79, -177}
, {-8, 46, -23, -12, -44, 62, -26, -39, 78, -49, -19, 17, -162, 68, 41, -13, -111, -70, -43, 33, -2, -91, -18, -59, -51, 10, -106, -121, -12, 16, -28, -185, 60, -32, -7, -46, -109, -28, -32, -39, -64, -15, 14, 5, -26, -95, -15, -40, -91, -3, 51, 20, -5, -6, -116, 45, -13, 45, -4, 8, -74, -16, 41, 0, 35, 21, -21, 77, 2, -96, 55, 94, -3, -40, -153, -64, -5, -65, 8, -17, -3, 6, -68, 3, -25, 129, 10, -17, -72, -93, -63, 81, 4, 12, 49, 1, -2, 2, 5, 17, 95, -110, 25, 179, -104, 4, -32, -62, -23, 53, 15, 72, 131, -231, -56, -6, -124, -41, -9, -89, -29, -181, -148, 19, -15, 132, -98, 33, -11, 105, -118, 38, -43, 20, 78, 11, 61, -21, 53, 71, -129, -45, -55, 27, -102, 121, -12, -153, 121, 4, -54, -80, -135, -24, 15, -100, 45, -60, 52, -91, -183, -123, -31, 10, -208, 67, -75, -105, -2, -156, 7, -20, 104, -38, -58, -75, -100, 43, 18, -35, 74, -68, -126, -112, -60, 86, 13, 60, -166, -125, 58, 83, 23, 62, 79, 39, 98, 9, -19, 86, 31, 5, -82, 37, -25, -143, -29, 63, -1, 38, 8, -48, -46, 67, 14, 70, -52, 52, 45, -57, -51, -105, -43, 52, 80, -6, -24, -106, 13, 79, -49, 58, -23, -33, 91, -14, -109, -55, -42, 29, 98, -51, -2, 47, 33, -65, 86, 2, -56, -196, 64, 48, 61, -17, 1, -106}
, {-88, 76, -107, -52, -134, 8, 36, -58, 110, 68, -34, -13, -83, -50, 39, -79, -65, -68, -151, -23, -32, 47, 45, -2, -56, 11, 6, -124, 24, -61, 32, -110, 54, 83, -35, 52, -89, 101, -83, -6, 72, 78, -56, -27, -28, 4, 27, -99, 27, 69, -1, 18, 100, 92, -23, -22, -40, -21, -38, 65, -31, -69, 46, 2, 10, 9, 68, 36, 40, 2, -44, 51, 83, -27, 50, 109, 15, -18, 13, 38, 1, -12, 89, -120, -42, 147, -5, 7, 19, 64, 55, -52, 30, -7, -40, 18, -9, -91, -11, 126, 38, 76, 25, 26, 46, 2, -56, 27, -8, -5, 7, 16, 122, -2, -25, 20, 43, -35, 84, 37, -6, 2, -37, 61, 7, -47, 4, 26, -39, 15, 7, 198, 26, 76, 113, 87, -69, 106, -13, -17, 33, -25, 1, 22, 5, -132, -39, 2, 9, 87, 2, 9, 24, 62, 57, 100, -35, 12, -46, -46, -15, -29, 15, -85, 17, 62, 94, 18, -49, -58, 124, 13, -24, 70, -144, -47, 5, -131, -53, 200, 69, 47, 6, -105, -9, -3, -43, -1, -32, -151, -4, 65, 19, -153, 110, 19, -70, 63, -81, 8, 61, 13, 43, -100, 63, 29, 90, -90, 21, -161, -73, 66, 152, 18, 50, 62, 69, -96, 7, 134, -153, 88, -67, 39, -50, -85, 98, 18, -75, 23, 75, 34, -51, -34, -23, -71, 70, 209, -135, -165, 38, -54, 148, -139, -78, 93, -62, -73, -2, 141, -46, -47, 11, -16, 79, 25}
, {-65, -28, -18, 44, 36, 67, -145, -82, 7, -18, 55, 24, -51, 38, 25, -76, -14, -69, 43, 53, -3, 5, -49, -31, -65, 3, 30, -13, -23, 0, -190, -103, -31, -96, -69, 109, -8, 86, -10, -38, 81, 101, 26, -6, -36, -49, 39, 67, 0, 5, 77, 8, 70, -32, 26, -24, 107, 55, 27, 29, -15, -53, 68, -14, -52, 93, -146, 72, -25, -136, 52, 1, -107, -91, 28, -75, -8, 17, -78, 104, -2, -10, -132, 75, 23, -53, -10, -55, -144, -5, -51, -67, 22, 36, 23, 101, 117, -22, -35, 33, -20, -113, -89, 36, -67, 54, -88, -49, -11, -15, 1, -31, 39, -98, -90, 47, -97, 67, -2, -4, -80, -148, -23, 43, -67, 45, -174, -51, 12, 62, 55, -24, 18, -34, 84, -52, -140, -85, 22, -36, 68, -111, 37, 27, -4, -27, 33, -126, 66, -143, -22, -6, 19, 58, -29, -72, -106, 12, -138, 18, 32, 30, 70, -39, 28, 29, 82, -10, -33, -28, 119, 18, -5, -19, -68, -43, 49, -98, 42, 168, -77, -51, 67, -145, -18, 13, -35, -8, -100, -8, -22, 23, -25, 72, -111, 10, 83, -31, 12, -33, 3, -237, -85, 30, -89, 13, -41, 88, -39, 131, -56, -7, -106, -80, -67, -49, -31, 78, -78, -118, 93, -18, 30, -108, 30, 115, -28, 107, -2, -23, -108, -77, 63, -70, -119, 94, -44, -40, -11, 51, -53, 55, 63, 21, -31, -106, 16, 60, 23, -133, 108, 84, 86, -14, -179, -30}
, {-71, -44, -7, 102, -1, 32, -29, 44, -92, 57, 70, -51, 33, 5, 67, 8, 59, -79, 137, 30, 33, 87, 6, 24, -9, 31, 139, 11, 19, -20, -147, -21, 22, 40, -46, 4, 76, -13, 45, -43, 29, -28, 19, 54, -29, -13, 5, 20, -45, -23, -53, 23, 30, -33, 93, -61, -16, 1, -38, -9, 0, -28, -40, -83, -21, -6, -137, 57, -46, -101, 53, -63, -73, -96, 30, 50, -64, 65, -147, 65, -51, 146, 99, 40, 67, -37, -94, -20, -42, -11, -100, -40, 48, 20, 12, 8, -44, 2, 39, -42, -102, 41, 74, -95, 16, -69, 56, 38, -52, 50, -16, -18, -84, -26, -14, -111, -39, -131, 34, -3, -61, -117, -45, 35, -83, 40, -62, 14, 24, 68, 134, 77, 75, -31, 10, 81, -129, -34, -68, -25, 30, -134, 37, 68, 3, 25, -14, 39, 12, 11, 9, -68, 126, 24, 9, 6, -91, 37, -1, 11, 50, 112, 87, -22, 53, -133, -47, -11, 3, 61, -77, -23, 26, -16, -45, -48, 41, -49, -146, 55, -116, 15, 137, -139, 24, -135, 17, -29, 162, -39, -190, -5, -25, -76, 2, 38, 31, 51, 26, -79, -47, 3, -18, -11, -4, 34, -20, -76, -61, 151, -85, -22, 23, -175, 0, -93, 39, -5, -9, -97, 66, 104, 66, -207, -69, 44, 28, -40, -28, -160, 49, -10, 12, -125, -26, 9, 18, 27, -132, 17, 7, 27, -14, -35, -50, 58, 45, 30, 35, -30, -16, 50, 82, -8, -104, -56}
, {90, -82, -28, 39, -31, -132, 35, 125, 75, 64, -86, -278, -48, -13, -8, 21, 52, 91, -69, -33, -56, -95, 20, 67, 49, -153, 63, 63, 14, 12, 40, 83, 67, -23, 34, -183, -48, 20, 37, 48, -184, -11, -5, 37, -2, 60, -40, -39, 12, 1, -124, -19, -123, 89, 17, 15, -96, -46, -209, -41, 47, 24, -92, -44, -45, -139, 198, -115, 93, 92, -79, -15, 164, -4, -1, 78, 121, -48, 3, -143, -107, -26, -31, 6, -28, -52, -32, 27, 195, 98, 144, -53, -22, -103, -138, 2, -136, 15, 73, 40, 5, 66, 18, -91, 43, -83, 24, 106, 26, -51, -38, -80, -98, 73, 90, 3, 26, 35, 89, 1, 100, -4, 62, -14, 42, 8, 71, 40, 61, -128, -51, -74, 11, 35, -45, -15, 19, 21, -23, -13, -7, 68, -57, -135, 7, 23, -4, 83, 29, -112, -38, -48, -31, 57, 50, -13, -54, 44, -6, 6, -4, -58, -180, -76, -51, -112, -8, 34, -62, -2, -39, -27, -13, 158, 11, 29, 104, 65, -17, -17, -2, 23, -80, 55, 64, 23, 30, -36, 50, 31, -28, -105, 102, -28, 20, -41, -52, -61, -2, 88, -110, 6, 59, -59, -98, -101, -22, -7, -30, -123, -86, 67, 122, -113, 68, 17, -46, -110, 54, -48, 5, 42, 84, 92, -72, 0, -18, 16, -238, -2, -4, -202, 13, 63, 57, -121, -1, 54, 25, 32, -104, -35, 91, 0, -42, -17, -231, 14, 60, 53, 39, -213, 16, -100, 104, 10}
, {-203, 6, 71, -48, -47, 68, -31, 9, -114, -22, 22, 88, 78, 30, 30, -19, 41, 25, 9, -55, -10, 4, -49, 17, -19, 55, 70, 31, -103, -1, -142, -46, -4, 30, -46, 140, 26, 45, -7, -12, 106, 103, 45, 9, -25, -70, -43, -220, 70, 64, -32, 46, 43, -107, 28, -29, 49, 15, 64, 70, 3, -118, 57, 33, 113, 173, -160, 57, -12, -149, 20, -38, -78, -169, -14, -54, -178, -28, -109, 102, 3, -12, 47, 44, -2, 21, -127, -68, -139, -103, -191, -7, 149, -170, 77, -9, 99, -62, -88, 18, 128, -32, -41, -53, -29, 167, -41, -74, -162, -102, 77, 122, -101, -173, -82, -3, -231, -108, -12, -28, -17, -151, -76, 23, -130, -48, -137, 31, -101, -52, 78, -27, -14, 35, -14, 41, -69, -104, 15, -57, 45, -6, -10, 33, 27, -3, 39, -59, 39, -155, -63, 72, 51, 51, -38, -12, 27, 13, -61, -7, 109, 12, -33, -33, 48, -15, -28, 24, -15, -8, 79, 42, -18, 6, -11, 42, -42, -9, 5, 20, -30, -20, 42, -37, 43, -13, -3, -3, 19, 85, -46, -13, -30, 6, 34, 57, 102, -66, 11, -67, -61, -21, -13, 0, 127, 25, 20, -9, -32, 65, -85, -35, -43, 50, 80, -17, 82, -7, -45, -134, 25, 45, -38, -92, 39, 19, 2, -16, -69, 19, 2, 16, 16, -95, -46, 14, -2, 59, -21, -44, -32, -9, 93, -114, 24, 24, 20, -35, 32, 36, 28, -18, -88, -20, -3, -76}
, {-96, 29, 6, 8, -27, 19, -24, 19, -43, 4, -34, -23, 24, -51, 79, 37, 13, -35, 31, 11, 35, 24, 26, -7, 40, -12, 5, 26, -40, 58, -45, 93, 46, 115, 48, -57, 19, -94, 70, 4, -29, 53, -31, -7, 2, 43, 0, -1, -25, -69, -34, 32, -13, 10, 52, -55, -66, 29, 16, -85, 35, 86, 26, 32, 27, 64, -154, 22, 56, -201, 101, -64, -55, -181, -9, -28, -31, 63, -40, 43, -6, 104, 219, -1, 215, 13, -14, -73, 3, -78, -87, 26, 27, 1, 76, -14, -69, 25, 36, 19, 12, -93, -28, -24, 72, 28, -19, -73, -96, 15, 9, 14, -77, -177, -48, -2, -104, -111, -85, -20, 19, -181, -41, -38, -91, -65, -99, -20, -106, 53, 132, -98, 69, 19, 3, -171, -9, -95, -38, 36, 95, -35, 47, -4, 13, 71, -58, 141, -72, -62, 44, -2, 13, 32, -100, -9, 116, -67, 18, 53, 49, 50, 5, 77, -56, 0, 8, -21, 34, 129, 14, 67, 36, -35, 98, -37, 0, 16, 59, -64, -85, 0, 40, 50, -17, -116, 13, 13, 117, 49, 47, 1, -43, -100, -64, 33, -11, 32, -120, -61, 35, 50, -32, -98, 14, 109, -81, -36, 2, 3, 75, -136, 40, -80, -70, 15, -64, -33, -2, -88, -13, -108, -151, -22, -16, -38, -43, -4, 41, -6, 96, 1, -40, -146, -66, 39, 29, 80, 22, -20, 0, 19, 66, -25, 110, 28, 33, 97, 27, 25, 28, 40, -94, 79, -117, -128}
, {18, 76, -82, -91, 27, 26, -20, -47, 80, 10, 4, 76, -145, -86, 2, 22, 17, 29, -23, 15, 62, 113, -140, -34, -95, -10, -68, -40, 40, -70, 28, -87, 33, 0, -61, 153, -41, 142, -8, -35, 13, 77, 18, -19, -107, -143, 70, 131, 83, 34, -8, -14, 101, 15, -22, -26, 18, 1, -9, 189, -12, -76, 140, -16, 20, 37, -57, 105, -13, 32, -36, -1, 1, 45, 95, -48, -91, -47, 43, 37, -27, -48, -72, -43, 20, 111, 16, -62, -72, -16, 29, 11, -23, -54, 28, 22, -13, -43, -43, 29, 44, 47, 59, 60, -46, 33, -45, -121, -12, 61, -23, -9, 134, 157, -37, 28, 60, 96, 106, -21, -23, 94, -40, 32, -84, 24, -36, 16, -72, 1, -124, 103, -6, 15, 61, 57, -99, 152, 99, -61, -100, -69, 5, 2, -143, -145, -101, -62, 46, 32, -23, 3, -17, 58, 66, -32, -20, -107, -149, -203, -64, -154, -4, -36, -83, 83, 67, -101, -12, -89, 133, -89, 21, 71, -145, -20, -95, -45, 19, 44, 28, 7, -52, -142, -31, 102, -23, -78, -135, -161, 58, -30, 36, -21, 135, -52, -16, 21, 48, 34, 49, 54, -114, 64, 169, -30, -3, -37, 9, -70, -71, 41, 9, 26, 82, 61, 56, 35, 22, 39, 9, 6, 20, -42, 23, -48, 88, 109, -28, 5, -29, 22, 4, 47, -19, -35, 1, 49, 66, 11, -3, 39, -3, -38, -53, -1, -17, 30, 22, 23, -1, 42, 71, -28, 3, 120}
, {136, 67, -213, 56, 58, 14, 8, -25, 20, -20, 49, -15, -66, -1, 6, -29, -3, -4, -67, 45, 49, 129, -48, -39, -123, -26, 11, -67, 0, 16, 49, -114, -30, -74, -90, 91, -113, 120, 16, -21, 36, 89, -15, -103, -57, -109, 46, 116, -51, 59, 52, -70, 36, -42, -156, 5, 197, -40, 20, 66, -130, -127, 103, 34, 62, 67, -25, 116, 68, -34, -73, 109, 67, 34, -59, -84, 16, -30, -17, 5, 69, -51, -57, -12, 61, 226, 32, -106, -60, -11, -97, 73, -21, -14, 47, -99, 49, -49, -51, 22, 38, 30, 44, 32, -91, -52, -44, -20, 49, 121, 62, 45, 182, -91, -82, -14, 37, 186, 59, -50, -34, -40, -89, 13, -52, 51, -20, -47, -125, -21, -27, -47, -113, 11, 191, 79, -44, 72, 2, 22, -86, 82, 16, -42, -39, -121, -66, -5, -47, 51, -27, 34, -19, 32, 0, 36, 47, -28, -46, -40, -66, -48, 84, -17, -58, 34, 22, 1, 4, -20, 86, -1, -26, 46, -87, -58, -118, -17, 73, 39, 39, 33, -96, -96, 40, 46, 29, -34, -62, -26, 41, 55, -10, 3, 99, -73, -6, 30, 33, -13, 61, 49, -38, -67, 47, 131, 72, -17, 13, -91, -42, 46, -54, 32, 8, 188, 6, 58, 9, 40, -35, -86, -27, 58, 24, -21, 83, 59, 49, 7, 71, 8, 7, 23, -50, 46, -8, -12, 97, -1, 59, 84, -17, -34, -68, 91, 74, 135, 11, 28, 89, 35, 196, 47, 116, 18}
, {213, -32, 0, 172, 45, 1, -10, 92, 32, -25, -2, -5, 83, -23, -67, 61, -41, -7, -75, -80, 17, 107, 43, -11, -69, -109, 88, -21, 36, -18, 94, -44, -71, 52, 45, -81, -23, 34, -129, -34, -71, 2, -66, -88, -4, 62, -9, 48, -24, 23, -15, -42, -6, 45, -3, -52, -67, -53, -101, -42, -40, -57, 71, -51, 83, -8, 138, -10, 46, -3, 26, 33, 36, 38, -12, 53, 82, 75, -23, 63, 52, -27, 185, -57, 115, 67, 79, -32, 33, 6, 135, 45, 64, 31, -45, 16, -38, -43, 105, 83, -187, -5, -64, -38, -5, -59, -62, -13, 19, 34, -104, 48, -20, -52, -65, -101, 57, 140, -53, -83, -28, 48, -117, -77, 45, -35, 37, -55, 16, -194, -2, 85, 67, -65, 73, -12, 82, 52, -11, -55, 34, 90, 40, -58, 25, -16, -20, 207, -58, -20, 42, -15, 44, -21, 34, 44, 58, -15, 9, 67, 23, 63, 9, -50, 1, -41, 72, 62, -111, 115, 18, -15, 26, 43, 22, -54, -27, -24, 16, -105, 18, 87, 54, -24, 18, -11, -3, 8, 102, 95, 3, -11, -51, -188, 36, -85, -33, -26, -101, 1, -1, 8, 69, -84, 33, 107, 64, -61, -21, -35, 17, 132, 11, -89, 30, 131, 91, -136, 1, 17, 19, -31, -56, 126, -52, 40, -123, 12, -5, 19, 29, -6, -59, 42, 27, 65, 42, 195, 35, -35, 32, -64, 78, 21, -137, 22, -71, 180, 30, 52, 6, -48, -97, 81, 27, -54}
, {-102, -69, -39, -15, -40, 122, -38, 64, -26, -11, -1, 40, 107, -90, 26, 75, -9, -129, -53, 37, -11, 107, 5, -23, -31, -88, 71, -156, -36, 22, 29, 31, 31, -103, 58, -18, 28, 58, 1, 45, 52, 36, -42, -100, 36, 42, -109, -181, 26, -54, 18, 2, 72, -23, -46, -35, 65, 41, -27, -45, -51, -168, 49, 112, 3, 13, 39, 28, -30, 48, -10, 57, -55, -16, -1, 8, 6, -45, -41, 64, 68, 3, 83, -10, 28, 125, -10, -109, 10, 59, -13, -74, -1, 20, 57, -92, -112, -35, -19, 20, 46, 87, -112, -31, -12, -1, -26, -55, -32, -82, 5, 60, -49, -5, -68, 10, 33, 55, -29, -80, 42, 69, -15, -70, 55, 7, 31, -30, -50, -241, 24, -31, 35, 15, 92, -56, -34, -112, -253, -175, 47, 56, 53, -132, 17, 54, -44, 50, -118, -64, 54, 24, 75, 27, 6, 76, 0, -48, -115, 72, 41, 18, -59, -152, 61, 65, 25, 35, -71, 39, -32, 24, -44, 90, -63, -50, 13, -130, -25, 66, -81, -114, 40, -32, -105, -127, -54, -103, 92, 38, -28, 59, 8, -46, -104, -69, -100, -87, 56, -156, 96, 24, -168, -27, -11, 124, -12, -98, -65, -24, 80, 33, -127, -12, -38, 30, -31, -31, -123, -71, 65, -79, 0, -35, 29, -118, -87, 100, 23, 16, 4, -87, -28, 133, 53, -10, 12, -85, 8, 76, -117, -62, 2, 69, -121, 30, -50, 11, 80, 92, 67, -4, -114, 73, -77, 116}
, {-126, -29, -25, -33, 47, 49, 54, 99, -122, -113, -32, -10, 25, 91, 45, -50, 34, 98, 76, -48, 59, -21, -83, 12, 19, -17, -105, -45, -55, -46, -98, 42, 31, -29, 1, 11, -22, -5, 18, -30, -10, -96, -14, 23, -35, 29, 0, -143, 47, -46, -7, 25, -60, -120, 54, -29, -86, 54, 16, -20, 44, -113, -4, -10, -22, -13, -60, -30, 42, 39, 5, -52, 33, -3, -16, 25, -187, 9, -77, 5, -24, -6, -171, 7, -56, -21, 38, 24, -81, 71, 60, -59, 31, -96, 20, 54, 12, 76, -61, -11, -109, 2, -42, -153, 48, 5, 77, 28, -31, -81, -12, 17, -162, 41, 60, -24, -37, -71, -3, -6, -63, -27, 27, -15, -58, 15, 21, 16, -55, 127, 55, -103, -136, 17, -28, -48, -26, -80, 179, 70, 26, -128, -44, -41, -79, -88, -76, -103, 52, -83, -79, -79, 20, -39, -27, -70, -58, -20, -14, -78, -57, -101, -11, -9, -52, -26, -123, 15, 37, -162, -80, -1, 86, -42, 4, 26, 24, 47, -31, -66, 123, 38, -100, -137, 15, 118, 73, 63, -28, -212, 67, -27, 51, 85, -107, -43, 65, 6, 9, 64, -85, -111, 9, -23, -49, -14, -178, 96, 11, 63, 2, -32, 52, -53, -100, -96, -34, 56, 4, 22, 69, 2, 99, 21, -36, -40, 49, 134, 49, -44, -228, -148, 8, 17, -65, -135, -111, -69, 42, 140, -31, -151, -81, -18, -71, -93, -27, -37, 9, -98, 13, 12, -78, -49, -116, -30}
, {-18, -5, 17, 46, 18, -128, 17, 62, -134, -25, -73, -138, 16, -94, -16, 32, -79, -24, 19, -50, -66, -54, 73, -68, 18, -125, 2, 17, -8, 36, 41, 67, -19, 9, 63, -252, 63, -103, -21, -25, -154, -86, -83, -58, -23, 38, -77, -32, -32, -8, -120, -21, -103, 41, -25, -154, -19, 2, -168, -162, -20, -45, -109, 27, -39, -29, 59, -82, 58, -20, -58, 2, -13, -22, -4, 72, 76, -4, -37, -144, -41, 141, -53, -2, 19, -34, -10, 19, 13, -45, 53, 43, -48, 45, 3, 31, -87, 5, -107, -47, 41, 68, -40, -81, 13, -89, 15, 160, 61, 3, -26, -78, 61, 99, 99, 8, -98, 22, 14, 35, -54, 26, -29, -12, 48, 7, 49, 5, -67, 58, -89, -140, -33, -10, 1, -116, 71, -145, 83, 27, -76, 37, 19, 59, -95, -1, -29, 8, 69, -3, -63, -49, -162, 48, -102, -4, 30, -59, 37, -57, -56, -128, -89, 25, -130, 84, -41, -30, 55, 63, -99, -12, 23, 42, -18, 87, -85, 100, 123, -150, -22, -72, -73, 53, 71, 30, 41, 72, -7, -15, 28, 24, 125, 0, -161, 56, -6, 33, -97, 3, -34, 74, -12, 39, -141, 40, -122, 29, -30, 41, 120, -157, 35, -8, -25, 52, -132, -25, 89, 35, 5, -101, -2, 70, -26, -120, 6, -8, 135, 81, 7, -3, 56, -62, -125, -101, 18, 70, 5, -57, 59, 83, -151, 22, 53, -50, 12, -48, -145, 90, -161, 2, -41, -9, 95, 20}
, {-12, -118, 23, 45, -64, 15, -54, 31, -18, 67, 0, -40, -9, -133, 44, 50, 49, -110, -74, -31, -180, -8, 108, 1, 35, 57, -14, 3, -113, 54, -54, -22, 7, 39, 59, -87, 21, 26, 49, 22, -51, 46, 56, 64, -33, -3, -52, -74, 23, 15, -17, -3, 45, 16, 38, -70, -60, -2, 31, -50, 42, 20, 12, 32, 55, 63, -9, -88, 57, -44, 4, -113, 8, -124, 26, 16, -14, 3, -45, -2, -87, -37, 123, -40, -15, -85, -223, 24, -27, -43, -24, -42, 34, -7, 20, -44, -24, 10, 63, 9, 46, -61, 18, -25, -28, 19, -8, 36, -89, -24, 59, 56, -83, 15, 18, -48, -56, -106, -3, 9, 44, 102, -65, -23, 8, -104, 14, 10, 23, 15, 81, -196, -17, 18, 5, -145, 89, -45, -168, -36, 99, -25, -49, -146, 16, 30, -31, 98, 65, -143, -120, 19, -120, 26, -9, 3, -21, -91, 38, 88, -58, 76, -287, 78, -77, -27, -122, 7, -17, 3, -87, -17, -122, -81, 58, 13, -41, 66, 1, -214, -19, 119, 25, 26, 11, -85, 26, -15, 9, 118, -18, 36, -163, -138, -30, 16, -16, -74, -63, 53, -182, 118, 17, -100, -13, 33, -27, -23, 26, 46, 44, -40, 22, -28, 71, -33, -66, -97, 69, -110, -69, -65, -164, 76, 33, -84, -11, -124, 5, 50, 47, -2, -25, -50, 88, -2, 23, 121, -97, -184, 32, -118, 82, -49, 20, -48, 113, -29, -128, 72, -79, 1, -168, 32, -29, 38}
, {-66, 23, -63, -6, 24, 41, 0, 71, -44, -111, 22, -2, -47, -37, 79, 3, -101, -37, -11, 22, 35, 42, -112, -28, -79, 16, 150, -14, 66, -61, -94, -52, 7, -20, -5, -4, 89, 6, -3, -116, 76, 52, 13, -49, -53, -82, -3, -66, -2, -63, -57, 68, 40, -122, -63, -8, -49, 115, -6, 15, -2, -116, 46, 0, -25, 29, -29, -4, -40, -54, 72, -21, -81, -64, 59, -23, -120, 84, -39, 13, 16, 108, -23, 56, 60, -88, -53, -62, -79, 59, -107, -62, 36, -42, 5, 33, -33, 62, 35, -8, -27, -46, 3, -147, 49, 20, 14, -36, -61, -42, -5, -19, -135, -15, 24, 26, 18, -116, -57, 1, -45, -113, 33, -39, -52, 61, -134, 36, 12, 35, 24, 12, -13, -36, 65, -138, -123, -75, 12, -22, 61, 41, 79, -60, -113, -100, -46, 19, -66, -111, 0, 45, 32, 85, 5, 40, -218, -3, -65, -88, 9, 104, 24, -104, 31, -8, 60, 11, -60, 9, 45, 2, -8, -16, -104, 43, 4, 11, -24, 123, -10, -122, 10, -50, 50, -5, -2, -84, 77, -131, 25, -96, -2, 15, -83, -32, 59, 35, 12, -61, -50, -83, -30, 54, -55, 99, -101, 33, -45, 30, -10, 74, 31, -97, -70, 0, 10, 59, 73, -70, 84, 6, 51, -98, -36, 25, -7, 201, 56, -198, -62, -141, 55, 45, 10, 13, -49, -117, 36, 162, -125, 45, -6, -16, 129, 38, -45, 1, 139, -31, 56, 5, -129, -35, -143, 55}
, {-268, -162, 57, -169, -189, -97, -14, 13, 82, 7, -44, 2, -17, 21, 25, 72, -95, -61, -74, 9, 22, -200, 77, 45, -2, -120, -94, -63, -60, 35, -3, -110, 14, 63, 54, 4, 7, -10, -17, 18, -171, 26, 25, 15, -47, 2, -62, -198, 49, -39, -13, 53, -149, 30, -165, 1, -112, 148, -65, 31, 20, -69, -52, 73, 68, 67, -42, 107, 45, 80, 43, 137, 24, -227, -126, 10, -108, -110, 27, -20, -33, -55, -31, -54, -212, 27, -224, 18, -108, -98, 16, -23, -8, -159, 108, 41, 25, 67, -25, -14, 6, -27, -16, 68, -17, 66, 44, -50, -191, -306, 39, 2, 5, -47, 43, 57, -51, -199, -24, 29, 10, -110, 46, 7, 27, -22, -35, 39, -88, -47, 6, -93, -54, -86, -1, 93, 37, -6, -63, -35, -36, -5, 7, 48, -114, 21, -64, -74, 58, 9, 64, -25, -11, -62, 3, -97, 22, 37, 64, -81, 5, -95, 70, -10, -129, -10, 11, 73, 83, -93, -14, -41, -13, 26, -23, -34, -77, -49, -52, -85, -70, -2, -157, -14, -147, 72, -16, 31, -62, -32, -4, -7, -21, -45, 77, 18, 90, 27, -43, 102, 43, -15, -113, -83, 100, 36, -63, -20, 49, -28, -89, -46, -25, 53, -25, -174, -60, 15, 89, 5, -121, -73, -68, 88, 93, 42, -11, -113, 24, 71, -94, 15, -131, -11, 18, 2, -80, -91, -42, -27, 50, -72, -154, 0, -36, 4, 24, -164, -66, -153, -131, 35, -57, 16, 6, -167}
, {-37, 51, -32, -99, -58, 106, 125, -200, 108, 8, 50, -85, -96, -110, -33, 12, -171, -103, -53, -29, 73, -118, 20, 32, 18, -54, -79, 93, 21, 29, 145, -92, -59, 28, 23, -5, 33, -90, -11, 60, -61, 26, 4, -4, -17, 70, -35, 124, -4, 9, -59, -26, -79, 97, -73, 27, 75, -20, -73, 16, -10, -71, 30, -20, -176, -36, -16, -131, -103, -53, -50, -163, 11, 43, 183, 29, -207, 16, 74, -41, -169, 58, 64, -91, 9, -76, 58, 58, -100, 19, -166, 42, 10, -15, -111, -71, -94, -14, -213, 34, 27, 42, 123, 14, -15, 16, -46, -69, -1, 142, -49, -101, -39, 31, 68, 96, -51, -44, -50, 61, -30, 71, -2, 78, -107, -66, 12, 62, 80, 12, -46, -37, 79, 58, 26, 58, 41, 83, 13, -60, -1, 93, -29, 55, -9, -48, -51, 60, 28, 141, -21, -29, -71, 96, 4, -80, -50, -38, -6, -49, -67, -81, -4, -23, -129, 1, 33, 30, 37, -80, -33, 14, -35, 11, -23, 18, -123, 58, 59, -43, -9, -3, 79, 64, -11, 17, -7, -15, 91, -24, -32, -126, 12, -88, 35, 25, -93, -69, -73, 48, 63, 58, 22, -34, 11, -88, 70, 7, -121, -8, -34, 41, 55, 50, -18, -64, -28, -57, 10, -35, -61, -82, -3, -146, -15, -83, 0, -47, -92, -35, 73, 76, 17, -94, -117, -68, 62, 61, -24, 46, 30, -37, -69, -34, 12, 47, -123, -8, 25, 15, -38, -183, -42, -35, 90, 16}
, {135, 53, -106, 74, 0, 60, 20, 165, -80, -96, 11, -69, 16, 22, -88, 35, 60, 64, 63, -88, 40, -41, -69, -43, -93, -120, 105, -23, 55, -101, -13, 79, -124, -65, 11, -79, -61, 49, -20, -58, -84, -34, -92, -51, -9, 6, -71, -34, -12, -58, -63, -108, -42, -60, -104, -54, -27, 73, -65, -31, -59, -47, 24, -79, 76, -21, 133, 7, -8, -21, 19, 23, 140, 49, 50, 67, 82, -8, 54, 18, 53, -41, -16, -34, 138, -6, 67, 7, 139, 124, 64, 52, 25, -1, 18, 81, -64, -14, -56, -48, -93, -6, 9, 10, 43, -149, -55, 6, 119, 64, -69, -29, 83, 103, -2, -87, 26, -4, -62, -100, -18, 106, -61, -69, 17, 42, 72, 0, -215, -48, 58, 17, 21, -136, -7, 27, 36, -30, 21, -27, 22, 45, 2, -42, -29, -63, -44, 42, -63, -39, -18, 53, 54, -8, -99, 104, -6, 39, -52, -49, 61, 100, 1, -76, -4, 62, 101, 57, -29, 30, 107, 53, -16, -73, -6, -74, -5, -65, 90, 38, -62, -13, -14, 73, -40, -116, -24, 5, 122, -63, 6, -14, 47, -75, -92, 22, -6, 56, -20, -33, 54, 124, -31, -1, -41, 139, 6, -5, 0, 18, 1, -24, -121, -24, 41, 62, -81, 9, 32, 107, -97, -89, 14, 77, -133, -157, 11, 17, -5, 42, 71, -59, 30, -12, -161, -79, -50, -1, 32, 30, -23, 100, 46, 36, -18, 53, -21, 14, 36, 166, 51, -22, -106, 73, 27, -36}
, {-63, -78, -16, -91, -7, -31, 71, 44, 6, 53, -25, -232, 11, 22, 10, -90, -49, 59, 36, -89, 61, -38, -2, 63, 80, -167, 15, -1, 41, 23, 45, 23, -15, 36, 9, -165, 11, 0, 49, 35, -282, -95, 3, 47, -23, 51, -18, -87, 43, 11, -27, 17, -74, -50, 11, -47, -36, 84, -104, 26, -8, -69, -75, -16, -16, -67, 62, -156, 35, 36, -58, -23, 160, -105, 23, 126, -119, -46, 34, -93, -114, -23, -22, 73, -119, -133, -146, 91, -74, 111, -13, -73, 7, -196, -33, 52, -41, 77, -83, -5, 64, 35, 96, -17, 122, 34, 65, 32, -113, -117, 70, 8, -123, 169, 82, -2, 92, -146, -18, 64, 33, -21, 9, 11, 24, -15, 42, 33, 95, -28, 120, 72, 86, 11, -59, 11, 7, 35, -69, 8, -60, -4, -24, 3, 13, 45, 63, 66, 4, -25, 57, -26, -47, 26, 9, 1, 78, 110, -56, 58, -17, -12, -86, 11, 61, -32, -106, 97, -39, -36, -133, 27, -44, 47, 67, 10, 121, -45, 1, -30, 5, -38, -25, 9, -10, 29, 9, 51, 16, -25, 24, -37, 118, -17, 10, 20, 26, -15, -4, 132, 6, -28, 27, 28, -72, -129, -3, 1, -33, -90, 14, 16, 157, 13, -26, -12, -15, -77, 32, 43, -36, 121, 35, 55, -134, -73, 2, -65, -170, 28, -26, -94, -31, -17, 25, -107, 26, 16, -59, -49, -5, -3, -31, -49, -92, 46, 5, -170, -69, 38, -40, -56, -146, -31, -82, 63}
, {-165, 85, -141, -62, 69, 76, 64, 56, -123, -92, -78, -23, 45, 81, 31, 14, -24, 123, 111, 82, 48, 17, -102, 16, -38, 73, 38, -53, -8, -77, -54, -1, 2, 58, -115, 25, 8, 3, -11, -17, 95, 9, 16, 106, -72, -41, -60, -66, 0, -125, -10, 67, -35, -100, 6, -47, -25, 144, 36, 10, -54, -57, 9, -63, -64, -14, -38, -29, -25, 33, 28, 22, -6, -44, -26, -3, -110, -15, -75, 53, 6, 6, 5, 121, 8, -94, 58, -13, -107, 74, -35, -53, 37, -22, -38, 42, 67, 60, 91, -26, -120, -52, -52, -201, 79, -55, 36, 11, -118, -37, 17, -48, -64, -67, 14, -63, 5, -134, -15, 30, -29, -12, 32, -7, 15, 54, -93, -55, -2, 58, 86, -20, -67, 51, -5, -40, -167, -6, 74, 48, 25, -129, -29, 13, -40, -208, -107, -10, 42, -135, 3, -10, -9, 42, -14, -21, -84, -36, 9, -159, 65, 150, 95, -10, 4, -65, 97, -25, 36, 96, 94, -14, 53, -75, -108, 16, -62, -16, 18, 157, 56, -9, -12, 11, 53, 32, 20, 58, 52, -191, 26, -4, 71, 45, 8, 31, 46, 19, 47, -36, -15, -207, 24, 21, -17, -27, -50, 47, 11, 77, -12, -32, 52, 30, -90, -1, 72, 82, 42, 3, 89, 3, 159, -42, 9, 14, 53, 167, 92, -139, -105, -111, 34, 31, 36, -75, -28, -120, 57, 164, -51, -2, -62, 43, 87, -97, -33, -24, 90, -109, 7, 62, -24, -14, -171, 6}
, {-172, -55, 11, -74, 42, 35, -38, 102, -98, 19, 2, -29, 105, -18, 26, 38, -45, -52, -93, -5, -51, -23, 65, 13, -3, 45, 106, -44, -19, -8, -86, 48, -22, -22, 90, -37, 87, 0, 0, 1, -36, -4, 43, 58, 7, -53, 23, -174, 2, -50, -33, 12, -9, -53, 60, -11, -42, -25, 20, 0, 36, -57, 35, 58, -2, 19, -8, -22, 80, 27, 30, 13, -57, -150, -6, -38, -185, 65, 5, 108, 62, 30, -3, 117, -32, 58, -141, -44, -81, 93, 4, -81, -6, -208, 76, 26, 6, -33, -11, -3, 16, -3, 71, -2, 157, 57, 84, -2, -143, -184, -26, 28, -123, 154, 7, -9, 75, -138, -53, 15, 36, -109, 92, -2, -1, -54, 209, 24, -10, -6, -41, -37, -76, 73, -40, -137, 7, -102, 7, 6, 108, -30, -5, -57, -6, 72, -9, 42, -56, -102, -105, -33, 1, -19, -68, 161, -12, -77, -31, 50, -86, 33, -88, -34, 75, -27, -14, 17, -72, 28, -29, 45, -100, -44, 15, 24, -64, -21, 66, 0, -2, -44, -31, 43, -29, -30, 90, -26, 60, -40, 22, -17, -16, 57, -142, 25, 23, -45, 144, -18, -53, -28, -5, -13, -36, -41, -173, 29, -23, 39, 42, 95, -6, -6, -7, -31, -74, 21, 26, -114, 180, -9, 73, -45, -3, 45, -77, 137, -32, -48, -52, -101, 85, 152, 122, -32, 14, -12, -4, 151, -106, -67, 44, 73, -16, -28, -8, 27, 157, 93, 100, 8, -184, -2, -74, 211}
, {-113, 117, -22, 65, 14, 16, -44, -119, -42, 29, 91, 27, 19, 25, 2, -29, -45, 33, -2, 8, -56, 143, 7, 2, 36, 45, -76, -18, -10, 18, 160, 20, 39, 14, -9, 38, 49, 98, 11, -26, 61, 44, -25, 10, 78, -5, -23, 108, -48, 25, 41, 3, 25, 13, 19, -12, 51, -15, 43, 0, -31, 2, 90, 62, 74, 14, -14, 75, 82, -72, -29, 92, 69, -42, -24, -97, 65, -36, -58, 52, 66, -56, -79, -15, 7, 116, 16, -65, -46, -105, -17, 57, 6, 48, 32, 5, 70, -35, -65, -58, 50, -23, -36, 131, -46, 13, -15, -111, 140, -63, 17, 67, 85, -152, -100, 62, -67, 27, -63, -64, 21, -22, -56, -51, -47, -32, -84, 4, -84, -29, -61, -77, -98, 17, -28, 111, 3, 6, 62, 35, -32, 59, 49, 32, -39, -101, -75, 39, -62, 52, 54, 39, 24, -1, 7, 0, 30, -41, 27, -85, -11, -103, 15, 28, -75, 92, 16, -37, -22, -37, 4, 18, -45, -96, -49, -59, -74, -4, 51, -23, 105, 28, -4, -25, 6, 40, 13, 55, -36, -10, 20, 42, 8, -7, 6, -32, -7, 120, -134, -62, 132, 222, -98, -90, 88, 39, 28, -26, 52, -16, 146, -84, -33, 14, -103, 56, 14, -13, 28, 132, -62, -99, -135, 111, -9, -192, -14, -25, 31, 25, 186, 128, -55, -98, -105, 13, 3, 5, -18, -102, 67, 36, -27, -102, 31, 28, 54, 68, -85, -96, -59, 20, 14, 44, 157, -117}
, {-60, -7, 35, 46, -94, -75, 40, -104, -179, 17, -17, -86, -24, 9, 9, 34, -64, -21, 31, -108, -39, -25, -25, 31, -2, -53, -124, -16, 4, -7, 18, 20, 0, 17, 57, -56, -33, -17, 73, 25, -88, -35, 4, -44, -49, 6, -6, 50, -40, -12, 85, 68, -88, 64, 9, -3, 39, 5, -61, 2, -44, -8, 35, -8, 44, -31, 67, 12, 79, -39, 12, -11, 33, -109, 32, 9, 72, -20, -114, 7, 12, -19, -61, -23, 35, -117, -18, -35, -9, -44, -85, 78, -33, 73, -5, -65, -27, -1, -141, 0, 67, -39, -12, -8, -14, 34, 20, 12, -3, 23, -6, 46, 66, 77, 17, 72, 54, 12, -32, -46, -10, -76, -59, -82, 48, -61, -38, -17, -104, -14, -131, -102, -116, -29, -56, 3, 84, -134, -54, 39, -10, 62, -101, 50, -53, -54, 38, 93, 52, 78, -14, -14, -61, -88, -46, 49, 29, -48, 10, -78, -110, -104, -75, 108, -11, 113, -77, 72, 37, 44, -126, 3, -102, 46, -16, -34, 18, 26, 153, -41, 116, 27, 57, 39, -130, 44, -75, 36, -166, -10, 54, -24, -39, -70, -15, 23, -4, 11, -195, 24, 59, 61, -28, -35, -69, 97, 21, -7, 15, 24, 101, -65, 91, 87, -25, 109, 31, -46, 55, -8, -130, -5, -143, 159, -46, -106, -80, -64, 54, 136, 139, 116, -75, -28, -73, -32, 51, 64, -79, -129, 158, 66, -31, -70, -15, 69, 65, 91, -184, -17, -32, 62, 54, 57, 113, -72}
, {116, 55, 46, -29, 81, -3, -19, 103, -126, -123, -38, 0, 56, -3, 15, 68, 43, -1, 88, 62, 2, -27, -24, -19, 7, 94, -14, -4, 42, 53, -81, 121, 87, -83, 14, 3, 142, -34, -5, 4, 57, -49, 62, 97, 6, -33, 20, 49, -29, 12, -16, 73, 1, -72, 65, 8, -92, -57, 33, -56, 51, 77, -18, 2, -89, -8, -73, 0, 34, -9, 19, -6, -134, -40, -184, 19, 13, -14, -26, -75, -46, -3, -85, 45, -65, 9, -23, -53, -55, 30, -76, -12, 33, 82, -19, 45, 20, 68, -28, -26, -100, 31, -7, -95, -18, 7, 22, 28, 28, -17, -21, -15, -120, -73, 49, 4, 10, 6, -13, -21, -5, -107, 75, -8, 57, 76, 13, 17, -50, 147, 47, -47, -81, 88, -59, -142, 78, -145, 169, 139, -3, -88, 24, -5, -21, -63, -42, -93, 34, -28, -49, -80, 6, -20, -52, -199, -152, 30, -31, -147, 45, -30, -9, 48, -56, 92, -75, 8, 114, -1, -72, -25, 92, -82, -23, 39, 63, 151, -25, -90, 20, -132, -2, -67, 118, 54, 9, 116, -42, -217, 129, -20, -49, 88, -208, 0, 71, -45, 92, 9, -81, -146, 7, 20, -91, -73, -129, 119, -46, 95, 52, -75, 25, 34, -79, -98, -25, 57, 11, -212, 28, -35, 86, -109, -89, 21, 50, 29, 104, -44, -149, -46, 28, -38, -46, -84, -154, -80, -56, 64, 37, -135, -74, -32, 70, -170, 30, -102, -12, -100, -25, -38, -143, -9, -108, -8}
, {7, -53, 16, -3, 65, 88, -19, 44, 50, 37, -52, -39, 105, -114, -4, 72, 50, -20, -43, -95, -3, 138, 53, -51, 43, 1, 91, -56, -99, 48, 97, 3, -48, 25, 25, -64, 34, 103, -26, 59, -40, -60, 13, 56, 9, 28, -14, -12, -56, 48, 14, 24, 34, -41, -36, 0, -5, 46, 30, -76, 23, -52, 71, -9, -75, -27, 111, -1, 60, 83, -86, 21, -112, 134, -36, 7, 70, -46, 31, 94, -68, -7, 13, -52, -53, -32, 5, -80, 1, -21, 86, -63, -23, 86, -44, 23, -76, -8, 24, -28, 22, 39, -19, -4, 141, -67, -67, -85, -45, -29, -89, -30, 8, 84, -97, 51, 88, 59, 7, -20, 30, 74, -46, -59, 68, -46, 91, 47, -81, 62, -18, 7, -14, 32, 41, -6, -2, -32, -89, -53, 59, 63, 78, -38, 41, -9, 20, 3, -76, 64, -1, 1, 28, 21, 81, -35, -25, -19, -68, 79, 13, 20, -50, -93, 26, -6, -36, 56, 49, 18, 45, 37, -119, 54, -19, -13, -21, -109, 7, 102, -16, -76, -14, 42, -21, -135, -24, -74, 15, 1, -59, 39, 13, -87, -32, -4, -107, -111, 10, -85, 199, 59, -104, -8, 99, 203, -11, -110, 28, -119, 28, -8, -46, -37, -61, 135, 1, 37, -125, 6, 108, -57, 77, 2, 5, -53, 6, 35, 44, 44, 75, -83, -50, 17, -11, 103, -134, -161, -19, 123, 21, 93, 23, 1, -119, -58, 6, 55, 85, -88, -12, 36, 131, 24, 75, 44}
, {-244, -78, 36, -86, -79, 50, -30, -44, -99, 1, -7, -163, 57, 64, 90, -77, 13, -8, -27, -11, -51, 11, 15, 51, 64, 54, -107, -47, -55, -31, -97, 69, -58, 6, 25, -181, -7, -60, 72, -20, -97, -45, 58, 4, -96, 28, -33, -124, -66, 32, -35, 1, -30, -36, 47, -16, -116, 5, 57, -16, -14, -48, -14, 56, 10, 3, -30, -42, 26, 46, -23, -105, -69, -137, -58, 109, -210, 12, -41, -49, -158, 13, 2, 72, -142, -5, -210, -17, -174, -20, -2, -88, -14, -234, 22, -162, -72, 6, -20, 1, -4, 51, -39, -41, 24, 48, 41, 6, -173, -246, -16, -20, -88, 136, -3, 29, 125, -186, -70, -21, -5, -22, 73, -41, 17, -111, 251, -9, -84, 13, -47, -179, -101, 39, -173, -36, 18, -81, 26, -27, -50, 28, 5, -72, -110, 5, -7, 125, 13, -43, -67, -45, -226, -66, -15, -65, 61, -84, -9, -33, -115, -126, -71, -1, -77, 31, 46, -84, -22, 10, -143, -54, -146, -150, -4, 11, -78, -43, 69, -108, 25, -112, 3, 64, 65, 39, 8, 1, -23, -122, 18, 34, 81, 22, -147, 2, 45, 52, -20, 18, -47, 35, 40, -20, -20, 74, -79, 57, 11, -5, 77, -18, 35, 2, 23, 10, 34, -57, 20, -5, -33, -18, -73, 42, -71, -127, -110, 77, -41, 58, -9, -105, -1, 62, 13, -59, 89, -25, -55, -30, -24, -3, 152, -22, -13, -75, 12, -36, -82, 60, -55, 9, -160, 4, -32, 87}
, {-96, -14, -3, -140, 13, 65, -50, 27, 61, -35, -9, 91, -23, -20, -64, -107, -46, 39, -7, 54, 20, -18, -46, -25, -92, -9, -1, -49, -1, -61, -129, -150, 52, -57, -69, 201, -62, 137, -39, -14, 34, 74, -35, -8, -19, -16, 50, -17, -37, 66, 70, 34, 42, -77, -40, -3, -21, 87, 8, 69, 14, -10, 155, 7, 12, 24, -61, 134, -23, -25, 6, -44, -8, -97, 98, -9, -69, -16, -85, 161, 8, -6, -50, 38, 133, -10, -15, -105, -91, 98, -106, -14, 59, -77, 70, 136, 117, 23, -36, 50, 16, -83, 0, 1, 7, 35, -11, -105, -25, -17, 61, 54, 96, -55, -3, 11, -40, -24, 55, -42, -30, 10, -31, 29, 15, 13, -200, 21, -198, -32, 91, 22, -19, 11, 176, -12, -273, 48, 23, -79, -54, -12, 111, -27, -36, -155, -5, 12, -57, -157, 4, -58, 69, 23, 42, -20, 28, 39, -64, -29, 90, 76, 108, -79, 62, -51, 68, 75, 1, 57, 119, 50, 56, 65, -100, -120, 109, -88, -4, 220, -104, -7, -23, -83, -61, 39, -50, -48, 7, -99, 5, -18, 35, 5, 55, -41, 104, 38, -43, -10, 56, -50, -83, 6, 41, 73, -8, 36, 21, 67, -42, 30, -90, -69, -43, -10, -115, 63, -32, -89, 74, 17, 38, -51, 29, 136, -93, 92, 5, -30, 21, -45, 7, -96, -35, 85, -76, -18, 8, 28, -105, 89, 83, 51, -21, 27, 51, 38, 154, 15, 52, 51, 59, 25, -72, -69}
, {89, 91, -35, 58, -57, -70, 54, 95, -70, 2, -6, -163, -95, -59, -173, 12, -34, 11, 77, 4, 118, -96, 32, -64, 5, -142, -107, 64, 44, 3, 34, -45, -8, -80, 6, -89, 21, -25, -42, -22, -19, -35, -136, -130, 53, -20, -12, 114, 24, -119, -127, -65, -129, 26, -21, -116, -2, -85, -84, -119, -27, -49, -114, -42, -26, -129, 94, 39, -104, -133, 31, 14, 17, 27, -65, -127, 91, -80, -120, -121, 52, 46, 83, -38, 47, -50, 38, 18, -23, 24, -40, 79, -48, 28, -109, 53, -48, 0, -31, -56, -88, -111, -128, -22, -129, -128, -23, 40, 65, 88, -138, 40, 162, -170, -139, -95, -158, 41, 19, -184, -47, -46, -182, -149, -4, 62, -190, -75, -28, 58, 34, -12, 3, -58, -30, 26, 35, 13, -53, 26, -70, 31, -17, 116, 31, -1, -1, -10, 7, -9, 0, -23, 56, 51, -36, 48, 61, 26, -10, -117, 39, 82, 83, 65, -20, 75, -23, -16, 30, 110, -17, 10, -12, 65, -1, -92, 49, 24, 9, -106, 10, 10, 82, -7, -124, -34, -3, 49, 84, 6, 32, 0, 8, -58, 5, -31, 29, -29, -219, 5, 85, 57, 9, 49, 7, 27, 111, -4, -71, 25, 103, -135, 39, 27, -50, 33, -86, 17, 27, 34, -91, 26, -24, -39, 20, -7, -74, 21, 49, 59, 35, 125, 50, -86, -117, 45, 28, 137, 83, -100, -2, 48, -6, 19, 25, -73, 8, 71, -77, 86, -1, 19, 74, -14, 121, -221}
, {-155, -114, 26, -50, -108, 76, -47, 82, -7, -24, -109, -123, 35, 10, 48, 10, -86, -18, 46, -85, 25, 79, 12, 8, 34, 52, -178, -38, -104, 6, -146, 3, 22, 45, -7, -31, 2, -86, -11, 15, -36, -65, 68, 79, 14, 54, -15, -151, 4, -35, 24, 78, -19, -119, 20, 44, -128, 67, 20, -40, -21, -5, -118, -55, 38, 111, -147, -19, 53, -98, 57, -52, -60, -239, -85, 9, -150, 63, -51, 4, 4, 27, 56, -12, -129, -9, -146, -67, -181, -121, -65, -39, 61, -177, 47, -49, 65, -16, -61, -72, 28, -34, -37, -18, -5, 27, 35, -37, -224, -183, 41, -4, -105, -99, -49, -9, -90, -202, -45, 10, -60, -183, -20, 27, -65, -34, 9, 6, 35, 1, 168, 46, -75, -23, -116, -64, 28, -52, 10, -20, -22, -128, 48, 2, 50, 58, 19, -56, -16, -24, 36, -61, -23, -86, -26, 13, -31, 57, -10, 27, 90, -13, -38, 58, 55, -68, -83, -45, -21, -137, -51, -7, -28, -53, 62, 45, -36, -27, -14, -17, -70, 15, 53, 26, 20, -7, 12, 8, 31, 13, -26, -41, 60, 22, -58, 27, 74, 12, -30, 29, -21, 48, -79, -6, 25, -30, -23, -38, 53, -23, -66, -79, 17, 33, 8, -237, 89, 32, -17, -95, 74, 34, 36, -43, 29, -50, -54, -105, -40, 5, -37, -12, -29, -144, -126, 6, 20, 63, -81, -11, 55, -101, 16, -61, 56, -30, 50, -95, 46, 13, 9, 1, -114, -29, -30, -224}
, {166, 39, 7, -30, -29, -75, 84, -3, 39, -135, 34, -61, 20, 56, -22, -52, -95, 47, 61, -24, 44, -161, -97, 45, -35, 45, -55, 30, 84, -10, -38, 11, -44, -13, -18, -52, -75, -87, -47, 28, -109, -14, 89, 31, 18, -12, 42, 119, 10, -20, 0, 5, -188, -24, -89, -23, 21, 55, 42, -70, 24, 174, -46, 31, -50, 45, 81, 37, -64, -117, 59, -22, 23, -5, -2, 3, 60, -14, -2, -72, -2, -38, -127, -11, 80, -115, 106, -43, 54, 30, -12, 44, -13, 56, -64, 0, 33, 17, -30, 47, 25, -75, -136, 143, 8, -37, -88, 20, 164, 179, -60, -33, 79, -46, 2, 5, -128, -9, -6, -13, -68, 14, -56, 59, -7, 14, -173, 27, -110, 8, 42, 53, -25, -108, 10, 92, -84, -29, -103, 2, -4, -151, -28, 70, 35, 30, 26, -160, 31, 0, 35, -73, 23, -91, -68, -35, 51, 43, 38, 61, 115, 68, 52, 67, -29, -115, -120, 35, 9, -58, -118, 48, -15, -39, 46, -53, 56, -89, -90, 59, -19, 63, 80, -94, -105, -113, -136, 63, 10, -77, -182, -29, 36, 41, 88, -57, 52, 5, -70, -26, 69, -64, -128, 116, -28, -70, 28, 6, -10, 115, 26, -127, -19, -7, 30, -78, -74, 14, -26, -55, 63, 62, 2, 26, -66, 90, -82, -35, -82, 15, -64, -63, 49, -83, -67, -91, -68, 1, -67, 22, -115, 33, 83, 7, -39, -36, 75, 8, -39, -55, -115, 50, -28, -16, 15, -104}
, {179, -221, 46, 71, 79, 12, -48, 145, -37, -39, -26, -95, 87, 28, 49, -91, -13, -2, -46, -65, -76, 96, 57, -17, 38, 0, 39, -37, -166, 69, 12, 24, 77, 75, 36, -28, 40, -12, -20, 20, -67, -35, -45, 11, -18, 31, -81, -68, -32, 18, -67, -10, 82, 25, 68, -68, -24, 12, 45, -71, 54, -29, 29, -65, -45, 38, -83, 28, -13, -22, 56, -63, -79, 8, -139, -39, 42, -10, -113, 50, 56, -56, 6, 90, -17, 63, 42, -94, -9, -13, 56, -15, 5, 102, 56, -12, 34, 7, 80, 46, -46, 9, -75, -42, 2, -13, -13, -17, -12, 47, -37, -51, -37, -44, -50, -27, -115, 102, 8, -79, 46, 28, 73, 7, 43, 29, -45, -45, -127, 21, 67, 14, -111, -31, 83, -79, -31, 27, -84, 5, -8, 22, 14, 44, -34, 28, 67, 30, -59, -67, 85, 13, 55, -35, -55, -19, 7, -24, 58, 53, 53, 77, 59, -32, -9, -71, -81, 34, -70, 11, -1, 95, -94, -39, -29, -70, -27, -133, -39, 105, -52, 1, 61, 54, -98, -216, -56, 2, 100, 72, -127, 39, 1, 18, 31, 62, 62, -5, -28, -132, -15, -50, -92, 12, 75, 111, -72, 6, -18, 91, 89, 37, -100, -51, -86, -23, -163, 160, -53, -35, 65, -174, 40, -133, 12, 108, -36, -22, 170, -105, -30, 44, -8, -26, -37, 124, -97, -81, -32, 46, 118, -36, -26, 90, 18, -55, 41, -41, 141, -83, 72, -37, 48, 96, -123, -48}
, {-116, -35, -5, -48, 9, -60, -25, 65, -29, -32, 10, -35, -84, 15, 49, -60, -75, -48, -1, 60, 16, -67, 19, -6, -2, 17, -66, -5, -60, 54, 9, -14, -4, -5, 49, -45, -17, 16, 19, -29, 57, -23, 11, 38, 77, -94, -97, -83, -34, -92, -30, 4, -77, 13, -6, 32, -40, 77, -30, 3, 4, -77, 35, 18, -36, 84, -32, 6, 13, 2, 1, -144, 7, -166, -91, -39, -58, 72, -32, -19, -36, 32, 8, -22, -75, -27, -155, 37, -40, 2, -165, -23, -6, -189, 37, -11, 67, -7, -22, -18, 31, -28, 49, -9, 20, 33, 71, 82, -136, -180, 47, 36, -93, 17, 58, 16, -72, -162, -44, -19, -77, -65, 29, 9, -15, 5, 10, -27, 160, -35, -67, -189, -10, 19, -34, -171, 97, -85, 112, -29, -47, -5, -3, -215, -51, 37, -60, 88, 42, -122, -148, 98, -208, 14, 5, 38, 53, -220, 79, -30, -41, -21, -201, 0, -59, -67, 39, -173, -16, 87, -40, -57, -37, -139, 30, 156, -123, 67, 48, -101, -22, 34, -160, 73, 17, 36, 71, 22, -44, 28, 7, -111, 44, 30, -99, 88, 24, -116, -3, 36, -183, -36, 113, -123, -98, -79, -38, 62, 60, 1, 2, 72, 58, 39, -12, -127, 15, -55, 15, -14, 8, 8, 6, -13, 69, -29, 6, -49, -98, 50, -53, -42, -32, -22, -15, -61, 51, 57, -66, 16, 22, -69, -107, -53, -44, -90, 10, -30, -22, 53, 40, -65, -130, -10, 22, 83}
, {2, -19, -41, -26, -89, -124, 8, -41, -28, 72, 2, -107, -106, -82, 28, -41, 19, 13, 46, -40, -20, -10, 8, 39, 18, -24, -57, -30, -72, -90, 55, -86, 30, 53, 26, -60, -21, -16, -27, -106, -96, -19, 9, 42, 14, -10, -20, 28, -35, 60, -85, -39, -69, 69, 61, -43, -39, -66, -26, 22, 33, -35, -141, -11, -46, -9, 28, -33, -14, -156, 51, -118, 29, -189, 4, 9, 31, 42, -62, -6, 20, 47, 2, -29, 45, -46, -163, 31, -61, -51, -80, 88, 79, 25, -6, -21, 78, 84, -105, -34, -12, -17, 38, -8, 29, 1, 9, 22, -2, -37, -25, 5, 50, -51, 24, -27, -50, -247, -12, 2, -35, -54, -95, 3, 62, 31, -16, -38, 63, -113, 102, -56, 54, 45, -21, -86, 49, 16, -211, -154, 1, -20, -76, -88, 16, 15, -128, 45, 1, -151, 64, 40, 2, 79, 33, -29, 17, 42, -18, 79, -29, 11, -114, -87, 31, -147, -71, 38, -63, -19, -82, -8, -79, 32, 65, -41, 24, -66, -61, -172, -65, 93, 14, 9, 16, -11, 13, -106, 45, 32, -55, -130, -143, -113, 33, -38, -72, -76, -83, 34, -107, -51, 52, -164, -27, -162, 36, 42, -103, -7, -215, 45, 20, -55, 75, 22, 0, -161, 41, -128, -81, 46, -178, -85, -74, 8, -9, -35, -148, 143, -21, 45, -17, -64, 35, -24, -1, 93, -239, -190, -2, -108, 47, -66, -86, -15, -116, -75, -50, 60, 20, 3, -71, -30, -48, 1}
, {79, -82, 15, 59, 53, -67, 22, -22, 71, 69, -55, -22, -37, -83, -25, -31, 50, 45, -8, 52, -28, -15, -51, -51, -27, 2, 67, -22, 76, -31, 18, 42, 26, -82, 36, -71, 30, 41, 4, 1, -9, 4, -94, -23, -35, -65, -33, -28, 34, -42, 18, -24, -37, 91, 43, -66, 25, 124, -74, 19, -7, 5, -41, -51, -37, -40, 107, -35, 58, 44, -1, -83, 0, 9, -41, -11, 46, 0, 99, -89, -75, -32, -39, -80, -147, -4, -43, 89, 88, 69, 45, -18, -101, 19, -56, 33, 51, 64, 163, -73, 111, 77, 36, -17, -11, -47, 27, 112, 31, -60, -21, -105, 21, 67, -17, 70, 9, 91, 6, 36, 65, 76, 72, -57, 71, 66, 88, 52, 118, -21, -22, -89, 16, 15, 21, -125, -38, -21, 31, -164, 15, -43, -52, -160, 25, 27, 45, 41, -20, -117, -92, 49, -118, 2, -21, 1, -40, -51, 5, 27, 31, 72, -101, -101, 71, -94, -16, -28, -120, -41, 71, 11, 69, 56, 110, 88, -26, 10, -54, 132, -74, 101, -74, 20, 35, -103, 22, -87, -21, 51, -60, -25, 37, 91, -8, 94, -83, -175, 130, -2, -138, -82, -8, -23, -56, -16, -47, 91, 19, -98, -118, 92, 55, 25, -7, -77, 38, 20, -162, 29, 66, 22, 42, -40, 58, 16, 17, -9, -186, -91, -80, -111, 16, -52, 22, -50, 1, -38, -19, 31, -60, -57, 27, -103, -33, -54, -194, -64, -3, 50, 155, -93, 52, -106, -80, 59}
, {97, -14, 18, 61, 38, -7, -116, 34, 67, 18, -59, -66, 143, -14, 21, -4, 6, 28, 46, 8, 44, 83, -1, 1, 65, 42, 139, 70, 5, -17, 50, 135, -30, -21, 45, 0, 22, -83, -11, 72, -46, -104, 2, 10, 0, 16, -40, 56, 66, 0, -31, 19, -6, 59, 70, 1, -29, -70, -20, -7, 32, 209, -64, -59, -74, -110, 102, -22, -20, 46, -65, -41, -40, 8, 8, -29, 41, 12, 92, 23, -64, -43, 24, 18, -126, 34, 35, -56, 122, 68, 123, -96, -33, -30, -38, 18, -74, -13, 85, 44, -21, 70, 24, -56, 46, -22, 41, -1, 9, -85, 3, -65, -99, 122, 57, 1, 114, -7, -11, -23, 2, 10, 86, 14, 3, 22, 108, -7, -114, -26, -25, 135, -2, -18, 26, -78, -64, -57, -41, -10, 46, -28, 60, 14, 29, -26, 4, -5, -67, -72, 55, -1, 149, 2, -54, -13, 22, 56, 63, -7, 28, 76, -22, -78, 23, -86, -6, -17, -23, 13, 46, 45, -6, -38, 38, -62, 97, -148, -83, 26, -9, 14, 1, -22, -45, -101, -30, -13, 73, -67, -55, 64, 62, 6, -102, 30, -2, 43, 35, -125, -31, -10, -38, 61, -29, -72, -92, -49, -84, -28, -113, 10, -61, -119, 42, 18, -53, 22, -1, 47, 62, -46, 108, -124, -83, 75, -40, -95, 70, -209, -45, -122, -48, 19, 100, 55, -24, -188, -2, 103, -152, -9, 8, 135, 31, 71, -64, -75, 163, 97, 25, -124, 39, -43, -94, 13}
, {105, -3, 30, 17, 44, -153, 27, 27, 87, 104, 37, -119, -67, -110, -32, 9, 27, -82, -28, 37, -35, -19, 4, -47, 21, 83, -39, 27, -40, 17, 116, -55, 92, -17, 36, -123, 23, 6, -14, -25, -91, -6, -45, -38, -27, 10, 12, 42, 54, 16, -117, -60, -166, 50, 11, -127, -35, -106, -90, -77, 26, -26, -154, 11, -80, -8, -2, 10, -1, 0, -13, 35, 107, -37, -91, -7, 116, 46, -28, -1, 94, 46, 113, -18, 166, -45, 41, 51, -3, 27, 58, -28, 33, 100, -5, -18, -68, 36, -16, -29, -46, -102, -9, 74, 38, -23, 43, -39, -87, 13, -51, 72, 28, 29, 24, -61, 27, 27, 20, 6, -61, 93, 21, -57, 5, 10, -16, -85, 31, -207, 41, -6, 54, -18, 8, 108, 51, -6, -227, -113, -6, 18, 6, -118, 87, 59, 29, 63, -100, -62, -60, -50, -53, 85, 50, 14, -6, -66, 2, 3, 37, 17, -130, 34, 87, -92, -52, 46, -24, 5, -60, 21, -8, 18, 77, -41, 61, 41, -58, -153, -64, 83, 59, -40, 5, -49, -12, -86, 54, 36, -18, -64, -186, -83, 84, -76, -183, -128, -4, -49, 22, 49, 66, -36, 14, -109, 98, -51, -106, -112, -114, 17, 53, 2, 7, -14, -5, -112, -147, -51, -47, 81, -42, -58, -50, 27, 4, -102, -85, 6, 68, -2, 15, -28, 19, -44, 33, 41, -93, -127, 13, -161, -26, -120, -83, 36, -148, 88, 18, 26, 22, -30, 32, -26, -176, 9}
, {-27, 53, -41, 62, -123, 68, 87, -119, 146, -38, 22, -10, -3, -69, 30, -92, -68, -19, -54, -44, 94, 3, 35, 37, 30, -76, -54, 70, 103, 30, 39, -39, -3, 173, 86, -31, 50, 3, -57, 25, -107, 70, -25, -20, 69, 83, 18, -6, 42, 43, -67, 23, -24, 12, -68, 5, -14, -28, -23, -49, 34, -114, 46, 7, 28, -42, -78, -152, -77, -40, -94, -203, 3, -7, 99, 106, -109, -17, 73, -54, -149, -23, 66, -71, -35, -32, -46, 67, -125, 44, -140, -90, 93, -46, -49, -163, -159, -30, -84, 50, -55, 60, 13, 54, 64, -1, 15, -21, -26, -8, -1, 10, -81, 138, 61, -20, 33, -55, -55, 73, -34, -11, -13, 44, 35, -73, 12, 90, 87, -67, -8, 82, 6, 120, 11, 78, -36, 42, -10, -5, -6, -64, 43, 78, -78, -65, -103, -90, -25, 30, 85, -89, 18, 58, 28, -15, -55, -20, 52, 3, -12, -4, 39, -56, -39, -15, -52, -44, -40, -16, -22, -12, -4, 7, -74, -72, 0, 5, -44, 12, -7, 16, 80, -51, 60, -1, -36, -50, 131, -50, 50, -117, 10, -21, 34, 33, -119, -35, 20, 80, 26, 16, -1, -128, 45, -163, 55, -45, -47, -45, -8, -43, 102, 54, 60, -144, 19, -75, -38, 10, 11, 26, -47, -76, -30, -105, 89, -15, -85, 24, -82, -27, -21, 37, -130, -122, 72, 44, 6, -18, 38, -105, -67, -203, -104, 66, -128, -67, -32, 5, -93, -148, -32, -181, -13, 36}
, {-40, -13, -2, -76, 111, 3, -175, 28, 10, 26, -38, 38, 86, -13, -32, -78, 29, -52, -18, -11, -17, 12, -11, 9, 2, 82, 77, -68, -109, 51, -6, -1, -42, -77, 8, 20, 14, 36, -24, 0, 72, 27, -16, 6, -38, -12, 34, -115, 8, -13, 18, -11, 79, -50, 69, 17, 39, -73, 43, -20, 57, -54, 5, -18, 43, 30, -105, -9, 1, -25, 46, -34, -73, -14, 48, -31, 17, 28, -70, -6, -55, 54, 56, -29, 67, 65, 8, -72, 56, -10, -30, -74, 28, -62, 94, -27, -39, -79, 125, 77, 8, 28, -17, -221, 42, -48, 59, -4, -36, -99, -45, 33, -103, 62, 15, -116, -2, 85, 11, 7, 101, 64, 56, 5, -76, 50, 83, -9, -37, -22, 17, 176, 7, -78, -29, 13, -118, -47, -87, -112, 59, 36, 25, 4, 72, 87, 16, 60, -48, -73, -13, -55, 83, -123, -32, 60, 8, 31, -107, 14, 113, 28, 76, -135, 98, -148, 3, 40, -119, 120, 41, 16, -120, -36, -28, -68, 121, -99, -92, 82, -40, -51, 25, 25, -141, -107, -22, -132, 70, -20, -161, -2, -1, -6, 42, 3, -23, 58, -38, -149, -54, 88, -128, -14, -29, 24, -90, -86, -35, -42, -62, 52, -1, -127, 15, 34, 79, 48, 25, -57, 19, 45, 58, -93, 26, -33, -64, -40, -30, -174, 34, -91, 38, -2, 22, -1, 21, 4, -156, 42, -38, -67, -22, 62, 43, 62, 27, 59, 127, 35, 81, -74, -23, 6, -67, -20}
, {-14, 153, -145, 21, -6, 25, 64, 56, -39, -27, -21, 63, -121, -1, -110, 68, 20, -28, 2, 41, 24, -33, -81, -59, -96, -61, 32, -104, 27, -116, 59, -190, -56, -19, 13, -12, -52, 9, -14, -16, 140, -43, -19, -177, 28, -99, 21, -108, -5, 6, 19, -103, 68, -60, -66, 3, 4, 134, -30, 43, -207, -174, 16, -53, 35, 20, 64, 44, -28, 34, -35, -14, 28, -35, 70, -20, -12, 0, -27, 20, 32, 38, -46, 35, 25, -3, 36, -104, 17, 7, 85, -26, -48, -33, -19, 5, -7, -61, -34, 68, -119, -67, -65, -22, -5, -85, 87, 5, 91, 41, 16, 20, 77, 186, -80, -62, 77, 28, 23, 12, 14, 90, -11, -55, 11, 16, -20, 33, 6, -27, -13, -121, -92, 54, -2, -73, -26, -53, 116, 39, -143, -19, -22, -83, -149, -92, -156, 85, 79, -98, -18, 120, 11, 59, 33, 68, 88, -82, -17, -72, 38, -19, -28, 16, -169, -15, -6, -180, -85, 125, 5, -67, -22, 2, 24, 145, -103, -18, 87, -60, 12, -19, -114, 29, 59, 30, 5, -53, -117, -106, -35, 45, 30, 50, 40, -13, -4, -17, 80, 53, 1, 39, 41, -29, -24, 60, -81, -75, 11, -105, -53, 49, -10, -48, 18, -9, 108, -15, 0, 102, -2, 86, 90, 17, 3, -39, 46, 67, -38, 10, 97, -127, 6, 70, 61, -112, 27, 25, 67, 83, -109, 20, 24, 76, -53, -38, -46, -33, 69, 132, 5, -64, -21, 0, -31, 79}
, {-204, 53, 117, -137, 50, 47, -31, -20, -148, -107, -27, 5, 39, 40, 7, 43, 8, -42, 48, 50, -34, 5, -80, 0, -31, 50, 30, -59, -55, 65, -140, -58, 63, -14, 36, 78, 50, -11, 17, 10, 74, 63, -38, 1, -27, -120, -46, -62, -40, -43, -5, 20, -25, -214, -28, -64, -39, 8, -20, 67, -12, -135, 1, 24, 67, 122, -155, 83, -100, -75, 63, 43, -55, -63, 13, -122, -133, -14, -123, 63, 102, -77, -45, 81, 27, 104, 41, -55, 2, -61, -228, 38, 76, -26, 85, 36, 128, -49, -81, -58, -45, -10, -73, -102, -128, 92, -117, -92, 0, 58, -14, 104, -111, -173, -147, -62, -171, 78, 27, -38, -5, -55, -59, 23, -151, 95, -178, -9, -95, -14, 39, -52, -104, 39, -43, -40, -86, -123, 9, -10, 76, -59, 12, -2, 17, -27, 0, -8, 82, -43, -22, 13, 55, 111, -81, 5, -46, -30, -67, 26, 5, 89, -10, 51, 62, -7, -35, 0, 7, -33, 42, 112, 10, -101, -10, -16, 24, -1, 29, 1, -39, -132, -34, -33, 36, -26, -21, -44, 69, 7, 13, -71, 15, 11, -126, 25, 44, 15, 92, -78, -38, -21, 13, -6, -122, -2, -14, 49, -7, 157, -31, 74, 34, -31, 0, 7, 20, 33, 21, -79, 60, -58, 9, -100, 51, 20, 16, 66, 69, -103, 14, -129, 49, -67, 5, -157, -47, -54, -12, 73, -51, -129, -10, -75, -40, 34, -149, -7, 66, 34, 70, -98, -214, -1, -82, 40}
, {-132, -200, 81, -59, -114, -38, 10, 93, -82, -2, -74, -124, -32, -19, 47, 57, -56, -64, 80, -142, -31, -136, 28, 65, 10, -52, -41, -111, -58, 50, -41, 23, 17, 43, 31, -164, 24, -130, 28, 66, -153, -11, 26, 39, -52, 66, -46, -156, 31, -95, -48, 18, -113, -106, 32, 16, -139, 78, -52, -49, 30, -28, -132, -18, -46, 78, 149, -2, -33, 91, 8, -20, 50, -140, -80, 95, -36, -22, -88, -31, 57, 4, -26, 10, -124, -1, -112, -7, -17, 86, 37, -63, 4, -108, 80, -16, 17, -14, -2, -18, -20, -34, -46, 10, 14, 26, -18, 2, -50, -148, 13, 53, -24, -15, 46, 28, 106, -159, 21, -1, 60, -127, 58, 12, 77, -60, 108, 13, -34, 4, 70, -159, -205, -98, -57, -48, 119, -94, -106, 24, -118, -10, -28, 31, 22, 35, 127, 21, 104, -30, 117, 32, 75, -80, -165, -11, 56, 79, 1, -7, 8, -18, -10, 64, 6, -1, -112, -4, 45, -45, -170, 0, 23, 31, 45, -62, 21, -48, -42, -59, -39, -38, -191, 16, -86, -31, -89, -9, -46, 38, -21, -45, 63, -31, -99, -7, 5, -16, -56, 99, 16, -25, -113, -5, 37, 34, -139, -52, 84, -10, 138, -83, -76, 15, 52, -103, -139, -27, 51, 12, -11, -80, 20, -31, 18, 90, -82, -83, -8, 5, -138, 38, 4, -53, -24, -51, -166, -97, -93, -29, -29, -25, -16, -15, -9, -34, 37, -146, -14, -53, -218, 47, -150, 13, -41, -67}
, {221, 44, 5, 130, 101, -27, 59, 33, 124, -5, 50, -3, 2, -97, -28, 75, 19, -13, 25, -25, 34, -25, 9, -133, -15, -78, 131, -76, -10, 67, 109, 12, -46, -76, 67, -110, 0, -123, -103, -42, 20, -13, -119, -116, -38, 6, -40, -8, 52, -97, -120, -59, -49, -29, -15, -153, -39, -38, -119, -123, -54, -63, -88, 58, 33, -137, 151, -49, -32, 57, -21, -10, -28, 31, -69, -79, 161, -23, 99, 4, -5, -1, 65, -60, 79, 6, 39, 55, 114, 46, 48, -1, -142, 99, -33, 16, -125, -29, 21, 17, -7, -30, 0, 77, 67, -87, -89, -57, 32, 57, -57, 3, 10, 50, -22, 8, 82, 7, -44, -35, 24, 38, -11, -151, -56, -12, 57, 51, 70, -74, 7, -52, 31, 30, 33, -131, 7, -122, -54, -7, 47, 97, -14, -25, -20, 50, 25, 19, -33, -29, -96, 45, -73, 48, -64, -1, -67, -81, 13, -4, 1, 34, -117, 44, 35, 27, 40, 40, -129, 22, -57, 31, -54, -39, 44, 41, 33, 27, 67, -50, -101, -21, -43, 62, 48, -75, 83, 74, 70, 85, -48, -54, -57, -25, -67, 53, -56, -132, 99, -65, 32, -26, 20, 16, -21, 1, 46, -59, -79, -134, -69, 39, -53, 15, -39, 138, 74, -5, -119, -27, 78, 15, 51, -108, 22, 62, -14, 150, -46, 66, -8, -27, -46, 24, 29, 74, 20, -134, 4, 25, 14, -19, 104, 65, -29, -21, -44, 164, -45, -3, 50, 31, 5, -60, 38, 73}
, {-121, 80, 5, -72, -11, 94, -99, -70, -32, -45, -44, 115, 97, -14, -12, -60, -7, -5, 21, 12, -81, 120, -91, 2, -31, -16, 44, 25, 45, -44, -72, -31, 69, 27, -42, 14, 27, 38, -57, 31, 160, 122, 18, 4, 9, -33, 27, -72, 17, -37, -17, 50, 70, -122, -23, -8, 24, -34, -1, 4, -22, -66, 52, 52, 20, 22, -208, 31, 28, 27, 26, -149, -101, -68, 1, 2, -89, 0, 142, 75, -134, 7, 32, -91, -19, -52, -92, -27, -193, -12, -87, -39, 7, -201, 25, 37, -15, -14, -64, 69, 41, -4, 71, 28, 31, -21, -99, -54, -175, -71, 57, 40, -33, 17, -9, 43, 40, -59, 4, 43, 7, 26, 10, 46, -47, -7, 8, 74, -116, 60, -1, 25, 0, -185, 37, -47, -163, 104, -127, -51, -58, -71, 74, 99, -54, -72, 2, -72, -99, 33, 84, 26, 170, -88, 47, 17, -104, 87, -72, -89, 59, 22, 64, -72, 65, 7, 86, 16, 18, -33, 134, 80, -12, 49, -70, -139, -11, 15, 27, 165, -24, -61, 10, -72, -87, -109, -140, -64, 106, -200, -61, 31, 12, -79, 115, -41, -6, 73, -137, -25, 100, -49, -95, 129, 56, 29, 65, -99, -60, 9, -57, -84, -62, -22, -17, 123, -35, 8, 12, 6, -16, 0, -97, -11, -36, -32, -144, -35, 21, -39, 13, 24, -77, -42, -39, 22, -108, -87, 31, 50, -30, 112, 93, 0, 14, 15, 23, 93, 81, -30, -80, 14, 23, -4, 22, -76}
, {131, 46, -80, 135, 30, -79, 36, 93, -30, -25, -35, -103, 42, 3, -230, 60, 65, 91, 22, -28, 98, -80, 29, -64, 43, -64, -30, 11, -40, 47, 44, 189, -29, -78, 63, -153, 19, 25, -125, -2, -175, -106, -138, -50, 24, 51, 11, 17, -32, -54, -160, -56, -137, 81, -10, 11, -29, -7, -94, -97, -28, 118, -13, -31, 24, -70, 104, -53, 73, 101, -116, 3, 34, 56, -21, 54, 102, -31, 38, -113, -19, 19, -2, -19, 64, -1, 70, -43, 116, 68, 108, 23, -176, 31, -175, 68, -89, -53, -42, -102, -58, 59, -30, 36, 58, -111, -14, -5, 31, 68, -98, -23, 100, 84, 38, 1, 104, 40, 10, -68, -38, 64, 15, -52, 99, 7, 82, 65, -87, 134, -138, -27, 62, 37, -72, -7, 39, 45, 84, 16, -50, 51, -79, -12, 25, 13, -16, -30, 47, -1, -18, 21, -46, -22, -17, -41, 19, 40, -89, 24, -117, -62, -96, -21, -41, 17, -50, 4, 7, -77, -66, -60, 69, 60, -23, -85, -25, 12, 102, -27, 54, -4, -52, 86, 14, 23, 29, 69, -116, -46, 50, -8, 33, 14, -50, 1, -43, 19, -4, 26, 74, 64, 9, -3, -46, 25, -61, -58, 29, -24, 92, -21, 13, 27, -3, -1, -44, -140, -5, 80, 6, -34, 2, 106, -16, -3, -21, 4, 51, -10, -7, 0, 53, 50, 55, -1, -15, -5, 140, -17, 25, 17, -22, 75, -76, -13, 18, 57, -132, 91, -61, -9, 17, 20, 94, 57}
, {-111, 4, -21, -33, 79, 89, -20, -1, -84, 11, 44, -12, -23, 24, -77, 38, 42, 23, -31, 18, 13, -106, -54, 14, -48, 22, 131, -13, 90, 2, -95, -94, 57, -80, 55, 49, 32, 50, -64, 12, 25, 0, 1, -2, 110, -67, 70, -106, -20, -54, 47, 16, 129, 2, -35, -14, -25, 111, 26, 43, -23, -15, 26, -1, 25, 18, -26, 28, -21, 68, -4, -17, -162, -66, -6, -5, -33, 0, 30, 23, 32, -67, -44, 91, 20, 95, 36, -60, -25, 105, -18, -122, -17, -19, 9, 66, -9, 42, 39, -51, -45, 87, -35, -26, 8, 12, -49, -7, 17, -48, 5, -8, -113, 82, -13, 26, 65, 4, 23, 2, -48, 8, 81, 17, -58, -9, 102, -17, -123, -14, 65, 19, -88, -4, 75, -180, 1, -55, -26, -4, 40, 33, -6, -6, -14, -19, -8, 25, -9, -56, 11, 18, 18, 29, -69, 7, -40, 69, -112, 36, 31, 25, 146, -1, 48, -92, -15, 7, -8, -71, 118, 0, 81, -13, -4, -127, 56, -68, -10, 58, -36, -94, -7, 28, -2, -62, -55, -43, 52, -80, 6, -29, 16, 109, -173, 46, 44, -39, 213, -64, -42, -123, -38, 90, -23, -45, -116, -43, -81, 5, -17, 56, -141, -65, -7, 28, -30, 119, 27, -47, 195, -31, 195, -160, -114, 111, 29, 44, -14, -202, -93, -188, 42, 75, 117, -80, -58, -181, 109, 160, -234, -61, 45, 43, -16, -70, -25, 21, 117, -48, 116, -66, -91, -11, -206, 135}
, {-139, 65, -64, 115, 33, -73, 47, -59, -37, 72, -64, -2, -18, 1, 46, 7, -5, 15, -28, 30, -21, 22, -1, 38, 3, 34, 114, -16, -30, 23, 63, 4, 63, -25, 18, -28, -91, 99, 48, -19, 53, -23, 8, 48, 38, -38, 20, -19, -44, -17, 23, -18, -53, -88, -2, 37, -19, 122, 41, 36, 13, -53, 7, 22, -46, 10, 13, -15, -38, 57, -79, -115, -6, -53, 21, 10, -54, 45, 67, 42, -65, 118, 130, -3, 27, -153, -167, 15, 21, -9, -119, -33, 4, -139, 20, -80, -42, 32, 45, 28, -56, -8, 16, -70, 42, -11, 19, 39, -53, 16, 25, 10, -80, 56, 83, -12, 85, -4, -63, 8, 18, 66, 26, 27, -38, -45, 33, -3, 94, 23, 18, -136, -11, -21, -44, -52, -4, 1, 81, 25, -123, -31, 82, -44, -105, -72, -167, 34, -13, -89, -73, 105, -101, 89, 85, 24, -107, -95, -35, -86, -49, -37, -78, -15, -37, -3, 16, -36, 18, 106, 20, -70, 17, -21, -56, 138, -95, 64, 79, 35, 0, 55, -97, 84, 150, 69, 63, 7, -6, -18, 153, -50, 46, -28, -12, -73, 47, -19, 79, -17, -104, -28, 101, -76, -31, -129, -79, 14, -84, -4, -16, 39, 49, -110, -7, -74, -25, -51, 114, -110, 14, 25, -59, -17, 5, -79, 2, 39, -93, -45, 34, -78, 19, 22, 27, -88, -9, 174, 36, 0, -74, -24, 22, 55, 7, 54, -111, -29, 116, 122, 121, -132, -37, -35, -21, 73}
, {128, -85, -11, 50, 102, -161, -22, 65, 90, 6, -51, -39, 27, -104, -32, 63, 27, -108, -72, -83, -23, 126, 55, -186, -32, -44, 50, -99, -1, 65, 68, -41, -35, 9, 23, -61, -35, -24, -58, -9, -75, -50, -85, -78, -112, -3, -94, 22, -6, 80, -135, -75, 61, 65, 54, -71, -63, -108, -74, 1, 91, 55, -4, 21, 59, -104, 112, -37, 76, 102, -86, 30, 2, 58, 38, -79, 50, -35, 104, -20, 38, -123, -34, -132, -126, 16, 17, -76, 132, 5, 29, 17, -241, 30, 14, -58, -7, -88, 140, -10, 3, 40, -26, 43, -36, -26, -63, 42, 76, -78, -135, -7, -203, -23, -110, -41, 17, 97, 61, -167, 141, 45, -1, -77, -14, 3, 92, 25, 77, -163, 57, -35, 67, -45, -71, -43, 27, 19, -174, -124, 17, 4, -34, -148, 66, -49, -4, -58, 51, -1, 11, -27, 29, -93, 38, -12, 47, 48, -86, 123, 34, -5, -77, -84, -30, -14, -46, -20, -146, -39, -43, 3, -13, 100, -51, -31, 12, -193, -61, -33, -32, 53, 18, -39, -36, 7, -66, -141, -31, 40, -148, -27, -29, -20, 50, 1, -86, -177, -4, -35, 2, -35, 3, -2, 52, -27, -61, -55, 25, -76, -166, 67, -77, -28, 3, 0, -11, 2, -163, -20, 60, 11, 0, -66, 104, 31, 10, -66, -123, -53, -68, -68, -52, 60, -18, 71, 0, -96, 36, 19, -3, -73, 47, 72, -122, -45, -42, 30, 26, -57, 40, -40, 29, 52, -55, -6}
, {-114, 27, 60, 78, -110, -16, -42, 176, 20, 26, -76, 41, 8, -77, 2, 68, -63, -30, -91, -45, -141, 107, 17, 51, 17, 39, 57, -15, -84, 68, -18, -36, 60, 47, 17, 50, 47, 67, -42, 53, 111, 59, 18, 78, -77, 71, -41, -68, 65, 22, 6, 57, 108, -14, 14, -46, -111, -104, 20, 34, 42, -69, 40, -8, 68, -12, -7, 25, 66, 2, 0, 52, 4, -252, 38, 31, -93, -12, 16, 6, -58, 3, 30, -196, -107, 70, -137, -98, -91, -66, -11, -24, -5, -120, 102, 9, 16, 33, -32, 1, 63, 27, 8, 14, 90, 76, -11, -38, -95, -176, -9, -15, -43, 48, 81, -40, 50, -238, 33, -11, 9, -98, -67, 0, -12, -37, -81, -3, 30, -14, -26, 46, -16, -39, -62, 97, -32, 74, -92, 9, -56, -22, 65, 41, -79, -77, -40, -70, 6, 5, 63, 41, -11, 38, 69, 68, -85, 35, -42, -22, -24, -45, 26, -19, 56, 37, 30, 19, -46, -22, 63, 51, -160, 116, -99, -52, 71, -38, -20, 138, -22, 41, 94, 11, -76, 19, -79, -4, -3, -48, 11, 16, 52, -119, 26, 23, -54, 9, -47, 72, 19, 132, 1, -160, 98, 27, 131, -74, 37, -151, -36, 19, -1, 51, 41, -18, 35, -149, -21, -11, -205, 60, -139, 26, 53, -145, 28, -112, 3, 7, 25, 78, -126, -6, -80, -10, -39, 161, -100, -164, 32, -22, 55, -91, -52, 31, 3, -13, -113, 59, -151, -19, -58, 28, 94, -48}
, {-58, 17, -53, -58, -58, 69, -32, 86, -5, 14, -21, -46, 37, 11, 23, -56, -22, 45, -17, -58, -60, 50, 17, 23, 27, 30, 120, 24, 47, 21, -40, 9, 9, 50, 0, -19, -26, -3, -42, 0, -59, -35, 77, 41, -10, -17, 55, 62, 93, 14, -43, 38, 78, -11, 17, -78, 90, 84, 39, 34, -40, 92, -25, 83, -55, -39, -141, -56, -45, 70, -38, -119, -77, -56, 85, 28, -152, 39, 22, -5, -121, 130, 71, 47, -64, -62, -7, 92, -18, 53, -83, -115, -13, -127, -32, -22, -38, 11, 24, 72, -8, 6, 79, -4, -3, -23, 43, 35, -101, -78, 36, -29, -49, 96, 34, 41, -12, -20, 0, 74, 93, 92, 34, 9, -10, -52, 30, 76, -7, -24, 23, 260, 77, 42, -51, 51, -52, 30, -18, -39, 39, -32, 51, 56, 20, -78, 57, -13, -75, 103, 67, -95, 150, 48, 39, -32, -23, -2, 16, -9, 47, 54, 83, -71, 44, -39, -13, -1, -64, 64, 0, 35, -45, -2, -19, -37, 0, -108, -101, 116, -38, 69, 125, -61, -14, -82, -44, -91, 97, -69, -22, -45, 41, -44, 137, 32, 10, -64, -1, -30, 25, -23, 33, 25, 63, -26, 18, -42, -48, -33, -88, 80, 92, -81, 53, -31, 46, -1, 0, -32, 43, 135, 41, -112, -96, -3, 102, 9, -109, -67, -39, -39, -2, -66, 45, 13, 54, -3, -56, 114, -11, -24, 75, -132, 54, 31, -54, 43, 45, 80, -8, -135, 11, -1, -30, 71}
, {-10, 0, 16, -16, -4, -147, 13, 46, -66, -77, 2, -172, 25, -54, 48, -39, -15, -29, 87, -15, -74, 82, 43, -11, 17, 10, -207, 33, -74, 83, -58, 35, -22, 30, 8, -269, 106, -74, -12, -24, -104, 4, 30, 27, 27, -27, -72, 43, -18, 60, -35, 75, -91, 8, 83, 44, -10, -217, -13, -67, 68, 4, -166, 30, -17, 36, -62, -28, 57, -167, -24, -12, -98, -93, -14, -5, 25, 87, -120, -35, 12, 13, -7, 53, 108, -128, 4, -51, -87, -206, -29, 7, 58, 131, 33, -86, 12, 50, -144, 6, -79, -77, -110, -56, -1, -57, -12, -52, 84, 20, -13, 22, -93, -171, -59, -27, -146, -56, -120, -43, -45, -38, 1, -61, -95, -106, -31, -72, -58, 85, -17, -112, -6, -2, -72, -100, 99, -121, 46, 54, -13, -5, 6, 59, 19, 128, 52, 26, 57, 16, -10, -81, -30, -73, -51, -33, 77, 1, 56, 135, 4, -10, -30, -26, -72, -62, -28, 16, 20, 24, -250, 70, 25, -93, 133, 15, 8, 22, -17, -223, 5, -16, 32, 75, -68, -34, 20, 113, 8, 49, -58, -24, 12, 46, -179, 65, 41, 73, -74, -25, -27, 53, 19, -63, -45, 48, -35, -25, 3, 21, 93, -152, -57, 39, -34, -77, -60, 48, 54, -80, -29, -110, -114, -7, -81, -19, -81, -56, 56, 38, 3, 87, -11, -115, -88, -30, 31, 113, 3, -62, 62, -45, 6, -82, 51, -26, 84, -7, -88, 9, -48, 63, -90, 43, 13, -146}
, {245, 9, 16, 127, 76, -12, -28, 106, 2, 40, 102, -71, 26, 15, -79, 40, 40, 10, 10, -56, 28, 74, 85, -54, 18, -198, 128, -4, 32, -3, 31, -6, -56, -128, -31, -78, 11, 83, -6, 22, -84, -89, -42, -75, 18, -33, 28, 108, 22, -40, 13, -37, -56, 41, -100, -18, 21, 37, -14, -56, -88, -35, 102, -46, 39, -47, 34, 81, 3, 28, 2, 69, 127, 95, -59, 4, 112, -51, -24, 93, 69, 19, -33, -105, 88, 121, 71, -176, 85, 72, 145, 17, -1, 113, -14, 75, -7, -51, 29, -55, -57, 6, -84, 68, -18, -106, -154, -80, 151, 134, -84, -21, 109, -26, -144, -55, 2, 80, 69, -110, 2, 15, -111, -137, 24, 55, 38, -100, -52, 42, 83, -113, 78, -74, 100, -35, -5, 56, -32, 78, -11, 32, -18, 94, -28, -27, 51, 5, -40, -58, 56, 16, 71, 36, 65, -22, -16, 23, -1, -21, 45, 75, 85, 31, 2, 1, 0, 37, 74, 39, 48, -18, -61, 70, -34, -70, 0, -23, -18, 117, 21, -32, 14, 3, -74, -14, -1, -15, 24, 48, -10, 65, 1, -148, -99, -46, -6, 71, -101, -131, 97, 101, -80, 62, -60, 131, 26, -102, -38, -32, -4, 96, -183, -24, -27, 88, -127, 15, -45, 86, -69, -125, -15, 24, -37, -6, -49, 13, 62, 25, 88, -1, -49, -5, -18, 23, -105, -17, 120, 50, -14, 106, 14, 52, -23, -4, 34, 55, 15, 83, 27, 31, 57, 26, -32, -42}
, {163, -56, -6, 66, 45, -42, -26, 68, 23, 9, -71, -48, 58, -53, -29, 85, 40, 39, -48, -136, 10, 17, 36, -74, 68, -20, 9, 48, -8, -7, -22, 38, -47, -86, 34, -209, 98, -71, -48, 5, -39, -136, -94, -56, 80, 28, -108, 103, -47, -4, -139, -33, 73, 23, 147, -202, -58, -145, -44, -46, 15, 19, -126, -16, -75, -229, 159, -27, -56, 100, -41, 1, -42, 128, -35, -73, 88, -95, 80, -216, -70, -78, -141, 30, -57, -65, 135, 38, -1, 103, 81, -23, -189, 96, -207, -63, -56, -56, 164, -24, -108, 85, 3, 62, -12, -140, -56, 48, 64, -29, -195, -96, 60, 73, -43, -52, 76, 61, 40, -113, 9, -10, 43, -96, 51, -29, 70, 32, 70, 58, -23, -8, 50, -47, -49, -76, -18, 14, 32, -44, 33, 22, -80, 69, -66, 55, 52, -57, -8, 7, -125, 52, -58, -4, 5, -31, 20, 32, 3, 5, -70, -84, 3, 26, 23, -35, 39, -10, -83, -55, 22, 85, -16, -22, 52, 51, -10, 1, -33, 56, -11, -21, -66, 14, 13, -75, 61, 22, -12, 31, -13, -118, -4, 0, -3, 22, 19, -11, 88, 14, 23, -6, -116, 73, 13, -132, -12, -52, -27, -43, -59, 55, -164, 15, -2, 102, -7, 58, -78, 21, 3, -19, 109, -120, -47, 68, -11, -46, 73, -24, -5, -32, 16, 4, 48, 50, -25, -150, -14, 84, 26, -27, -18, 82, 96, -63, -24, 0, -33, -57, 51, 66, 236, -61, 30, -29}
, {23, 38, -46, -49, -7, 54, 5, 57, 62, -132, 4, 70, -126, -10, -56, 46, -107, 39, -17, -17, 98, -7, -41, -51, -92, 9, -53, -42, 10, -115, -13, -46, 16, -27, 53, 3, -17, 42, -78, -43, -63, -43, -4, -59, 3, -43, -14, -49, -10, -15, 30, 10, -7, -81, -62, 45, -21, 115, -18, 88, -52, -70, 65, -35, -59, -20, 89, -9, 35, -8, -54, 53, 46, 12, 1, -1, -2, 15, 82, -21, 89, -78, -115, -41, 13, -34, 139, -63, 101, 42, 98, -81, -67, 62, -97, 2, -53, -43, 10, -111, 10, 65, -60, 15, 8, 34, -13, 37, 47, 181, -41, -76, 108, 133, 12, -40, 28, -8, 10, 1, -34, 21, 7, 37, -26, 30, 54, 21, 106, 40, -43, -100, 79, 53, 0, 47, -33, -7, 99, 56, -147, -55, 13, -28, -203, -90, -163, -64, 31, -13, -144, -28, -230, 78, 72, 76, -76, -148, -15, -185, -59, -149, -31, -45, -202, 22, -7, -121, 24, -75, 3, -73, 62, -59, -2, 79, -154, 28, 39, 27, 67, 55, -138, 2, 71, 93, 8, 43, -123, -47, 61, -24, -4, 54, 55, -90, 44, -29, 77, 70, -118, 6, 132, 23, -28, -43, -90, 36, -53, -19, 35, 3, 78, 12, -84, -82, -18, -51, 38, 58, 12, -38, 75, 36, -111, -115, 28, 108, -62, -42, -24, -118, 2, 23, 46, -168, -21, -42, 27, 109, -137, -138, -29, 32, 33, 1, -52, -24, 80, 18, 17, -66, 30, -27, -11, 123}
, {129, -26, -13, 106, -75, -49, -41, 27, 59, -71, -33, -81, 25, 11, -29, 39, -75, -11, 11, -44, 98, 11, 72, 49, 33, 35, 9, 15, -15, 5, -41, 53, -84, -35, 104, -62, 76, -146, -22, 21, -94, -92, -2, 43, 14, 92, -53, 138, 10, 23, -36, -10, -62, 24, 62, 38, -5, -66, 6, 13, 74, 174, -160, -17, -82, -125, 67, -14, -39, 71, -50, -15, 24, 177, -31, 45, 100, -74, 97, -112, -90, -108, -119, 35, 28, -111, 190, 10, -4, 128, 154, -114, -108, 100, -181, 17, -35, -34, 55, -21, -68, 39, -42, 91, 21, -80, -66, 15, 164, 138, -127, -159, 38, 88, -37, -13, 10, 29, 24, -49, -44, 20, 47, 105, 48, -18, 41, -1, 25, 43, -36, -31, 32, -22, 17, -59, -1, 83, 66, 26, 12, 18, -5, 96, 49, 45, 80, -52, 20, -24, -38, -39, 6, -12, -20, -86, 66, -34, 32, 22, 83, -17, 68, 17, -47, -87, -5, -67, 7, -18, -47, 31, 78, -38, 82, 19, 10, -46, -52, 29, -34, -32, 31, 40, 35, -12, -19, 60, -76, 0, -74, -104, 42, 36, -103, -34, 10, -28, 17, -27, -28, -26, -39, 97, 41, -15, -31, -47, -19, -16, 22, 21, -107, -7, -10, -25, -29, -12, -70, 39, 67, -92, 101, -88, -20, 163, -23, 5, 16, -69, -53, -33, 89, 45, 19, -19, -78, -198, -10, 85, -61, 76, -2, -3, 35, -42, -70, 31, -24, -43, 8, 14, 145, -39, 2, -57}
, {154, 34, -132, 42, 35, -52, 53, 100, -31, -146, 4, 19, -136, -20, 66, 20, -8, -30, -21, 7, -56, 50, -160, -49, -148, -30, -87, -140, 36, -168, -37, -86, -13, -42, -130, 90, -89, 108, 0, -51, -12, 39, -14, -16, -54, -113, 8, 40, -18, -18, 82, -26, 10, -90, 39, 32, 71, 15, 8, 9, -83, -131, 40, -28, 87, 46, 57, 45, -17, -9, 3, 210, 97, 28, -11, -54, 89, 23, -52, -6, 58, -90, -115, -22, 57, 14, 37, -88, 33, -56, 108, 54, -16, 26, 41, 65, 18, -50, 31, -10, -41, 3, 2, 70, -136, 53, 24, -13, 38, 210, 13, 12, -38, -136, -35, -14, -65, 134, -27, -26, -48, 40, -30, 23, -50, 50, -68, -6, 89, -55, 26, -120, -56, -33, 81, 49, -47, -34, 36, 38, -86, -64, -43, -166, -221, -77, -74, -51, 137, -96, -192, -33, -167, 27, 46, -56, -20, -131, -62, -175, -115, -102, -81, -31, -130, 9, 16, -146, -27, -136, 76, -38, 91, -25, -162, 15, -125, 51, 13, -10, 18, 13, -53, -147, 7, 49, 11, -7, -86, -23, 0, -30, -3, 3, -60, -27, 50, 27, 48, 49, -43, -87, 38, 67, 19, -119, -168, 37, -50, 37, 11, 1, -71, 44, -51, -149, 19, 124, 78, -44, 30, -45, 7, 58, 17, 0, 29, 41, 88, 2, -87, -120, 28, 44, -52, -133, -71, -76, 70, 68, -12, -81, -64, -70, -26, -42, -45, -58, 71, -78, 116, -85, -54, 17, -72, 45}
, {120, 66, -46, 8, -26, 22, -68, 6, 80, 104, -3, -10, -31, 35, -18, -52, -28, 49, -31, -7, 14, 14, 14, -30, -10, -73, -28, 72, 63, -73, 36, 0, -61, -34, -38, 16, -102, 111, -25, -21, -56, 24, -4, 4, -10, -107, 44, 75, -10, 75, 24, 7, 56, 47, -31, 44, 134, 52, -31, 63, -29, -32, 28, 15, -31, 46, 52, 62, -18, -43, -79, 27, -12, 79, 102, 49, -15, -57, 18, 20, -74, -55, -28, 15, 19, 58, 22, 7, -85, -24, 119, -61, -52, -13, -196, 21, -18, -53, -14, 126, -7, 48, 3, 48, 63, 2, -40, -10, 17, 20, -11, -53, 114, 51, -2, 35, 99, 144, 105, -29, 80, 97, -19, 83, 24, -58, 7, 43, -38, 3, -35, 40, -5, 75, -15, 96, -87, 204, 92, -12, -96, -10, 73, 50, -218, -143, -60, -12, -60, 35, -40, -25, -73, 106, 152, 38, -80, -43, -93, -203, -56, -51, 37, -56, -88, 22, 165, -60, 6, 142, 71, -124, -8, 32, -193, -35, -33, -60, 11, 166, 4, 66, -30, 9, 74, 64, -21, -7, -42, -185, 105, 78, 54, -20, 126, -21, -42, -20, -33, 8, 77, 82, 56, 17, 110, -35, -46, -109, -49, -86, -30, -28, 85, -35, 4, 74, 59, -100, 15, 11, 32, 57, 19, 36, 26, -64, 34, 111, -69, 23, 89, 53, 22, 46, -52, -112, -31, 88, 40, -32, 0, 33, 29, -16, -147, 153, -24, 68, -37, 90, -75, -22, 86, -41, 84, 92}
, {70, -57, 11, 5, 18, -58, 21, 35, -145, -46, 13, -189, 40, 30, -16, 2, -73, 52, 127, -101, 57, -120, -22, 76, -5, -50, -32, 72, 36, -16, -49, 105, -57, -89, 23, -161, 90, -122, -16, -59, -266, -69, -24, 10, 8, 80, 2, 113, -34, -123, -65, 33, -179, -24, 92, -37, -10, -36, -38, -113, 95, 114, -70, -60, -27, -20, 26, 11, -25, -31, -27, 42, 46, 14, -36, 34, 31, -59, -33, -84, -12, 72, -65, 62, 38, -76, 71, 18, -69, 21, -17, -60, -21, 19, -56, 4, -56, 7, 25, -88, -67, -91, 44, -22, -22, -83, 6, -59, 83, 17, -97, 12, 78, 11, 107, 19, -5, -30, -20, -40, 13, -35, -17, -39, 63, -71, -5, 43, -63, 123, 52, -42, 23, -39, -83, 131, -18, -245, 43, 92, -48, -99, 1, 182, 12, -13, -38, -57, 27, 31, 58, -52, -1, -4, -51, -101, 61, 106, 42, -38, 88, 65, 81, 50, -21, -70, -12, -44, 39, -28, -61, 40, -47, 6, 36, 40, -32, 6, -26, 83, 13, -114, 52, 38, -28, 22, 47, 30, 79, -154, 2, -71, 111, -10, -122, 27, 24, 105, -96, 14, 18, -22, -35, 34, -44, 3, -47, -4, -22, 69, 88, -30, 26, 8, -45, -1, -62, 9, 94, 38, -36, -56, 4, 85, -84, 10, -70, -22, -44, 3, -100, 21, 28, -107, 17, 5, -181, -100, -51, -47, 26, 124, -65, -8, 123, -16, 115, -133, -35, 46, -81, -62, -73, 11, 64, -87}
, {63, -11, 4, 149, 62, 85, -30, 83, -58, 22, 63, -11, 57, -91, -33, 55, 14, -66, -53, -59, 10, 105, 78, -74, -18, -52, 73, -6, -61, 20, 80, -15, -36, -90, 55, 36, 73, 7, -72, -8, -54, -56, -129, -106, 132, 17, 8, 112, -51, -72, -29, -71, 62, 41, -40, -58, -38, 27, -7, -144, -110, -64, 47, 21, 59, -11, 36, 22, 22, -15, -45, 21, -20, 38, -93, -27, 178, -92, -104, 97, -15, -29, 82, -60, 104, 112, 51, -103, 95, -46, -2, 57, 6, 85, -33, -24, -168, -21, 70, 27, 46, 56, -102, 13, 4, -92, -139, -19, 11, 49, -57, 22, -23, -25, -149, -2, 15, 33, -68, -100, 23, 57, -108, -92, -83, -18, 64, -25, -45, -64, 11, 37, 115, -72, 20, -99, 34, -159, -186, -21, 81, 96, 36, -33, -27, 81, 3, 114, -77, -39, -21, 70, 92, 55, 53, 84, 21, -16, -83, 36, 38, 7, 20, -17, 101, 120, 30, 81, 15, 0, 40, 80, -116, 15, 5, 28, -40, 33, 72, -45, 4, -90, 111, -18, 14, -170, 6, -1, 74, 22, 35, 35, -100, -72, -22, -6, -108, -104, 3, -88, 22, -101, -83, -29, -29, 130, 42, -130, -79, -29, 75, 27, -87, 17, -62, 69, 39, -20, -113, -9, 55, -121, -57, -50, -3, -51, -35, 106, -30, 84, 29, -37, -65, 26, -51, 54, -76, -91, -7, -15, 49, 6, 62, -33, -35, 25, 65, 133, 76, -99, -71, 32, -29, 11, -23, -74}
, {9, -79, 2, 39, 74, -23, -33, -104, -82, 3, -17, 47, -55, -20, 31, 40, -37, -93, 0, -3, -33, -10, -12, -17, -48, 109, -60, -25, -48, 21, -57, -104, 55, 9, -35, 95, -53, 51, -49, -13, 56, -18, -49, -14, -1, -33, -64, -14, -53, -104, 46, 70, -20, -92, -6, 9, 33, 17, 40, 77, 1, -117, 38, 33, 65, 5, -38, 116, 40, -135, 17, 75, -19, -65, -29, -97, -16, 14, -162, -38, 136, 6, 50, 31, 72, 14, 18, -105, 0, -182, -87, 72, 46, 47, 50, 33, 38, -32, 15, -38, -42, -75, -86, 115, -116, 6, -99, -17, 6, 140, 65, 75, 92, -201, -113, 29, -98, -14, 14, -63, -30, -48, -96, -70, -45, 74, -41, -34, -6, -51, 2, -164, -88, 14, -92, -59, 61, -25, 49, -19, -18, 6, 76, -162, -28, -18, -145, 64, -20, 40, -85, -21, -103, 18, 22, -155, 27, -114, -17, -23, -39, -102, -97, -54, -120, 120, 69, -52, -17, 1, -5, -71, 2, 46, -28, -17, -102, 10, 64, -119, 15, -113, -107, -6, 15, 63, 6, 27, -4, 136, 82, -3, -19, 48, -135, 35, 103, -1, 24, 51, 58, -58, 16, 1, 27, -58, 39, 27, -19, 52, -16, -131, 16, 61, -20, 9, 0, -39, 32, -9, 29, -99, -44, 39, 27, -86, 17, -25, 104, 30, 19, 130, -10, -83, -42, 2, -21, 32, 59, -52, 139, -11, -162, 4, -27, -134, 69, -15, -56, -62, 103, -5, 64, 23, -95, -29}
, {-60, -137, -15, -105, -44, 41, -75, -48, -21, -7, -38, -160, 4, -119, 32, 33, -33, -28, 1, -60, -46, 0, 19, 87, -19, -52, 0, -13, -62, 61, 52, 65, -28, 23, 64, -180, 53, 0, 79, 93, -105, -61, -10, 40, 56, 28, -48, -175, 48, -47, -52, 18, -45, 30, 30, -55, -5, 1, -116, -5, 17, -12, -57, 47, -55, -76, 78, -58, 182, 78, -19, -19, 16, -28, 23, 130, 6, 8, 17, -91, -64, -65, -35, 8, -153, 55, -90, -4, -6, 62, 161, -83, -40, -107, 23, 11, -106, 14, 47, -80, 31, 61, 41, -32, 115, 16, 91, 10, -38, -144, -27, -56, -73, 98, 72, 81, 119, -146, 19, 28, 43, -44, 116, -28, 97, -35, 154, -18, -147, 47, 17, -61, -112, 1, -80, -57, 19, -179, -52, -12, 64, -22, -13, -34, -35, 35, -31, -52, -34, -73, -40, 36, -5, -25, -184, -100, 50, 5, 40, 23, 31, -111, 36, 17, 39, -22, -145, 48, -34, -26, -129, 68, 37, -46, 26, 40, 8, 18, 20, -24, -92, -105, -96, 10, -16, -16, -41, 43, -55, -43, -37, 6, 105, 51, -142, 82, -7, 31, 83, 93, -23, 8, -54, 37, -24, -31, -188, 33, 23, 11, 25, 2, -126, -7, -1, -13, -78, 17, -10, 155, 55, -29, 18, -10, -131, 46, -51, -65, 54, -75, -39, -65, 64, 54, -1, -23, -39, -134, -83, 89, 10, -46, -70, 85, 53, 25, -24, -33, 52, -21, -59, 24, 9, 8, -74, 14}
, {-28, 62, -68, 47, 31, 9, 18, 91, -57, 37, 12, -56, 24, -29, 78, 117, -18, -70, 9, 36, -89, 65, 65, -8, 63, -21, 81, -13, -38, -2, 6, -12, -29, 48, 42, 14, -60, 72, 107, -9, 14, 9, 39, 12, 44, 24, -24, 62, -5, 14, -35, 39, 67, -34, 20, 3, 60, -47, -17, 24, 11, -107, 48, 24, 149, 22, -44, -22, -54, -183, -9, -71, -35, -224, 33, 43, -101, 18, -109, 98, -16, 16, 104, 3, 116, 64, -112, -9, -111, -26, -228, -7, 46, -60, 29, -73, -9, -44, -24, 60, -29, -60, -12, -16, 78, -31, -54, -26, -114, -19, -21, 85, -113, -20, -2, -65, -31, -174, -23, -45, 9, -140, -115, -32, -8, -57, -52, -37, 4, -100, -8, 3, 27, 8, 27, -50, 32, 35, -172, -56, 14, 27, 6, -53, 13, -18, -6, 69, -156, -165, 85, 119, 42, -7, 81, 42, -66, 43, -15, 42, 45, 157, -3, -156, 61, -161, -48, 75, -3, 33, 12, -41, -48, 33, -28, 4, 13, -99, -148, 90, -17, 33, 29, 26, -20, -102, -17, -81, 90, 30, -163, -10, -55, -253, 46, -42, -48, -11, -55, -157, 29, 97, 11, -202, 80, 55, 27, -169, -38, -106, -43, 2, 21, -184, 77, 104, 72, -137, 0, -63, -37, 77, -39, -88, -10, -34, -45, -65, 84, -113, 66, 42, -128, 31, 30, 63, 95, 64, -77, -155, 40, -130, 12, -160, -74, 69, -93, 4, -6, 103, 4, -129, -7, 8, -16, 30}
, {-116, -13, -85, -49, -72, 14, -51, -107, 78, -114, 94, 62, -91, 22, 18, -24, -65, -37, -20, -31, -12, -86, -46, -34, -165, -59, -30, -85, 16, -145, 22, -125, -14, 9, -137, 5, -135, 20, -6, -59, 26, 84, 35, -35, -58, -150, 6, -116, 17, -43, -5, 12, -25, -118, -97, -8, 13, 11, -16, 96, -35, -64, 20, 54, -12, 55, -169, 70, -3, 23, 42, 9, -65, -7, 34, -149, -53, -58, 115, -41, 16, -99, -88, -19, -121, 98, 31, 3, -23, -21, -106, 46, 23, -55, -19, -23, 88, 25, -32, 29, 92, 11, 22, 57, -109, 46, 5, -16, 0, -108, -15, 28, 58, -163, 2, 25, -44, 32, 64, -10, -38, -5, -40, 61, -25, 75, -80, 42, 158, 26, -11, 73, -22, 64, 96, 2, -70, -43, 19, -13, -51, -102, 26, -69, -165, -97, 23, -153, 74, -6, -30, -19, -108, 66, -1, -127, 16, -66, -28, -116, -226, -134, -37, -63, -134, 48, 14, -170, -2, -181, 24, -87, 52, -36, -89, -18, -43, 11, 21, 52, 2, -46, -58, -136, -24, 42, 9, 11, -123, 22, 21, -3, -83, 22, 18, -3, 73, -110, 96, 80, -9, -136, -58, 88, 12, -230, 28, -32, 25, -2, -100, -22, -108, 43, -3, 35, -34, 61, -61, -132, 3, -3, 36, -71, 91, 72, 99, -35, -47, 80, -151, -56, 38, 22, -82, 25, -27, 54, 61, 21, 34, 58, -93, -32, -22, -141, 39, 3, -7, -147, 34, 53, 75, -108, 6, -14}
, {-154, 64, -17, -99, -108, 11, -52, -4, -90, 29, -14, 43, 6, 19, -37, 50, 13, 43, 26, -37, 2, -33, -25, -68, 10, 53, 59, -25, 2, 40, -6, -33, -9, -75, -6, 5, -21, 32, 25, 17, 17, 13, 34, -24, 29, -70, -21, -24, 83, 0, -7, -28, 27, 21, -29, 52, -43, 2, 21, 5, 12, 30, 77, 26, -50, -7, -10, -31, -17, 44, 23, -90, -87, 2, -48, 4, -175, 27, 49, 63, -21, -5, -52, 8, -125, 120, -45, 22, 22, 26, -138, -92, 13, -34, 57, -90, 46, 5, 21, -4, 61, 100, 62, -68, -8, 11, 54, 60, 14, -76, 27, -47, -189, 81, 11, 64, -5, -8, 12, 7, -18, -7, 25, 33, -11, 37, 34, 43, -44, 62, -7, -79, -92, 50, 43, -152, 50, -69, 175, 63, -148, -5, -60, -17, -106, -70, -121, -73, 116, 38, -59, 61, -151, -19, -82, -151, 35, -22, -13, -165, -236, -165, -69, -21, -170, 40, -66, -170, -8, -102, -95, -63, 117, 135, 19, 72, -98, 103, 17, -40, 44, -11, -155, 10, 38, 62, -5, 3, -146, -127, 106, -48, 9, 104, -184, 65, 46, -44, 26, 65, -67, -2, -39, -30, -34, -31, -14, 62, 3, -24, 103, 2, 53, 120, 17, -143, -42, -29, 36, -66, 6, -15, -25, -41, 17, -177, 101, -67, 105, 61, -120, 11, 20, -35, -33, -162, -100, 74, -56, 43, 85, -10, -70, 22, 45, -88, -93, -114, -93, -11, -50, -32, -9, -59, 9, 49}
, {67, -113, 34, 22, -114, -217, 17, 36, -23, 12, -84, -135, 21, -73, -1, 39, -44, 21, 20, -114, -36, -3, -3, 24, 36, -122, -99, 9, 9, 44, -54, -23, -3, -12, 22, -128, 22, -150, 45, -77, -208, 7, -8, 12, 29, 43, -8, 80, -58, -48, -93, -26, -226, 17, 28, -76, 4, -76, -175, -69, 31, -118, -128, -30, -16, -34, 92, -28, -39, -136, -45, -74, 43, -93, -109, -25, 60, -8, -189, -65, 42, 71, 122, -81, 116, -73, -98, 61, 17, -25, -87, 84, 48, 47, -9, -49, 2, 24, -181, -4, 8, -56, -87, -47, -35, -48, -43, 23, 20, 112, -27, 70, 25, -38, -52, -114, -138, -111, -80, -48, -72, -5, -132, -6, 4, 5, -35, -14, 42, -76, 23, -82, 12, 30, -141, -4, 121, -27, -38, 31, -4, -25, -82, -13, 62, 13, 39, 27, 85, -44, -24, -48, 25, -12, -2, 30, 82, -25, 21, 77, -17, 0, -78, 40, 20, -64, -21, 19, 19, 34, -124, -16, -43, 46, 78, 66, 17, 104, 52, -172, 56, 11, 23, 13, -33, 47, 14, 65, 29, 129, 60, -93, 40, -30, -56, -2, 53, 15, -143, 39, 17, 4, 86, 49, -61, 1, 56, 1, 2, 71, 64, -99, 111, 18, 5, -12, -58, -78, 53, -48, -106, -7, -147, 125, -60, -36, -91, -62, 40, 76, -37, 91, -77, -143, -27, -30, 3, 153, -11, -186, 125, -29, -84, -123, -3, -42, 122, -40, -128, 64, -118, -5, -66, -8, 62, -24}
, {67, 98, -58, 154, 26, 73, 18, -92, 98, 35, 18, 82, -22, -3, -14, -27, -28, -49, -35, 29, 27, -77, -85, -21, -42, 36, 43, 8, -7, -36, -16, -22, -17, -12, -21, 31, 35, -30, -24, -7, 29, -19, -27, -2, -50, -6, 53, 181, 84, -85, 5, 43, 6, 49, -120, 47, 24, -16, -8, 33, -44, 28, 38, 81, -6, -60, 29, 24, 41, 74, -57, 63, -67, 189, 18, -86, 94, -95, 55, -47, 91, -114, -72, 37, -106, 58, 142, -41, 29, 117, -19, 30, -119, 10, -39, 51, -24, -23, 1, -94, 50, 84, -28, 39, -35, -58, -22, -82, 84, 131, -77, -72, 64, -53, -13, 34, 92, 55, 46, -72, -33, 27, 61, 50, 6, 83, 29, -23, 55, 8, 23, 87, 53, 2, -59, 64, -44, 47, 9, -18, -34, -1, 28, 19, 0, -10, 16, -125, -41, 65, -34, 7, -85, 45, 78, 17, -100, -58, -13, -56, -128, -63, 21, 53, 8, 37, 34, -46, 66, -22, 0, 38, -45, -6, -1, -5, -48, -17, -10, 66, 16, -96, 37, -43, -56, 10, -6, -12, -47, 6, -31, -63, -17, 59, 92, 41, -36, -113, 95, 5, 60, -120, -28, 113, 35, -205, 40, 29, 13, -3, -79, 11, 42, 63, -11, 12, -15, 19, -142, -63, 118, 23, 79, -234, 22, 177, 90, -54, -22, -38, -79, -34, 84, 42, -13, 41, 60, -124, 17, 97, -5, -35, -175, -38, 47, -89, -53, -74, -20, -146, 107, 32, 14, -165, 23, 54}
, {-164, -51, 27, -35, -1, 60, 16, -33, 40, -12, 56, 27, -19, -108, 10, -10, -70, 39, 81, -62, -44, 43, 30, 6, -12, 7, 29, -7, -69, 4, -41, -61, 1, 68, -64, -35, 41, -14, -23, 53, -16, 130, 20, 23, -5, 20, 2, -103, 43, 111, 3, 49, 39, 53, 65, 22, -23, -33, 11, 6, 23, -46, -1, 8, 25, 80, -200, -54, 39, 41, 118, -217, -102, -98, 96, 8, -241, -22, 58, 3, -190, -40, 35, -20, -97, -143, -226, -37, -122, -83, -21, -150, 13, -250, 79, -81, 14, 47, 5, 82, 69, -31, 76, 22, 127, 104, 76, -117, -171, -211, 60, 26, -125, -60, 2, 55, -88, -63, -74, -22, 72, -66, 49, -60, -30, -141, 24, 73, 98, -56, 82, 5, -7, -33, 30, -54, 22, 39, 33, -4, -130, -110, 36, -32, -53, 21, -7, 48, -16, 25, 74, -6, 29, -33, -8, -43, -63, 33, 3, 50, 37, 12, 54, -54, -70, -48, -38, -35, 8, -49, -57, -49, 36, -2, -36, -12, -69, -30, -28, -6, -31, 17, -33, 20, 43, 26, 33, -54, -32, 88, -41, 56, -85, -20, 55, -5, 30, 23, -60, -9, 24, 15, -119, -78, 76, 36, 14, 8, 68, -66, -46, 2, -166, 3, 21, -100, -25, 90, 15, -28, -103, -59, -50, -90, 120, 52, -27, -142, 32, 9, 17, 17, -42, -90, 51, 143, 7, 102, -105, 8, 50, 16, 93, -37, 43, -77, 56, 6, -2, -13, -67, 58, -60, 13, 31, -161}
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

typedef int16_t dense_5_output_type[FC_UNITS];

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
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
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
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
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


const int16_t dense_5_bias[FC_UNITS] = {31, 41, -101, 25, -38, -45, -79, -20, 145, 13}
;

const int16_t dense_5_kernel[FC_UNITS][INPUT_SAMPLES] = {{71, -78, -42, -159, 124, -121, -74, -194, 68, 90, 32, -165, -75, 99, 76, -76, 81, -42, -109, -83, -213, 60, -117, -65, -188, 41, 88, 61, -74, -13, -64, -186, 76, 1, -78, 24, -149, 37, -100, -222, -126, -225, 125, 53, -141, 106, 132, 92, -169, -242, 52, -102, 71, -119, 87, -84, 68, 95, -63, -155, 50, -58, -272, 76, -100, 105, 83, 51, 40, -112, -210, 78, -61, -158, -116, 117, -179, -205, 108, -87, 76, -230, 62, -150, -27, -212, -99, 44, -173, -27, 121, -20, -174, 70, 158, 86, -19, 109, 101, 70, -96, -71, -116, 130, -129, -163, -211, 61, 49, 47, 79, 39, -99, -111, -18, -171, -123, 31, -112, 82, -73, -119, 125, -39, -141, 94, -2, 39}
, {98, 91, 125, -59, 72, 116, -36, -128, -90, -250, 58, 16, 56, 64, 106, 123, -74, -264, 104, 2, 69, 15, -237, 109, 76, -199, -113, -155, -110, -190, 122, -141, -201, 51, 124, 101, -197, -189, -57, 19, -219, 11, 131, 38, 98, -254, -51, 113, 144, 13, -13, -78, -146, -102, -114, -82, -94, -177, -186, 72, -24, -10, 79, 65, -52, -144, 17, -52, -160, 71, 99, 99, -106, 120, 64, -305, -112, 103, -233, 111, 29, 19, -205, 69, 1, -129, -29, -179, 96, 103, -75, 89, -122, -179, 90, -182, -4, -146, -121, -54, 118, -77, 64, -171, -131, 97, -126, 17, -185, 46, -161, -186, 91, -212, 108, -38, 124, 81, 91, -102, 56, -9, -67, -95, -167, 105, -181, -201}
, {-124, -49, -104, -150, -84, -55, 138, -168, 95, -142, 43, 93, 42, -170, -22, 114, -167, -4, -196, 97, 62, -82, 88, 119, -88, 40, 76, 59, -176, -190, 111, 38, -113, -65, 80, -10, 44, 55, -100, -30, 60, -72, -16, 45, -43, -61, -102, -63, 9, -122, 47, -169, 42, -260, -112, 54, 70, -156, -38, 79, 111, 92, -125, 75, 94, 106, -169, 55, 66, 51, 80, 77, 124, -157, -31, -38, 87, -44, -32, 107, -173, 78, 37, 53, -21, -163, 103, 121, 96, 5, 51, -154, 69, 55, -168, -122, 44, -39, 48, 88, 81, 56, -218, 74, 93, 3, -1, 31, -88, 110, 68, -97, 63, -134, -98, -170, -84, 80, -152, 118, 29, -228, 138, -91, -82, -202, -107, 59}
, {-122, -100, -166, 87, -80, -47, -61, 8, 88, -61, -145, 117, -89, -143, -170, -109, -53, -43, 15, 83, -63, 65, 32, -54, -65, -146, -15, 110, -89, -81, -156, 61, -151, 109, 89, -160, -59, 49, -85, 76, -136, -20, -40, 41, -28, -50, 113, -68, 74, 19, 68, 85, 68, 71, -183, -58, -79, -94, -63, -119, 114, 104, 91, -48, 120, 3, -177, 32, -74, 59, 82, -112, -71, 119, -116, -215, 82, 53, -138, -106, -116, 89, 46, -7, -195, 104, -151, -43, 94, -197, 36, -187, -15, 92, -115, 77, -168, 5, -93, 15, 113, 76, -163, 12, 60, -58, 31, 35, -18, -91, -1, -87, -77, -171, 87, -108, 147, 23, -120, -112, 83, -87, -47, 127, 119, -129, 71, 56}
, {54, 72, 92, 73, -73, -159, -31, 84, 86, 58, 60, 66, 28, -82, -94, -146, 17, 98, 65, 82, -125, -163, -140, -197, 82, 108, -98, 104, -91, -45, 102, -119, -72, -125, -213, 104, 87, -216, -56, -231, 80, 81, 101, -141, 101, -157, -145, -195, -59, 38, -98, 101, 78, -40, 90, -170, 90, -102, 41, -105, -74, -178, 60, -240, -17, -45, -103, -103, 78, -201, 66, 105, 41, 16, 100, 57, -218, -56, 19, 34, -157, -162, -132, 71, 25, 83, 65, -165, -128, 131, -207, 48, -7, -36, -42, -130, -62, 114, -182, -119, -180, -152, -130, 83, -152, 85, -263, -26, 31, -254, -172, 100, 72, 50, -65, 46, 39, -76, 94, 121, 93, -87, 87, -129, 65, 85, 39, -114}
, {-72, -199, -31, 63, -1, 83, -141, 97, -99, -59, -177, 14, -121, 81, -92, -192, -98, -137, -116, -165, 66, 21, 25, -211, 59, -104, 20, -100, 124, 71, -144, -138, 55, 91, -8, -113, -213, 3, 109, 68, -105, 15, -90, -68, -208, 88, 116, 90, 85, 26, 50, 45, 96, 72, -146, 104, 81, 81, 72, -54, -96, -153, -50, 57, -135, 42, 61, -236, -177, -6, -115, -44, -74, 126, 86, -169, 47, -120, 106, 27, 47, 103, 59, -236, -55, 109, 79, 74, -115, -277, -117, -175, -117, 87, -143, 80, 53, -147, 60, -10, 105, -234, -45, 102, -57, 98, 7, 54, -32, -137, 76, -138, -146, 23, 75, 45, -60, 57, 93, -32, -171, 69, -93, -92, 106, -34, 61, -223}
, {32, 30, -86, -267, 132, -31, -34, -125, 105, 76, 39, -159, 39, 119, -99, -38, 120, -159, 75, -216, -89, 76, -210, -137, -110, 7, -133, -93, 147, 86, 92, -142, 94, 55, -2, 44, 2, -245, -19, -7, -79, -194, 103, -153, -181, -10, -72, 14, -145, -312, 52, 126, -50, 104, 86, 93, -74, 77, -32, 62, -107, -133, -242, 31, -46, -103, 76, -153, -61, -112, 12, 79, 82, -128, -193, 33, -113, -105, -202, 25, 37, -158, 62, -123, -146, -104, 119, 49, -194, -110, -172, -67, 60, -130, -39, 103, 60, 67, -105, 104, 2, -135, 4, 119, -268, 83, 28, -38, 191, -132, 54, -44, 60, 84, -94, 54, -95, -50, -140, 120, -229, 76, -10, -132, -215, -120, 67, 10}
, {-17, 110, -27, 68, -181, 123, -125, -79, -30, 62, -98, 105, -146, -152, 93, 34, -276, -152, -96, 102, 78, -261, 30, -30, 78, 59, 56, -75, -184, -176, -42, -170, 115, -141, 26, -154, 86, 38, 81, -113, -57, 87, -160, 52, 94, 13, -113, 135, -50, 16, -151, -112, 97, -119, 84, -133, -193, 92, 70, 49, 89, 104, 68, 79, 116, -156, -155, 45, -135, 60, 92, -129, -162, -108, -144, -43, -133, 111, 118, -140, 16, -127, -187, 73, 9, -67, 67, -179, 101, -89, 28, 81, -158, -121, -101, -202, -209, -33, 113, -137, -17, -128, -49, -136, 79, -159, -119, -132, -23, 120, 56, -77, 26, -52, -96, 8, -31, 86, 20, -20, 59, -252, -40, 119, 23, -47, 76, 65}
, {68, -159, 133, 71, 85, -110, -127, 62, -67, -33, -196, 106, 45, 92, 46, 7, -204, 99, 83, 94, 64, 62, 7, 115, -187, -111, 95, -55, 9, 65, -33, 116, -150, -8, -199, 83, 81, 20, 124, 68, -109, -233, 89, 15, -70, 30, -14, 7, -133, -25, -58, -45, -129, 39, -4, 125, 78, -44, -172, -22, 100, -52, -35, -164, 105, 81, 4, 34, 41, -105, -103, -125, -159, 109, -70, -26, 93, 76, -32, 51, 32, 111, 57, -185, -214, 105, 20, -19, 81, -66, 85, 83, 44, -81, -106, 72, 63, -151, -96, 61, -92, 77, 32, -30, 34, -69, 44, -175, -139, -63, 70, 83, 7, 3, -181, 34, -84, -184, 93, -18, -130, 50, -147, -125, -48, -50, 43, 56}
, {-124, -42, 20, 81, -82, -81, -9, 87, -204, -55, 19, -156, -113, -122, 67, -54, 135, 62, 78, 86, -20, -182, 79, -94, 80, 95, -133, -97, 31, 78, -91, 153, 33, -159, -165, 103, 31, 29, -129, -88, -212, 64, 62, -149, 99, 79, 96, -7, 72, 35, -158, -99, 90, 54, 57, -38, -33, -84, 72, -212, -180, 111, 69, -98, -147, -87, -76, 37, 51, -157, -93, 52, -55, 109, 93, 102, -88, 105, -19, -211, 29, 60, 32, 79, 26, 96, 3, 101, -128, -105, 92, -60, 5, 90, 53, -112, -91, 29, -23, -190, -106, 34, 73, -78, -95, -16, -141, -9, -120, 75, -62, 95, -165, -110, -120, -111, -134, -120, 77, 40, 92, 64, 128, -64, 100, 83, -118, 76}
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

#define MODEL_INPUT_SCALE_FACTOR 9 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_INPUT_NUMBER_T int16_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

#define MODEL_OUTPUT_SCALE_FACTOR 9 // scale factor of last layer
#define MODEL_OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_OUTPUT_NUMBER_T int16_t
#define MODEL_OUTPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[28][28][1];
typedef int16_t input_t[28][28][1];
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
