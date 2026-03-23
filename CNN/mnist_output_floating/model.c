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
#include "conv2d_16.c"
#include "weights/conv2d_16.c" // InputLayer is excluded
#include "conv2d_17.c"
#include "weights/conv2d_17.c" // InputLayer is excluded
#include "conv2d_18.c"
#include "weights/conv2d_18.c" // InputLayer is excluded
#include "flatten_4.c" // InputLayer is excluded
#include "dense_8.c"
#include "weights/dense_8.c" // InputLayer is excluded
#include "dense_9.c"
#include "weights/dense_9.c"
#endif


void cnn(
  const input_t input,
  dense_9_output_type dense_9_output) {
  
  // Output array allocation
  static union {
    conv2d_16_output_type conv2d_16_output;
    conv2d_18_output_type conv2d_18_output;
    flatten_4_output_type flatten_4_output;
  } activations1;

  static union {
    conv2d_17_output_type conv2d_17_output;
    dense_8_output_type dense_8_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d_16( // Model input is passed as model parameter
    input,
    conv2d_16_kernel,
    conv2d_16_bias,
    activations1.conv2d_16_output
    );
  
  
  conv2d_17(
    activations1.conv2d_16_output,
    conv2d_17_kernel,
    conv2d_17_bias,
    activations2.conv2d_17_output
    );
  
  
  conv2d_18(
    activations2.conv2d_17_output,
    conv2d_18_kernel,
    conv2d_18_bias,
    activations1.conv2d_18_output
    );
  
  
  flatten_4(
    activations1.conv2d_18_output,
    activations1.flatten_4_output
    );
  
  
  dense_8(
    activations1.flatten_4_output,
    dense_8_kernel,
    dense_8_bias,
    activations2.dense_8_output
    );
  
  
  dense_9(
    activations2.dense_8_output,
    dense_9_kernel,
    dense_9_bias,// Last layer uses output passed as model parameter
    dense_9_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif