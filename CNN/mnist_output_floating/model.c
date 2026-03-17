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
#include "conv2d_141.c"
#include "weights/conv2d_141.c" // InputLayer is excluded
#include "conv2d_142.c"
#include "weights/conv2d_142.c" // InputLayer is excluded
#include "flatten_24.c" // InputLayer is excluded
#include "dense_46.c"
#include "weights/dense_46.c" // InputLayer is excluded
#include "dense_47.c"
#include "weights/dense_47.c"
#endif


void cnn(
  const input_t input,
  dense_47_output_type dense_47_output) {
  
  // Output array allocation
  static union {
    conv2d_141_output_type conv2d_141_output;
    dense_46_output_type dense_46_output;
  } activations1;

  static union {
    conv2d_142_output_type conv2d_142_output;
    flatten_24_output_type flatten_24_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d_141( // Model input is passed as model parameter
    input,
    conv2d_141_kernel,
    conv2d_141_bias,
    activations1.conv2d_141_output
    );
  
  
  conv2d_142(
    activations1.conv2d_141_output,
    conv2d_142_kernel,
    conv2d_142_bias,
    activations2.conv2d_142_output
    );
  
  
  flatten_24(
    activations2.conv2d_142_output,
    activations2.flatten_24_output
    );
  
  
  dense_46(
    activations2.flatten_24_output,
    dense_46_kernel,
    dense_46_bias,
    activations1.dense_46_output
    );
  
  
  dense_47(
    activations1.dense_46_output,
    dense_47_kernel,
    dense_47_bias,// Last layer uses output passed as model parameter
    dense_47_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif