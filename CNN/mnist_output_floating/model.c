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
#include "conv2d_5.c"
#include "weights/conv2d_5.c" // InputLayer is excluded
#include "max_pooling2d_4.c" // InputLayer is excluded
#include "conv2d_6.c"
#include "weights/conv2d_6.c" // InputLayer is excluded
#include "max_pooling2d_5.c" // InputLayer is excluded
#include "flatten_2.c" // InputLayer is excluded
#include "dense_4.c"
#include "weights/dense_4.c"
#endif


void cnn(
  const input_t input,
  dense_4_output_type dense_4_output) {
  
  // Output array allocation
  static union {
    conv2d_5_output_type conv2d_5_output;
    conv2d_6_output_type conv2d_6_output;
  } activations1;

  static union {
    max_pooling2d_4_output_type max_pooling2d_4_output;
    max_pooling2d_5_output_type max_pooling2d_5_output;
    flatten_2_output_type flatten_2_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d_5( // Model input is passed as model parameter
    input,
    conv2d_5_kernel,
    conv2d_5_bias,
    activations1.conv2d_5_output
    );
  
  
  max_pooling2d_4(
    activations1.conv2d_5_output,
    activations2.max_pooling2d_4_output
    );
  
  
  conv2d_6(
    activations2.max_pooling2d_4_output,
    conv2d_6_kernel,
    conv2d_6_bias,
    activations1.conv2d_6_output
    );
  
  
  max_pooling2d_5(
    activations1.conv2d_6_output,
    activations2.max_pooling2d_5_output
    );
  
  
  flatten_2(
    activations2.max_pooling2d_5_output,
    activations2.flatten_2_output
    );
  
  
  dense_4(
    activations2.flatten_2_output,
    dense_4_kernel,
    dense_4_bias,// Last layer uses output passed as model parameter
    dense_4_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif