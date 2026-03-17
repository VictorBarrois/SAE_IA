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
#include "conv2d_148.c"
#include "weights/conv2d_148.c" // InputLayer is excluded
#include "conv2d_149.c"
#include "weights/conv2d_149.c" // InputLayer is excluded
#include "flatten_26.c" // InputLayer is excluded
#include "dense_50.c"
#include "weights/dense_50.c" // InputLayer is excluded
#include "dense_51.c"
#include "weights/dense_51.c"
#endif


void cnn(
  const input_t input,
  dense_51_output_type dense_51_output) {
  
  // Output array allocation
  static union {
    conv2d_148_output_type conv2d_148_output;
    dense_50_output_type dense_50_output;
  } activations1;

  static union {
    conv2d_149_output_type conv2d_149_output;
    flatten_26_output_type flatten_26_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d_148( // Model input is passed as model parameter
    input,
    conv2d_148_kernel,
    conv2d_148_bias,
    activations1.conv2d_148_output
    );
  
  
  conv2d_149(
    activations1.conv2d_148_output,
    conv2d_149_kernel,
    conv2d_149_bias,
    activations2.conv2d_149_output
    );
  
  
  flatten_26(
    activations2.conv2d_149_output,
    activations2.flatten_26_output
    );
  
  
  dense_50(
    activations2.flatten_26_output,
    dense_50_kernel,
    dense_50_bias,
    activations1.dense_50_output
    );
  
  
  dense_51(
    activations1.dense_50_output,
    dense_51_kernel,
    dense_51_bias,// Last layer uses output passed as model parameter
    dense_51_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif