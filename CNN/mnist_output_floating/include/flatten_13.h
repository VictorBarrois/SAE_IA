/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_13_H_
#define _FLATTEN_13_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 32

typedef float flatten_13_output_type[OUTPUT_DIM];

#if 0
void flatten_13(
  const number_t input[1][1][32], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_13_H_