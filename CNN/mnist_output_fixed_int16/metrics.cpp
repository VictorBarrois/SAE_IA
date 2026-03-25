#ifndef SINGLE_FILE
#include "metrics.h"

#include "Metrics/Metric.h"
#include "model.h"
#endif


#include "Metrics/Accuracy.h"



static Accuracy<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> metric_1;


std::array<Metric<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES>*, 1> metrics {

  &metric_1,

};