#!/usr/bin/env python3
"""Short repeat-prediction workload for a Core AI Instruments runtime trace."""
import sys
import coremltools as ct
import numpy as np

model = ct.models.MLModel(sys.argv[1], compute_units=ct.ComputeUnit.CPU_AND_NE)
pixels = np.load(sys.argv[2])["pixel_values"]
for i in range(100):
    result = model.predict({"pixel_values": pixels})
    if i % 10 == 0:
        print("prediction", i, flush=True)
print("completed", flush=True)
