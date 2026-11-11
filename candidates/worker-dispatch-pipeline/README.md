# worker-dispatch-pipeline

Ordinary output vector construction runs on the caller for the main point. The existing GPU submission sequence runs on the output worker, concurrently with caller construction. The worker initializes its CUDA context once; all operand work stays inside the measured call. Its request joins before call state is destroyed.

The smaller point retains the fused-output-pipeline implementation. All 141 local exact checks passed, including 48 cases at the two challenge points. AWS validation passed 48 cases for each CPU worker configuration (four and seven); these are proxy timings, not official scores.
