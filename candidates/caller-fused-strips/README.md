# Caller construction with fused output strips

Four copy workers prepare each call's inputs while the existing dispatch
worker queues GPU work and the caller constructs its ordinary output vector.
The main point uses32-column boundary and64-column interior output strips.
The small point uses64-column strips and a shared-memory inverse transpose
covering two fields and eight columns per block. Every worker and GPU operation
joins before returning. The generated harness and output ownership are unchanged.

This candidate combines independently measured components and requires its
own full exactness and standalone official checks. No ranked result yet.

Validation:141local exact cases passed, plus48AWS exact cases with the final default build. Source and binary hashes are recorded in VALIDATION.json.
