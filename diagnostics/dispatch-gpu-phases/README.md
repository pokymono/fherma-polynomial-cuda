# Fresh worker dispatch GPU milestones

One ordinary multiplication per fresh harness call, with CUDA timing events
at input startup, completion of A high transforms, final input DMA, all input
transforms, the middle transform, and the first and final output DMA.
CPU timers measure caller allocation and dispatch joining. All operand work,
allocation, event queries and logging occur within the timed call.

GPU intervals use the executed startup event as their origin, not the CPU
entry time. Event insertion can perturb execution. These are diagnostic
milestones, not a ranked candidate or a replacement for standalone timing.
No extra multiplies, sleeps, inter-call GPU work or retained answers.

Official run `6af3c6a2698e99b16a80c3e7` passed20/20. Its1.183077ms outer median is slower than the1.009217ms standalone candidate. The added timing events affect scheduling; queue-count and stream-placement follow-ups did not confirm a substantial queue-aliasing bottleneck. See RESULTS.json.
