# Controlled idle-period diagnostic

Unscored research only. Compare a complete baseline/tuned multiplication after
0,1,5,20 and50ms idle periods. Each measured inner call follows two verified
warm calls. Short and long delays alternate, with a zero-delay bracket at the
end. Setup, sleeps, all warm calls and output checking remain inside the outer
harness clock; the outer latency is not a competitive score. Only the explicitly
labelled inner timings are used to investigate sensitivity to idle time.

Inputs are reused only within the current diagnostic call; no operand or answer
is retained across outer calls. Every result is compared, and the unchanged
harness independently verifies the returned answer. CPU cache reuse means this
experiment isolates an idle-period effect rather than reproducing fresh-input
standalone latency. A difference would not identify PCIe power state by itself.
No keep-warm background thread is introduced.

Official unscored run `6af37295698e99b16a7cc4d4` passed 3/3 cases.
Across measured outer calls, baseline inner medians were 1.032559 ms without
idle and 1.263321 ms after 20 ms idle; tuned medians were 1.018680 and
1.336714 ms respectively. The delayed samples varied widely (roughly
1.026–1.563 ms for tuned/20 ms). This establishes idle sensitivity in this
experiment, not its cause or a standalone candidate score. Full samples and
counts are in RESULTS.json. The next check separates CPU and GPU phase timing
and varies a startup transfer performed inside the timed call.
