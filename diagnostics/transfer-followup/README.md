# Transfer follow-up diagnostic

Unscored comparison of the retained implementation, widened Shoup differences,
size-specific mapped input, and a flattened output graph. Every inner call runs
the complete multiplication and returns a newly allocated result. Outputs are
compared within the current call, and the unchanged platform harness verifies
the returned answer independently. Parameter setup and logging are included in
the outer harness time; inner measurements are diagnostics, not ranked scores.

Each mode has one state at a time to avoid competing idle worker teams. Five
repetitions expose warm-up behavior, and the baseline brackets the alternatives.
No operand or answer persists across calls. Use only the profile implementation.
