# Transfer topology probe

Read-only setup diagnostics for the allowed CPU cores, GPU PCI NUMA locality,
and physical NUMA node of this implementation's own pinned buffers. PCI IDs
are normalized to lowercase to match sysfs. An unavailable memory-policy syscall
is reported with its errno, not interpreted as a node assignment. Multiplication
and the generated harness are unchanged. Use the unscored profile implementation.
