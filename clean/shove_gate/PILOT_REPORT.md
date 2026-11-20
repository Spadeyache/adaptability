# Direct-force ten-cube pilot (revision 5)

The direct-force pilot passed its collection conditions. The subsequent full Plain–Oracle prediction gate failed; see [RESULTS_DIRECT.md](RESULTS_DIRECT.md).

- Same ten cube identities, 100 shoves; mass 0.18–0.50 kg and independently sampled friction 0.4–1.0.
- No pusher. Horizontal COM force uniform 4–7 N; duration uniform on the 60, 65, …, 100 ms grid; randomized direction/start/yaw.
- Launches above 2 cm/s: 100/100.
- Maximum coast distance: 0.692389m.
- Moving coast after trimming: 5275/5775 (91.34%).
- Trimmed stationary transitions: 24,225; retained 7,345.
- The actions-only probe is saved in [action_probe.json](results/action_probe.json). The probes found no useful predictive signal in the commands; observed trimmed length is an excluded side channel.

The passed pilot authorized collection of all 230 cubes and the five-seed comparison. All 2,300 shoves launched and stopped; three training episodes coasted over 1 m (maximum 1.449 m) and were retained. The prediction gate has now been evaluated once; no belief model was trained.

Previous pusher-pilot reports are preserved under versions/v4/ and earlier versions.
