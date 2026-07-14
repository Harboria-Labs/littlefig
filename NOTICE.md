# Little Fig Licensing Notice

Copyright (C) 2024-2026 Harboria Labs (0xticketguy).

Little Fig is distributed as a multi-license research and software stack. The license that applies depends on the component you use.

This notice is intended to make those boundaries explicit. If a file contains a more specific license header or sits under a component-local license file, that more specific notice controls.

## Component license map

| Component | Paths | License |
| --- | --- | --- |
| Ember's Diaries integration and Memory Fabric core IP | `src/little_fig/engine/ember_integration.py`, `src/little_fig/engine/memory_fabric.py`, related Memory Fabric adapter code | AGPL-3.0-or-later, with a commercial license option from Harboria Labs |
| Fig Engine training infrastructure | `src/little_fig/engine/` except AGPL-covered Memory Fabric / Ember integration files | Apache-2.0 |
| CogMemBench benchmark code and dataset | `cogmembench/` | MIT |
| Research papers and publication drafts | `paper/`, research writeups in `cogmembench/` | CC-BY-4.0 |
| Repository glue, packaging, tests, scripts, demos, and web UI | root files, `scripts/`, `tests/`, `src/little_fig/web/`, and other files not listed above | AGPL-3.0-or-later unless a more specific file or directory notice says otherwise |

## Commercial licensing

The AGPL-covered core IP is available under a dual-license model. You may use it under AGPL-3.0-or-later, or you may contact Harboria Labs for a separate commercial license if you want to use the AGPL-covered components in a proprietary product or hosted service without AGPL obligations.

See `COMMERCIAL_LICENSE.md` for the commercial-license notice.

## Papers and ideas

The research papers are licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0). The papers describe ideas, methods, and results. Implementations of those ideas may still be covered by the software licenses listed above.

## No warranty

All components are provided without warranty, to the maximum extent permitted by the applicable license and law.
