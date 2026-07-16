# Fig Engine Component License

Copyright (C) 2024-2026 Harboria Labs (0xticketguy).

Most Fig Engine training infrastructure in this directory is licensed under:

    Apache License 2.0
    SPDX-License-Identifier: Apache-2.0

This includes the CPU/GPU training, quantization, kernel, pipeline, optimizer, tier selection, packing, and model-loading infrastructure.

## AGPL-covered core IP exceptions

The following files are part of the Ember's Diaries / Memory Fabric core IP and are licensed under AGPL-3.0-or-later, with a commercial license option from Harboria Labs:

- `ember_integration.py`
- `memory_fabric.py`
- Memory Fabric adapter code that directly implements or depends on the Memory Fabric architecture

For the full repository license map, see `../../../NOTICE.md`.

For the Apache-2.0 license text, see:

    https://www.apache.org/licenses/LICENSE-2.0

For the AGPL-3.0 license text, see:

    https://www.gnu.org/licenses/agpl-3.0.html
