# Automated Musical Tempo Estimation (bpm-offset-detector)

Nathan Stephenson (@nathanstep55)

This is a program which detects BPM of music according to Bram van de Wetering's paper "Non-causal Beat Tracking for Rhythm Games," as well as the offset of the first beat.
This program is very useful for synchronizing things to music, as well as calculating tempo for a large amount of songs quickly.
This implementation is based on the one used in [ArrowVortex](https://arrowvortex.ddrnl.com/index.html), a chart/stepfile editor for rhythm games.

The paper is currently being rewritten with better dataset and research practices in order to be more useful, but the current version can be found in the `doc/syslab-version` folder.
Offset calculation is currently broken, and I will work to fix that.

The current implementation and scripts are required to be GPL due to their reliance on aubio as well as the fact that a good amount of aubio example code is used,
though I plan to rewrite it to not have a dependency on aubio and release it under LGPL, MIT or a similar license.

The GPL-3.0 license (included) applies to the `legacy` folder, `dataset` folder, `FindTempo_standalone.cpp`, `FindTempo_standalone.hpp` and `polyfit.h`.
The CC BY-SA 4.0 license applies to any file in the `doc` folder.
The PolyfitBoost library, originally by Patrick Loeber, is under the MIT license.