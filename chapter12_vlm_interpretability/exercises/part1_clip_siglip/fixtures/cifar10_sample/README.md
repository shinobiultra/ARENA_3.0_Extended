# CIFAR-10 lesson fixture

These 200 RGB PNGs and `manifest.json` losslessly represent the lesson's former
`cifar10_sample.npz`: same row order, uint8 [200,32,32,3] image bytes and int64
[200] label bytes. No resizing, colour conversion, resampling or new selection
was performed. Each file, decoded pixel array and encoded label has a SHA-256;
the manifest also records the complete array hashes and original NPZ hash.

Dataset citation retained from the original helper: Alex Krizhevsky (2009),
*Learning Multiple Layers of Features from Tiny Images*.
Canonical dataset page: <https://www.cs.toronto.edu/~kriz/cifar.html>.
The historical helper described a fixed random test-set sample with 20 images
per class. Its original dataset indices and random seed were not recorded.
`fixture_index` is the original NPZ row position; `original_cifar10_index` is
explicitly null. This conversion does not establish missing acquisition provenance.

The preserved original NPZ/helper locations, exact conversion proof and historical
execution boundary are recorded in `docs/mastery/evidence/cifar_fixture_20261019/`.
Current loader validation establishes array identity; it does not claim a rerun
of the historical CLIP/SigLIP experiments.
