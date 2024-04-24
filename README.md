# WAVENETS
My implementation of models from wavenet family using tensorflow2

Currently only supports VCTK dataset. To create downsampled dataset use `dev/downsampled_dataset.py` script. This code uses multiple GPU by default.

Currently implemented models:
- WaveNet
- WaveNet with multiple dilated convolutions in each layer
- WaveNet without skip connections
- Wavenet with output distribution as mixture of logistic distributions
- Wavenet with output distribution as mixture of normal distributions
- WaveNet with global conditioning
- WaveNet with local conditioning (not tested)
- Combination of above models

Fast (queued) wavenet is not yet implemented. The main problem is the multiple dilated convolutions in each layer. For possible implementation see [older version](https://github.com/jirsat/wavenets/blob/f8b5798f06ffd90b07aca937ed452563b1db2c1b/src/fastwavenet/glob_cond_wavenet.py)

## TODOs

- Implement fast wavenet
- Implement possibility to use different datasets
- Test local conditioning