# Calibration-Free Sound Localization

Sound source localization on a robot body without measured or labeled acoustic data. Acoustic data is hard to collect, label and train on: every device has to be measured, every recording needs its true direction, and a trained model has to be retrained when the robot or the microphone positions change. The body's response is computed from CAD, the channel delays are estimated from unlabeled audio, and the direction is found by matching, with no trained model. Each microphone channel has an unknown gain, phase shift and delay, and none of them is calibrated by hand. Research internship at the KIST AI and Robotics Institute, Humanoid Research Center (2026).

![System](docs/system.png)

## Body responses from CAD

The responses of the four microphones come from a boundary element model of the body's CAD geometry, with no acoustic measurement. `build_replicas.py` takes them from the BEM table, computed with an earlier version of [stl2bem](https://github.com/Kwon-Junhyeong/stl2bem).

## Matching microphone-pair ratios

The recording is matched to the computed responses through microphone-pair ratios. A constant gain and phase on each channel multiply a pair ratio by one complex number, which the scorer fits and removes, so gain and phase need no calibration (`scorer.py`).

## Delays estimated without labels

On a body, a change of direction changes more than the delay, so direction and channel delay can be separated. The channel delays stay fixed across 1 s blocks while the source direction varies, so pooling the blocks estimates the delays without labels (`self_calibration.py`).

## Results

A 3D-printed box with four microphones was measured from 264 directions. Gains, phases and delays of up to 4 samples are then added to its channels in software (`unknown_channels.py`).

- With the delays estimated without labels, the median error is 3.7 degrees. Without that estimate it is 31.9 degrees, and SRP-PHAT without calibration gives 48.4 degrees.
- With the delays recovered from 10 s of unlabeled audio, the median error is 4.2 degrees.

On the public LOCATA recordings (Task 1, 13 recordings, eight of the 32 capsules of a spherical microphone array, 109 one-second blocks), the median error is 5.7 degrees with the BEM model of the array body and 5.8 degrees with the analytic model of a rigid sphere, against 7.9 degrees with a free-field model. `locata.py` produces the rigid-sphere and free-field results.

## Run

```sh
pip install -r requirements.txt
python build_replicas.py BEM_TABLE.h5      # data/box_replicas_5deg.npz
python build_blocks.py RECORDINGS_DIR      # data/box_blocks.npz
python unknown_channels.py                 # each matcher under added gains, phases and delays
python self_calibration.py                 # delays from unlabeled blocks, then the directions
python build_locata_blocks.py LOCATA_DIR   # data/locata_blocks.npz from the LOCATA corpus
python locata.py                           # rigid-sphere and free-field matching on LOCATA
```

The box's BEM table and recordings are not part of the repository. The LOCATA corpus is public.
