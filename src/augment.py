from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PolarityInversion,
    PitchShift,
)

_train_transforms = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PolarityInversion(p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
])

_val_transforms = Compose([

])

# make transforms serializable for
# pytorch lightning cli compatability

def train_transforms(*args, **kwargs):
    return _train_transforms(*args, **kwargs)

def val_transforms(*args, **kwargs):
    return _val_transforms(*args, **kwargs)
