# Helpful audio augmentation methods
# Common approaches in the context of bioacoustics
    # Mixup: combines training pairs and their examples - https://arxiv.org/abs/1710.09412
    # Additive noise - Pink noise (preferred), white noise, even just sampling of background noise ("silence")
    # Pitch shift (careful!)
    # Echo/reverb 
    # Time/frquency masking

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
from numpy.typing import NDArray


# Generate 2 seconds of dummy audio for the sake of example
samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

class Compose(Compose):
    def __init__(self, transforms, p=1.0, shuffle=False, seperate=False):
        super().__init__(transforms, p, shuffle)
        self.transforms = transforms
        self.seperate = seperate

    def __call__(self, samples: NDArray[np.float32], sample_rate: int):
        if self.seperate:
            augmented_samples = []
            for transform in self.transforms:
                samples = transform(samples, sample_rate)
                augmented_samples.append(samples)
            samples = np.array(augmented_samples)
        else:
            for transform in self.transforms:
                samples = transform(samples, sample_rate)
        return samples
    

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5)
], seperate=True)

augmented_samples = augment(samples=samples, sample_rate=16000)
print(augmented_samples.shape)