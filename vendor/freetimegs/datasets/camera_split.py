"""Person4D extension: an explicit all-camera production training split."""


def camera_split(num_cameras: int, test_every: int) -> tuple[list[int], list[int], bool]:
    """Return training cameras, diagnostic cameras, and whether diagnostics hold out views.

    A zero interval fits every generated view and diagnoses camera zero using
    training observations. Positive intervals retain the upstream holdout split.
    """
    if num_cameras < 1 or test_every < 0:
        raise ValueError("A positive camera count and nonnegative test interval are required")
    if test_every == 0:
        return list(range(num_cameras)), [0], False
    validation = list(range(0, num_cameras, test_every))
    training = [index for index in range(num_cameras) if index not in validation]
    if not training:
        raise ValueError("The holdout interval leaves no training cameras; use zero to fit every view")
    return training, validation, True
