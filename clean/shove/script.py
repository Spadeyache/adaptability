"""Label-blind fixed shove program. No simulator or property-table access."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    cap_n: float
    dt: float = .001
    approach_s: float = .8
    preload_s: float = .1
    push_s: float = .08
    approach_speed_m_s: float = .005
    preload_force_n: float = .03
    drive_fraction: float = .95
    stop_speed_m_s: float = .001
    stop_angular_speed_rad_s: float = .05
    stop_confirmation_s: float = .1
    maximum_slide_s: float = 5.


def commands(settings):
    """Exactly the same force/time sequence for every cube and repeat."""
    for phase, seconds, force in [
        (0, settings.approach_s, 0.),
        (1, settings.preload_s, settings.preload_force_n),
        (2, settings.push_s, settings.drive_fraction * settings.cap_n),
    ]:
        for _ in range(round(seconds / settings.dt)):
            yield phase, force
