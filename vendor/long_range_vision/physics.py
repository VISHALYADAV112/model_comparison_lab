from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImagingInputs:
    sensor_height_px: int
    vertical_fov_deg: float
    target_height_m: float
    range_m: float
    aperture_mm: float | None = None
    wavelength_nm: float = 550.0
    fried_parameter_cm: float | None = None
    focal_length_mm: float | None = None
    pixel_pitch_um: float | None = None
    exposure_ms: float | None = None
    angular_rate_mrad_s: float | None = None


def target_pixels(
    sensor_height_px: int,
    vertical_fov_deg: float,
    target_height_m: float,
    range_m: float,
) -> float:
    if sensor_height_px <= 0 or not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError("Sensor height must be positive and vertical FOV must be between 0 and 180 degrees")
    if target_height_m <= 0 or range_m <= 0:
        raise ValueError("Target height and range must be positive")
    field_height_m = 2.0 * range_m * math.tan(math.radians(vertical_fov_deg) / 2.0)
    return sensor_height_px * target_height_m / field_height_m


def angular_pixel_pitch_rad(sensor_height_px: int, vertical_fov_deg: float) -> float:
    return math.radians(vertical_fov_deg) / sensor_height_px


def detail_band(person_height_px: float) -> str:
    if person_height_px < 3:
        return "sub-detection: use motion/temporal evidence; appearance is not reliable"
    if person_height_px < 8:
        return "weak detection: blob-level evidence with high false-alarm risk"
    if person_height_px < 20:
        return "tiny-person detection: class may be detectable; identity is not supported"
    if person_height_px < 50:
        return "coarse appearance: pose and broad clothing regions may be measurable"
    return "recognition candidate: evaluate face/body pixels separately; identity is not guaranteed"


def imaging_report(inputs: ImagingInputs) -> dict[str, object]:
    px = target_pixels(
        inputs.sensor_height_px,
        inputs.vertical_fov_deg,
        inputs.target_height_m,
        inputs.range_m,
    )
    wavelength_m = inputs.wavelength_nm * 1e-9
    target_angle_rad = inputs.target_height_m / inputs.range_m
    angular_px = angular_pixel_pitch_rad(inputs.sensor_height_px, inputs.vertical_fov_deg)
    field_factor = 2.0 * math.tan(math.radians(inputs.vertical_fov_deg) / 2.0)
    range_by_target_px = {
        f"{threshold}px": inputs.sensor_height_px * inputs.target_height_m / (threshold * field_factor)
        for threshold in (64, 32, 16, 8, 4, 2)
    }

    report: dict[str, object] = {
        "inputs": asdict(inputs),
        "target_angular_height_mrad": target_angle_rad * 1000.0,
        "angular_pixel_pitch_urad": angular_px * 1e6,
        "nominal_target_height_px": px,
        "sampling_limited_cycles_across_target": px / 2.0,
        "range_m_at_nominal_target_height": range_by_target_px,
        "approximate_feature_sampling_px": {
            "head_height_assuming_14_percent_of_body": px * 0.14,
            "face_width_assuming_9_percent_of_body": px * 0.09,
            "note": "Planning ratios only; pose and anatomy vary. Pixels do not guarantee usable contrast or identity.",
        },
        "detail_band": detail_band(px),
        "warning": (
            "Nominal pixels are an upper bound before lens MTF, defocus, atmosphere, motion, "
            "compression, noise, and demosaicing. Generative enhancement cannot turn inferred "
            "texture into measured evidence."
        ),
    }

    limits: list[float] = [px / 2.0]
    if inputs.aperture_mm:
        aperture_m = inputs.aperture_mm / 1000.0
        rayleigh_rad = 1.22 * wavelength_m / aperture_m
        diffraction_spots = target_angle_rad / rayleigh_rad
        limits.append(diffraction_spots / 2.0)
        report["diffraction"] = {
            "rayleigh_angle_urad": rayleigh_rad * 1e6,
            "rayleigh_ground_resolution_m": rayleigh_rad * inputs.range_m,
            "resolvable_spots_across_target": diffraction_spots,
        }

    if inputs.fried_parameter_cm:
        r0_m = inputs.fried_parameter_cm / 100.0
        seeing_rad = 0.98 * wavelength_m / r0_m
        turbulence_spots = target_angle_rad / seeing_rad
        limits.append(turbulence_spots / 2.0)
        report["long_exposure_turbulence"] = {
            "seeing_angle_urad": seeing_rad * 1e6,
            "seeing_ground_resolution_m": seeing_rad * inputs.range_m,
            "resolvable_spots_across_target": turbulence_spots,
            "note": "Fried parameter must be measured or estimated for the actual path and wavelength.",
        }

    if inputs.focal_length_mm and inputs.pixel_pitch_um:
        focal_m = inputs.focal_length_mm / 1000.0
        pitch_m = inputs.pixel_pitch_um * 1e-6
        sensor_height_m = inputs.sensor_height_px * pitch_m
        implied_vfov_deg = math.degrees(2.0 * math.atan(sensor_height_m / (2.0 * focal_m)))
        geometry_px = focal_m * inputs.target_height_m / (inputs.range_m * pitch_m)
        fov_difference = abs(implied_vfov_deg - inputs.vertical_fov_deg) / inputs.vertical_fov_deg
        report["sensor_geometry"] = {
            "target_height_px_from_focal_pitch": geometry_px,
            "instantaneous_fov_urad_per_pixel": pitch_m / focal_m * 1e6,
            "implied_vertical_fov_deg": implied_vfov_deg,
            "fov_input_relative_difference": fov_difference,
            "geometry_consistent": fov_difference <= 0.05,
        }
        if fov_difference > 0.05:
            report["sensor_geometry"]["warning"] = (  # type: ignore[index]
                "The supplied focal length, pixel pitch, sensor height and vertical FOV do not describe the same "
                "camera mode. Use measured FOV or correct the sensor/lens values before treating this as a limit."
            )
        if inputs.aperture_mm:
            f_number = inputs.focal_length_mm / inputs.aperture_mm
            airy_diameter_px = 2.44 * wavelength_m * f_number / pitch_m
            report["diffraction"]["airy_diameter_px"] = airy_diameter_px  # type: ignore[index]

    if inputs.exposure_ms is not None and inputs.angular_rate_mrad_s is not None:
        smear_rad = inputs.angular_rate_mrad_s * 1e-3 * inputs.exposure_ms * 1e-3
        smear_px = smear_rad / angular_px
        motion_limited_cycles = px / (2.0 * max(1.0, smear_px))
        limits.append(motion_limited_cycles)
        report["motion"] = {
            "smear_px": smear_px,
            "smear_urad": smear_rad * 1e6,
            "motion_limited_cycles_across_target": motion_limited_cycles,
        }

    report["conservative_independent_cycles_across_target"] = min(limits)
    return report
