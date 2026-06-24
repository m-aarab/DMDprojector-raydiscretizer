#!/usr/bin/env python3
"""Generate proprietary .ry ray files from a TOML projection configuration.

This script unifies multiple legacy ray-generator behaviors into a single,
config-driven tool similar in spirit to the mesh generators.

Usage:
  python projectionImageGen.py --config projectionImageGen.example.toml
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
	import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
	try:
		import tomli as tomllib  # type: ignore
	except ModuleNotFoundError as exc:  # pragma: no cover
		raise SystemExit(
			"TOML support requires Python 3.11+ (tomllib) or package 'tomli'."
		) from exc


Vec2f = tuple[float, float]
Vec2i = tuple[int, int]


@dataclass(frozen=True)
class ProjectionConfig:
	width: float
	height: float
	axis: str
	direction: int
	focus_plane_position: float
	source_plane_position: float
	center: Vec2f


@dataclass(frozen=True)
class MaskConfig:
	type: str
	offset: Vec2f
	rectangle_width: float | None
	rectangle_height: float | None
	circle_radius: float | None
	triangle_base: float | None
	triangle_height: float | None
	annulus_outer_radius: float | None
	annulus_inner_radius: float | None


@dataclass(frozen=True)
class EmissionConfig:
	origin_type: str
	base_intensity: float


@dataclass(frozen=True)
class FocusConfig:
	mode: str
	cone_half_angle_deg: float


@dataclass(frozen=True)
class ContinuousConfig:
	n_rays: int


@dataclass(frozen=True)
class PixelConfig:
	rays_per_pixel: int
	pitch: Vec2f
	size: Vec2f
	projector_shift_max_pixels: Vec2f
	bins_per_pixel: Vec2i
	activation_mode: str
	aa_samples: int
	intensity_mode: str
	gaussian_size_type: str


@dataclass(frozen=True)
class OutputConfig:
	output_folder: Path
	output_filename: str
	batch_count: int
	start_index: int
	append_index: bool
	zero_pad: int
	seed: int | None
	overwrite: bool
	normalize_batch_energy_to_mask_power: bool


@dataclass(frozen=True)
class RayGeneratorConfig:
	projection: ProjectionConfig
	mask: MaskConfig
	emission: EmissionConfig
	focus: FocusConfig
	continuous: ContinuousConfig | None
	pixel: PixelConfig | None
	output: OutputConfig


def _get_first(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
	for key in keys:
		if key in mapping:
			return mapping[key]
	return None


def _require_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
	value = raw.get(key)
	if not isinstance(value, dict):
		raise ValueError(f"Config must contain a [{key}] table.")
	return value


def _parse_float(value: Any, name: str) -> float:
	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"'{name}' must be numeric.") from exc


def _parse_positive_float(value: Any, name: str) -> float:
	out = _parse_float(value, name)
	if out <= 0.0:
		raise ValueError(f"'{name}' must be > 0.")
	return out


def _parse_nonnegative_float(value: Any, name: str) -> float:
	out = _parse_float(value, name)
	if out < 0.0:
		raise ValueError(f"'{name}' must be >= 0.")
	return out


def _parse_int(value: Any, name: str) -> int:
	if isinstance(value, bool):
		raise ValueError(f"'{name}' must be an integer.")
	try:
		return int(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"'{name}' must be an integer.") from exc


def _parse_positive_int(value: Any, name: str) -> int:
	out = _parse_int(value, name)
	if out <= 0:
		raise ValueError(f"'{name}' must be > 0.")
	return out


def _parse_nonnegative_int(value: Any, name: str) -> int:
	out = _parse_int(value, name)
	if out < 0:
		raise ValueError(f"'{name}' must be >= 0.")
	return out


def _parse_bool(value: Any, name: str) -> bool:
	if not isinstance(value, bool):
		raise ValueError(f"'{name}' must be a boolean.")
	return value


def _parse_vec2(value: Any, name: str, positive_only: bool = False) -> Vec2f:
	if not isinstance(value, (list, tuple)) or len(value) != 2:
		raise ValueError(f"'{name}' must be a 2-value array.")
	try:
		out = (float(value[0]), float(value[1]))
	except (TypeError, ValueError) as exc:
		raise ValueError(f"'{name}' must only contain numeric values.") from exc
	if positive_only and (out[0] <= 0.0 or out[1] <= 0.0):
		raise ValueError(f"'{name}' values must be > 0.")
	return out


def _parse_vec2_positive_int_or_scalar(value: Any, name: str) -> Vec2i:
	if isinstance(value, bool):
		raise ValueError(f"'{name}' must be a positive integer or 2-value integer array.")

	if isinstance(value, int):
		n = _parse_positive_int(value, name)
		return n, n

	if isinstance(value, (list, tuple)) and len(value) == 2:
		n0 = _parse_positive_int(value[0], f"{name}[0]")
		n1 = _parse_positive_int(value[1], f"{name}[1]")
		return n0, n1

	raise ValueError(f"'{name}' must be a positive integer or 2-value integer array.")


def _parse_vec2_nonnegative_float_or_scalar(value: Any, name: str) -> Vec2f:
	if isinstance(value, bool):
		raise ValueError(f"'{name}' must be a non-negative number or 2-value numeric array.")

	if isinstance(value, (int, float)):
		n = _parse_nonnegative_float(value, name)
		return n, n

	if isinstance(value, (list, tuple)) and len(value) == 2:
		n0 = _parse_nonnegative_float(value[0], f"{name}[0]")
		n1 = _parse_nonnegative_float(value[1], f"{name}[1]")
		return n0, n1

	raise ValueError(f"'{name}' must be a non-negative number or 2-value numeric array.")


def _parse_axis(value: Any) -> str:
	axis = str(value).strip().lower()
	if axis not in {"x", "y", "z"}:
		raise ValueError("'projection.axis' must be one of: x, y, z.")
	return axis


def _parse_direction(value: Any) -> int:
	if isinstance(value, str):
		text = value.strip().lower()
		if text in {"+", "+1", "pos", "positive", "forward"}:
			return 1
		if text in {"-", "-1", "neg", "negative", "backward"}:
			return -1
		raise ValueError("'projection.direction' string must be +1 or -1 style.")

	sign = _parse_int(value, "projection.direction")
	if sign == 0:
		raise ValueError("'projection.direction' must be either +1 or -1.")
	return 1 if sign > 0 else -1


def _parse_mode(value: Any, name: str, allowed: set[str], aliases: dict[str, str] | None = None) -> str:
	mode = str(value).strip().lower().replace("-", "_")
	if aliases is not None:
		mode = aliases.get(mode, mode)
	if mode not in allowed:
		allowed_text = ", ".join(sorted(allowed))
		raise ValueError(f"'{name}' must be one of: {allowed_text}.")
	return mode


def _resolve_output_folder(config_path: Path, raw_path: str) -> Path:
	out_dir = Path(raw_path)
	if not out_dir.is_absolute():
		out_dir = (config_path.parent / out_dir).resolve()
	return out_dir


def _load_toml(path: Path) -> dict[str, Any]:
	try:
		raw = tomllib.loads(path.read_text(encoding="utf-8"))
	except FileNotFoundError as exc:
		raise SystemExit(f"Config file not found: {path}") from exc
	except Exception as exc:
		raise SystemExit(f"Failed to parse TOML config '{path}': {exc}") from exc
	if not isinstance(raw, dict):
		raise ValueError("Top-level TOML content must be a table.")
	return raw


def load_config(config_path: Path) -> RayGeneratorConfig:
	raw = _load_toml(config_path)

	projection_raw = _require_table(raw, "projection")
	emission_raw = _require_table(raw, "emission")
	mask_raw = _require_table(raw, "mask")
	output_raw = _require_table(raw, "output")

	projection = ProjectionConfig(
		width=_parse_positive_float(_get_first(projection_raw, ("width",)), "projection.width"),
		height=_parse_positive_float(_get_first(projection_raw, ("height",)), "projection.height"),
		axis=_parse_axis(_get_first(projection_raw, ("axis",))),
		direction=_parse_direction(_get_first(projection_raw, ("direction", 1))),
		focus_plane_position=_parse_float(
			_get_first(projection_raw, ("focus_plane_position", "focus_plane", "focus_pos")),
			"projection.focus_plane_position",
		),
		source_plane_position=_parse_float(
			_get_first(projection_raw, ("source_plane_position", "source_plane", "source_pos")),
			"projection.source_plane_position",
		),
		center=_parse_vec2(_get_first(projection_raw, ("center", "center_uv", "offset")) or [0.0, 0.0], "projection.center"),
	)

	origin_type = _parse_mode(
		_get_first(emission_raw, ("origin_type", "mode")),
		"emission.origin_type",
		allowed={"continuous", "pixelated"},
		aliases={"pixels": "pixelated", "pixel": "pixelated"},
	)
	base_intensity = _parse_positive_float(
		_get_first(emission_raw, ("base_intensity", "intensity", "ibase")),
		"emission.base_intensity",
	)
	emission = EmissionConfig(origin_type=origin_type, base_intensity=base_intensity)

	focus_raw_any = raw.get("focus")
	if focus_raw_any is None:
		focus_raw: dict[str, Any] = {}
	elif isinstance(focus_raw_any, dict):
		focus_raw = focus_raw_any
	else:
		raise ValueError("[focus] must be a table when present.")

	focus_mode = _parse_mode(
		_get_first(focus_raw, ("mode", "ray_mode")) or "collimated",
		"focus.mode",
		allowed={"collimated", "focused"},
	)
	cone_half_angle_deg = _parse_nonnegative_float(
		_get_first(focus_raw, ("cone_half_angle_deg", "cone_angle_deg", "theta_target_deg")) or 0.0,
		"focus.cone_half_angle_deg",
	)
	if focus_mode == "focused" and cone_half_angle_deg <= 0.0:
		raise ValueError("focus.cone_half_angle_deg must be > 0 when focus.mode=focused")
	focus = FocusConfig(mode=focus_mode, cone_half_angle_deg=cone_half_angle_deg)

	mask_type = _parse_mode(
		_get_first(mask_raw, ("type", "shape")),
		"mask.type",
		allowed={"rectangle", "circle", "triangle", "annulus"},
		aliases={"circle_with_hole": "annulus", "ring": "annulus", "rect": "rectangle"},
	)
	mask = MaskConfig(
		type=mask_type,
		offset=_parse_vec2(_get_first(mask_raw, ("offset", "center_offset")) or [0.0, 0.0], "mask.offset"),
		rectangle_width=(
			_parse_positive_float(mask_raw["width"], "mask.width") if "width" in mask_raw else None
		),
		rectangle_height=(
			_parse_positive_float(mask_raw["height"], "mask.height") if "height" in mask_raw else None
		),
		circle_radius=(
			_parse_positive_float(mask_raw["radius"], "mask.radius") if "radius" in mask_raw else None
		),
		triangle_base=(
			_parse_positive_float(mask_raw["base"], "mask.base") if "base" in mask_raw else None
		),
		triangle_height=(
			_parse_positive_float(mask_raw["height"], "mask.height") if "height" in mask_raw else None
		),
		annulus_outer_radius=(
			_parse_positive_float(mask_raw["outer_radius"], "mask.outer_radius")
			if "outer_radius" in mask_raw
			else None
		),
		annulus_inner_radius=(
			_parse_nonnegative_float(mask_raw["inner_radius"], "mask.inner_radius")
			if "inner_radius" in mask_raw
			else None
		),
	)

	output_filename_raw = _get_first(output_raw, ("output_filename", "filename", "file", "output"))
	if not isinstance(output_filename_raw, str) or not output_filename_raw.strip():
		raise ValueError("output.output_filename must be a non-empty string.")
	output_filename = output_filename_raw.strip()
	if not output_filename.endswith(".ry"):
		output_filename = f"{Path(output_filename).stem}.ry"

	output_folder_raw = _get_first(output_raw, ("output_folder", "output_dir", "folder")) or "."
	if not isinstance(output_folder_raw, str) or not output_folder_raw.strip():
		raise ValueError("output.output_folder must be a non-empty string.")

	batch_count = _parse_positive_int(_get_first(output_raw, ("batch_count", "count")) or 1, "output.batch_count")
	start_index = _parse_nonnegative_int(_get_first(output_raw, ("start_index", "id")) or 0, "output.start_index")
	append_index = _parse_bool(
		_get_first(output_raw, ("append_index", "append_id"))
		if _get_first(output_raw, ("append_index", "append_id")) is not None
		else (batch_count > 1),
		"output.append_index",
	)
	zero_pad = _parse_nonnegative_int(_get_first(output_raw, ("zero_pad", "pad")) or 0, "output.zero_pad")
	seed_raw = _get_first(output_raw, ("seed", "random_seed"))
	seed = None if seed_raw is None else _parse_int(seed_raw, "output.seed")
	overwrite = _parse_bool(_get_first(output_raw, ("overwrite",)) if "overwrite" in output_raw else False, "output.overwrite")
	normalize_batch_energy_to_mask_power = _parse_bool(
		_get_first(output_raw, ("normalize_batch_energy_to_mask_power", "normalize_energy"))
		if _get_first(output_raw, ("normalize_batch_energy_to_mask_power", "normalize_energy")) is not None
		else False,
		"output.normalize_batch_energy_to_mask_power",
	)

	output = OutputConfig(
		output_folder=_resolve_output_folder(config_path, output_folder_raw),
		output_filename=output_filename,
		batch_count=batch_count,
		start_index=start_index,
		append_index=append_index,
		zero_pad=zero_pad,
		seed=seed,
		overwrite=overwrite,
		normalize_batch_energy_to_mask_power=normalize_batch_energy_to_mask_power,
	)

	if output.batch_count > 1 and not output.append_index:
		raise ValueError("output.append_index must be true when output.batch_count > 1")

	continuous: ContinuousConfig | None = None
	pixel: PixelConfig | None = None

	if emission.origin_type == "continuous":
		continuous_raw = _require_table(raw, "continuous")
		continuous = ContinuousConfig(
			n_rays=_parse_positive_int(_get_first(continuous_raw, ("n_rays", "nrays")), "continuous.n_rays")
		)

	if emission.origin_type == "pixelated":
		pixel_raw = _require_table(raw, "pixels")
		pitch = _parse_vec2(_get_first(pixel_raw, ("pitch", "pixel_pitch")), "pixels.pitch", positive_only=True)
		size = _parse_vec2(_get_first(pixel_raw, ("size", "pixel_size")) or list(pitch), "pixels.size", positive_only=True)
		projector_shift_max_pixels = _parse_vec2_nonnegative_float_or_scalar(
			_get_first(pixel_raw, ("projector_shift_max_pixels", "projection_shift_max_pixels", "plane_shift_max_pixels"))
			if _get_first(pixel_raw, ("projector_shift_max_pixels", "projection_shift_max_pixels", "plane_shift_max_pixels")) is not None
			else [0.0, 0.0],
			"pixels.projector_shift_max_pixels",
		)
		activation_mode = _parse_mode(
			_get_first(pixel_raw, ("activation_mode", "activation")) or "center",
			"pixels.activation_mode",
			allowed={"center", "antialiased"},
			aliases={"aa": "antialiased", "anti_aliased": "antialiased"},
		)
		intensity_mode = _parse_mode(
			_get_first(pixel_raw, ("intensity_mode", "intensity_profile")) or "flat",
			"pixels.intensity_mode",
			allowed={"flat", "gaussian"},
		)
		bins_raw = _get_first(pixel_raw, ("bins_per_pixel", "histogram_bins_per_pixel", "map_bins_per_pixel"))
		if bins_raw is None:
			bins_raw = [5, 5] if intensity_mode == "gaussian" else [1, 1]
		bins_per_pixel = _parse_vec2_positive_int_or_scalar(bins_raw, "pixels.bins_per_pixel")
		gaussian_size_type = str(_get_first(pixel_raw, ("gaussian_size_type", "size_type")) or "1/e2_diameter")

		pixel = PixelConfig(
			rays_per_pixel=_parse_positive_int(_get_first(pixel_raw, ("rays_per_pixel", "rpp")), "pixels.rays_per_pixel"),
			pitch=pitch,
			size=size,
			projector_shift_max_pixels=projector_shift_max_pixels,
			bins_per_pixel=bins_per_pixel,
			activation_mode=activation_mode,
			aa_samples=_parse_positive_int(_get_first(pixel_raw, ("aa_samples", "antialias_samples")) or 8, "pixels.aa_samples"),
			intensity_mode=intensity_mode,
			gaussian_size_type=gaussian_size_type,
		)

	cfg = RayGeneratorConfig(
		projection=projection,
		mask=mask,
		emission=emission,
		focus=focus,
		continuous=continuous,
		pixel=pixel,
		output=output,
	)

	_validate_cross_constraints(cfg)
	return cfg


def _validate_cross_constraints(cfg: RayGeneratorConfig) -> None:
	if cfg.focus.mode == "focused":
		direction = float(cfg.projection.direction)
		signed_distance = (cfg.projection.focus_plane_position - cfg.projection.source_plane_position) * direction
		if signed_distance <= 0.0:
			raise ValueError(
				"For focused rays, source_plane_position must lie 'behind' the focus plane "
				"along projection.direction."
			)

	if cfg.emission.origin_type == "pixelated" and cfg.pixel is None:
		raise ValueError("[pixels] table is required for emission.origin_type=pixelated")
	if cfg.emission.origin_type == "continuous" and cfg.continuous is None:
		raise ValueError("[continuous] table is required for emission.origin_type=continuous")


def _axis_indices(axis: str) -> tuple[int, int, int]:
	if axis == "x":
		return 0, 1, 2
	if axis == "y":
		return 1, 0, 2
	return 2, 0, 1


def _axis_unit(axis_index: int, direction: int) -> np.ndarray:
	axis = np.zeros(3, dtype=np.float64)
	axis[axis_index] = float(direction)
	return axis


def _local_uv_to_world(cfg: ProjectionConfig, u_values: np.ndarray, v_values: np.ndarray) -> np.ndarray:
	axis_i, u_i, v_i = _axis_indices(cfg.axis)
	world = np.zeros((u_values.shape[0], 3), dtype=np.float64)
	world[:, axis_i] = cfg.focus_plane_position
	world[:, u_i] = cfg.center[0] + u_values
	world[:, v_i] = cfg.center[1] + v_values
	return world


def _world_to_local_uv(cfg: ProjectionConfig, world_points: np.ndarray) -> np.ndarray:
	_, u_i, v_i = _axis_indices(cfg.axis)
	u = world_points[:, u_i] - cfg.center[0]
	v = world_points[:, v_i] - cfg.center[1]
	return np.ascontiguousarray(np.column_stack([u, v]))


def _local_uv_to_world_plane_coords(
	cfg: ProjectionConfig,
	u_values: np.ndarray,
	v_values: np.ndarray,
) -> np.ndarray:
	return np.ascontiguousarray(
		np.column_stack([u_values + cfg.center[0], v_values + cfg.center[1]])
	)


def _world_plane_coords_to_local_uv(
	cfg: ProjectionConfig,
	world_u_values: np.ndarray,
	world_v_values: np.ndarray,
) -> np.ndarray:
	return np.ascontiguousarray(
		np.column_stack([world_u_values - cfg.center[0], world_v_values - cfg.center[1]])
	)


def _mask_contains_world_plane(mask: _Mask, projection: ProjectionConfig, world_u: float, world_v: float) -> bool:
	local_u = world_u - projection.center[0]
	local_v = world_v - projection.center[1]
	return mask.contains(local_u, local_v)


def _sample_cone_directions(
	rng: np.random.Generator,
	axis_direction: np.ndarray,
	theta_max_rad: float,
	count: int,
) -> np.ndarray:
	axis = axis_direction / np.linalg.norm(axis_direction)
	reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
	tangent_1 = np.cross(axis, reference)
	tangent_1 /= np.linalg.norm(tangent_1)
	tangent_2 = np.cross(axis, tangent_1)
	tangent_2 /= np.linalg.norm(tangent_2)

	u1 = rng.random(count)
	u2 = rng.random(count)
	cos_theta = 1.0 - u1 * (1.0 - math.cos(theta_max_rad))
	sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
	phi = 2.0 * math.pi * u2

	dirs = (
		np.outer(sin_theta * np.cos(phi), tangent_1)
		+ np.outer(sin_theta * np.sin(phi), tangent_2)
		+ np.outer(cos_theta, axis)
	)
	dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
	return dirs


def _compute_origins_and_directions(
	cfg: RayGeneratorConfig,
	rng: np.random.Generator,
	focus_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	axis_i, _, _ = _axis_indices(cfg.projection.axis)
	axis_direction = _axis_unit(axis_i, cfg.projection.direction)
	n_rays = focus_points.shape[0]

	if cfg.focus.mode == "collimated":
		directions = np.repeat(axis_direction[None, :], n_rays, axis=0)
		origins = focus_points.copy()
		origins[:, axis_i] = cfg.projection.source_plane_position
		return origins, directions

	theta_max_rad = math.radians(cfg.focus.cone_half_angle_deg)
	directions = _sample_cone_directions(rng, axis_direction, theta_max_rad, n_rays)

	denominator = directions[:, axis_i]
	if np.any(np.abs(denominator) < 1e-12):
		raise RuntimeError("Focused direction sampling hit near-zero axis component.")

	t_values = (focus_points[:, axis_i] - cfg.projection.source_plane_position) / denominator
	if np.any(t_values <= 0.0):
		raise RuntimeError(
			"Focused rays produced non-positive back-projection distance; "
			"check projection.direction and plane positions."
		)

	origins = focus_points - directions * t_values[:, None]
	return origins, directions


def _gaussian_sigma(reported_size: float, size_type: str) -> float:
	st = size_type.strip()
	factors = {
		"FWHM": 2.355,
		"sigma": 1.0,
		"1/e2_diameter": 3.398,
	}
	if st not in factors:
		raise ValueError("pixels.gaussian_size_type must be one of: FWHM, sigma, 1/e2_diameter")
	return reported_size / factors[st]


class _Mask:
	def area(self) -> float:
		raise NotImplementedError

	def contains(self, u: float, v: float) -> bool:
		raise NotImplementedError

	def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
		raise NotImplementedError


class _RectangleMask(_Mask):
	def __init__(self, width: float, height: float, offset: Vec2f) -> None:
		self.width = width
		self.height = height
		self.offset_u = offset[0]
		self.offset_v = offset[1]

	def area(self) -> float:
		return self.width * self.height

	def contains(self, u: float, v: float) -> bool:
		du = u - self.offset_u
		dv = v - self.offset_v
		return (abs(du) <= self.width / 2.0) and (abs(dv) <= self.height / 2.0)

	def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
		u = rng.uniform(-self.width / 2.0, self.width / 2.0, n) + self.offset_u
		v = rng.uniform(-self.height / 2.0, self.height / 2.0, n) + self.offset_v
		return u, v


class _CircleMask(_Mask):
	def __init__(self, radius: float, offset: Vec2f) -> None:
		self.radius = radius
		self.offset_u = offset[0]
		self.offset_v = offset[1]

	def area(self) -> float:
		return math.pi * self.radius * self.radius

	def contains(self, u: float, v: float) -> bool:
		du = u - self.offset_u
		dv = v - self.offset_v
		return du * du + dv * dv <= self.radius * self.radius

	def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
		r = self.radius * np.sqrt(rng.random(n))
		angle = 2.0 * math.pi * rng.random(n)
		u = self.offset_u + r * np.cos(angle)
		v = self.offset_v + r * np.sin(angle)
		return u, v


class _AnnulusMask(_Mask):
	def __init__(self, outer_radius: float, inner_radius: float, offset: Vec2f) -> None:
		if inner_radius >= outer_radius:
			raise ValueError("mask.inner_radius must be < mask.outer_radius")
		self.outer_radius = outer_radius
		self.inner_radius = inner_radius
		self.offset_u = offset[0]
		self.offset_v = offset[1]

	def area(self) -> float:
		return math.pi * (self.outer_radius * self.outer_radius - self.inner_radius * self.inner_radius)

	def contains(self, u: float, v: float) -> bool:
		du = u - self.offset_u
		dv = v - self.offset_v
		r2 = du * du + dv * dv
		return self.inner_radius * self.inner_radius <= r2 <= self.outer_radius * self.outer_radius

	def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
		r2_inner = self.inner_radius * self.inner_radius
		r2_outer = self.outer_radius * self.outer_radius
		r = np.sqrt(rng.random(n) * (r2_outer - r2_inner) + r2_inner)
		angle = 2.0 * math.pi * rng.random(n)
		u = self.offset_u + r * np.cos(angle)
		v = self.offset_v + r * np.sin(angle)
		return u, v


class _TriangleMask(_Mask):
	def __init__(self, base: float, height: float, offset: Vec2f) -> None:
		self.base = base
		self.height = height
		self.offset_u = offset[0]
		self.offset_v = offset[1]

	def area(self) -> float:
		return 0.5 * self.base * self.height

	def contains(self, u: float, v: float) -> bool:
		du = u - self.offset_u
		dv = v - self.offset_v
		half_height = self.height / 2.0
		if dv < -half_height or dv > half_height:
			return False
		frac = (dv + half_height) / self.height
		half_width = (self.base / 2.0) * (1.0 - frac)
		return abs(du) <= half_width

	def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
		u = rng.random(n)
		v = rng.random(n)
		flip = (u + v) > 1.0
		u[flip] = 1.0 - u[flip]
		v[flip] = 1.0 - v[flip]

		x0, y0 = -self.base / 2.0, -self.height / 2.0
		x1, y1 = self.base / 2.0, -self.height / 2.0
		x2, y2 = 0.0, self.height / 2.0
		su = x0 + u * (x1 - x0) + v * (x2 - x0)
		sv = y0 + u * (y1 - y0) + v * (y2 - y0)
		return su + self.offset_u, sv + self.offset_v


def _build_mask(cfg: RayGeneratorConfig) -> _Mask:
	m = cfg.mask
	if m.type == "rectangle":
		width = cfg.projection.width if m.rectangle_width is None else m.rectangle_width
		height = cfg.projection.height if m.rectangle_height is None else m.rectangle_height
		return _RectangleMask(width, height, m.offset)
	if m.type == "circle":
		if m.circle_radius is None:
			raise ValueError("mask.radius is required for mask.type=circle")
		return _CircleMask(m.circle_radius, m.offset)
	if m.type == "triangle":
		if m.triangle_base is None or m.triangle_height is None:
			raise ValueError("mask.base and mask.height are required for mask.type=triangle")
		return _TriangleMask(m.triangle_base, m.triangle_height, m.offset)
	if m.type == "annulus":
		if m.annulus_outer_radius is None or m.annulus_inner_radius is None:
			raise ValueError("mask.outer_radius and mask.inner_radius are required for mask.type=annulus")
		return _AnnulusMask(m.annulus_outer_radius, m.annulus_inner_radius, m.offset)
	raise ValueError(f"Unsupported mask type: {m.type}")


def _aa_activation_fraction(
	mask: _Mask,
	center_u: float,
	center_v: float,
	pixel_size: Vec2f,
	aa_samples: int,
) -> float:
	su = pixel_size[0]
	sv = pixel_size[1]
	du = np.linspace(-su / 2.0, su / 2.0, aa_samples, endpoint=False) + su / (2.0 * aa_samples)
	dv = np.linspace(-sv / 2.0, sv / 2.0, aa_samples, endpoint=False) + sv / (2.0 * aa_samples)
	uu, vv = np.meshgrid(center_u + du, center_v + dv)
	flat_u = uu.ravel()
	flat_v = vv.ravel()
	inside = np.fromiter((mask.contains(float(u), float(v)) for u, v in zip(flat_u, flat_v)), dtype=np.float64)
	return float(np.mean(inside))


def _aa_activation_fraction_world(
	mask: _Mask,
	projection: ProjectionConfig,
	center_world_u: float,
	center_world_v: float,
	pixel_size: Vec2f,
	aa_samples: int,
) -> float:
	su = pixel_size[0]
	sv = pixel_size[1]
	du = np.linspace(-su / 2.0, su / 2.0, aa_samples, endpoint=False) + su / (2.0 * aa_samples)
	dv = np.linspace(-sv / 2.0, sv / 2.0, aa_samples, endpoint=False) + sv / (2.0 * aa_samples)
	uu, vv = np.meshgrid(center_world_u + du, center_world_v + dv)
	flat_u = uu.ravel()
	flat_v = vv.ravel()
	inside = np.fromiter(
		(_mask_contains_world_plane(mask, projection, float(u), float(v)) for u, v in zip(flat_u, flat_v)),
		dtype=np.float64,
	)
	return float(np.mean(inside))


def _build_pixel_centers(extent: float, pitch: float) -> np.ndarray:
	start = -extent / 2.0 + pitch / 2.0
	end = extent / 2.0 - pitch / 2.0
	if start > end:
		raise ValueError("Pixel pitch is too large relative to projection extent.")
	return np.arange(start, end + 1e-12, pitch)


def _generate_continuous_rays(
	cfg: RayGeneratorConfig,
	mask: _Mask,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if cfg.continuous is None:
		raise RuntimeError("Continuous config is missing.")

	n_rays = cfg.continuous.n_rays
	sampled_u, sampled_v = mask.sample(rng, n_rays)
	focus_points = _local_uv_to_world(cfg.projection, sampled_u, sampled_v)
	origins, directions = _compute_origins_and_directions(cfg, rng, focus_points)

	total_power = cfg.emission.base_intensity * mask.area()
	ray_power = total_power / float(n_rays)
	energies = np.full(n_rays, ray_power, dtype=np.float64)
	focus_uv = np.ascontiguousarray(np.column_stack([sampled_u, sampled_v]))
	return origins, directions, energies, focus_uv


def _generate_pixelated_rays(
	cfg: RayGeneratorConfig,
	mask: _Mask,
	rng: np.random.Generator,
	projector_shift_uv: Vec2f = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if cfg.pixel is None:
		raise RuntimeError("Pixel config is missing.")

	px = cfg.pixel
	centers_u = _build_pixel_centers(cfg.projection.width, px.pitch[0]) + projector_shift_uv[0]
	centers_v = _build_pixel_centers(cfg.projection.height, px.pitch[1]) + projector_shift_uv[1]
	centers_u_world = centers_u + cfg.projection.center[0]
	centers_v_world = centers_v + cfg.projection.center[1]

	active_pixels: list[tuple[float, float, float]] = []
	for cu, cu_world in zip(centers_u, centers_u_world):
		for cv, cv_world in zip(centers_v, centers_v_world):
			# Pixel centers are jittered in local UV; mask inclusion is evaluated in
			# world-plane coordinates so the mask remains fixed in real space.
			if px.activation_mode == "center":
				fraction = 1.0 if _mask_contains_world_plane(mask, cfg.projection, float(cu_world), float(cv_world)) else 0.0
			else:
				fraction = _aa_activation_fraction_world(
					mask,
					cfg.projection,
					float(cu_world),
					float(cv_world),
					px.size,
					px.aa_samples,
				)

			if fraction <= 0.0:
				continue

			i_active = cfg.emission.base_intensity * fraction
			if i_active <= 0.0:
				continue
			active_pixels.append((float(cu), float(cv), i_active))

	if not active_pixels:
		raise RuntimeError("No active pixels found. Check mask and pixel settings.")

	rays_per_pixel = px.rays_per_pixel
	total_rays = len(active_pixels) * rays_per_pixel
	origins = np.zeros((total_rays, 3), dtype=np.float64)
	directions = np.zeros((total_rays, 3), dtype=np.float64)
	energies = np.zeros(total_rays, dtype=np.float64)
	focus_uv = np.zeros((total_rays, 2), dtype=np.float64)

	pixel_area = px.size[0] * px.size[1]
	sigma_u = _gaussian_sigma(px.size[0], px.gaussian_size_type)
	sigma_v = _gaussian_sigma(px.size[1], px.gaussian_size_type)

	cursor = 0
	for center_u, center_v, i_active in active_pixels:
		local_u = rng.uniform(-px.size[0] / 2.0, px.size[0] / 2.0, rays_per_pixel)
		local_v = rng.uniform(-px.size[1] / 2.0, px.size[1] / 2.0, rays_per_pixel)
		sample_u = center_u + local_u
		sample_v = center_v + local_v

		focus_points = _local_uv_to_world(cfg.projection, sample_u, sample_v)
		pix_origins, pix_directions = _compute_origins_and_directions(cfg, rng, focus_points)

		if px.intensity_mode == "flat":
			pix_energies = np.full(rays_per_pixel, i_active * pixel_area / rays_per_pixel, dtype=np.float64)
		else:
			gaussian_intensity = i_active * np.exp(
				-(local_u * local_u / (2.0 * sigma_u * sigma_u) + local_v * local_v / (2.0 * sigma_v * sigma_v))
			)
			pix_energies = gaussian_intensity * (pixel_area / rays_per_pixel)

		next_cursor = cursor + rays_per_pixel
		origins[cursor:next_cursor] = pix_origins
		directions[cursor:next_cursor] = pix_directions
		energies[cursor:next_cursor] = pix_energies
		focus_uv[cursor:next_cursor, 0] = sample_u
		focus_uv[cursor:next_cursor, 1] = sample_v
		cursor = next_cursor

	if cfg.output.normalize_batch_energy_to_mask_power:
		target = cfg.emission.base_intensity * mask.area()
		total = float(np.sum(energies))
		if total > 0.0:
			energies *= target / total

	return origins, directions, energies, focus_uv


def _output_path_for_batch(output_cfg: OutputConfig, batch_idx: int) -> Path:
	base = output_cfg.output_folder / output_cfg.output_filename
	suffix = base.suffix if base.suffix else ".ry"

	if output_cfg.batch_count == 1:
		return base.with_suffix(suffix)

	file_id = output_cfg.start_index + batch_idx
	if output_cfg.append_index:
		if output_cfg.zero_pad > 0:
			id_text = f"{file_id:0{output_cfg.zero_pad}d}"
		else:
			id_text = str(file_id)
		return base.with_name(f"{base.stem}_{id_text}{suffix}")

	return base.with_suffix(suffix)


def _write_ry(path: Path, origins: np.ndarray, directions: np.ndarray, energies: np.ndarray) -> None:
	with path.open("w", encoding="utf-8") as f:
		f.write("# ray file v1\n")
		f.write("# format:\n")
		f.write("# nrays\n")
		f.write("# ox oy oz dx dy dz energy\n")
		f.write(f"{origins.shape[0]}\n")
		for origin, direction, energy in zip(origins, directions, energies):
			f.write(
				f"{origin[0]:.6e} {origin[1]:.6e} {origin[2]:.6e} "
				f"{direction[0]:.6e} {direction[1]:.6e} {direction[2]:.6e} {energy:.6e}\n"
			)


def _auto_bin_size_continuous(cfg: RayGeneratorConfig, mask: _Mask, n_rays: int) -> tuple[float, float]:
	ray_density = n_rays / max(mask.area(), 1e-12)
	target_rays_per_bin = 1000.0
	bin_area = target_rays_per_bin / max(ray_density, 1e-12)
	side = math.sqrt(max(bin_area, 1e-12))
	min_side = min(cfg.projection.width, cfg.projection.height) / 500.0
	max_side = max(cfg.projection.width, cfg.projection.height) / 20.0
	clamped = max(min_side, min(max_side, side))
	return clamped, clamped


def _map_bin_size(cfg: RayGeneratorConfig, mask: _Mask, n_rays: int) -> tuple[float, float]:
	if cfg.emission.origin_type == "continuous":
		return _auto_bin_size_continuous(cfg, mask, n_rays)

	if cfg.pixel is None:
		raise RuntimeError("Pixel config missing for pixelated map binning.")

	return (
		cfg.pixel.pitch[0] / float(cfg.pixel.bins_per_pixel[0]),
		cfg.pixel.pitch[1] / float(cfg.pixel.bins_per_pixel[1]),
	)


def _focused_max_spread(cfg: RayGeneratorConfig) -> float:
	distance = abs(cfg.projection.focus_plane_position - cfg.projection.source_plane_position)
	theta = math.radians(cfg.focus.cone_half_angle_deg)
	return distance * math.tan(theta)


def _build_edges(values: np.ndarray, bin_size: float) -> np.ndarray:
	if bin_size <= 0.0:
		raise ValueError("Bin size must be > 0")

	vmin = float(np.min(values))
	vmax = float(np.max(values))
	if math.isclose(vmin, vmax):
		vmin -= 0.5 * bin_size
		vmax += 0.5 * bin_size

	start = vmin - 0.5 * bin_size
	end = vmax + 0.5 * bin_size
	n_bins = max(1, int(math.ceil((end - start) / bin_size)))
	return np.linspace(start, start + n_bins * bin_size, n_bins + 1)


def _build_edges_aligned(values: np.ndarray, bin_size: float, anchor: float) -> np.ndarray:
	if bin_size <= 0.0:
		raise ValueError("Bin size must be > 0")

	vmin = float(np.min(values))
	vmax = float(np.max(values))
	if math.isclose(vmin, vmax):
		vmin -= 0.5 * bin_size
		vmax += 0.5 * bin_size

	first_bin = math.floor((vmin - anchor) / bin_size)
	last_bin = math.ceil((vmax - anchor) / bin_size)
	if last_bin <= first_bin:
		last_bin = first_bin + 1

	start = anchor + first_bin * bin_size
	n_bins = int(last_bin - first_bin)
	return start + np.arange(n_bins + 1, dtype=np.float64) * bin_size


def _pixel_bin_anchors(cfg: RayGeneratorConfig) -> tuple[float, float]:
	if cfg.pixel is None:
		raise RuntimeError("Pixel config missing for pixel-aligned histogram bins.")

	centers_u = _build_pixel_centers(cfg.projection.width, cfg.pixel.pitch[0])
	centers_v = _build_pixel_centers(cfg.projection.height, cfg.pixel.pitch[1])

	anchor_u = float(centers_u[0] - 0.5 * cfg.pixel.pitch[0])
	anchor_v = float(centers_v[0] - 0.5 * cfg.pixel.pitch[1])
	return anchor_u, anchor_v


def _sample_projector_plane_shift(cfg: RayGeneratorConfig, rng: np.random.Generator) -> Vec2f:
	if cfg.emission.origin_type != "pixelated" or cfg.pixel is None:
		return 0.0, 0.0

	max_shift_u_pixels = cfg.pixel.projector_shift_max_pixels[0]
	max_shift_v_pixels = cfg.pixel.projector_shift_max_pixels[1]

	if max_shift_u_pixels <= 0.0 and max_shift_v_pixels <= 0.0:
		return 0.0, 0.0

	shift_u = float(rng.uniform(-max_shift_u_pixels, max_shift_u_pixels) * cfg.pixel.pitch[0])
	shift_v = float(rng.uniform(-max_shift_v_pixels, max_shift_v_pixels) * cfg.pixel.pitch[1])
	return shift_u, shift_v


def _projection_plane_limits(cfg: ProjectionConfig) -> tuple[float, float, float, float]:
	return (-0.5 * cfg.width, 0.5 * cfg.width, -0.5 * cfg.height, 0.5 * cfg.height)


def _build_edges_for_limits(
	min_value: float,
	max_value: float,
	bin_size: float,
	anchor: float | None = None,
) -> np.ndarray:
	if bin_size <= 0.0:
		raise ValueError("Bin size must be > 0")

	if max_value <= min_value:
		raise ValueError("Invalid limits for histogram edges.")

	if anchor is None:
		n_bins = max(1, int(math.ceil((max_value - min_value) / bin_size)))
		return min_value + np.arange(n_bins + 1, dtype=np.float64) * bin_size

	first_bin = math.floor((min_value - anchor) / bin_size)
	last_bin = math.ceil((max_value - anchor) / bin_size)
	if last_bin <= first_bin:
		last_bin = first_bin + 1

	start = anchor + first_bin * bin_size
	n_bins = int(last_bin - first_bin)
	return start + np.arange(n_bins + 1, dtype=np.float64) * bin_size


def _intensity_map_from_edges(
	points_uv: np.ndarray,
	energies: np.ndarray,
	edges_u: np.ndarray,
	edges_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	hist, u_edges, v_edges = np.histogram2d(
		points_uv[:, 0],
		points_uv[:, 1],
		bins=[edges_u, edges_v],
		weights=energies,
	)

	du = np.diff(u_edges)[:, None]
	dv = np.diff(v_edges)[None, :]
	intensity = (hist / np.maximum(du * dv, 1e-30)).T
	return intensity, u_edges, v_edges


def _save_intensity_png(
	points_uv: np.ndarray,
	energies: np.ndarray,
	bin_size_u: float,
	bin_size_v: float,
	out_png: Path,
	title: str,
	axis_limits: tuple[float, float, float, float] | None = None,
	edges_u: np.ndarray | None = None,
	edges_v: np.ndarray | None = None,
	vmin: float | None = None,
	vmax: float | None = None,
) -> None:
	if points_uv.shape[0] == 0:
		raise RuntimeError("Cannot save intensity map for empty point set.")

	if edges_u is None:
		edges_u = _build_edges(points_uv[:, 0], bin_size_u)
	if edges_v is None:
		edges_v = _build_edges(points_uv[:, 1], bin_size_v)

	intensity, u_edges, v_edges = _intensity_map_from_edges(points_uv, energies, edges_u, edges_v)

	fig, ax = plt.subplots(figsize=(7, 5))
	im = ax.imshow(
		intensity,
		extent=[u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]],
		origin="lower",
		cmap="inferno",
		interpolation="nearest",
		vmin=vmin,
		vmax=vmax,
	)
	ax.set_xlabel("u")
	ax.set_ylabel("v")
	if axis_limits is not None:
		u_min, u_max, v_min, v_max = axis_limits
		ax.set_xlim(u_min, u_max)
		ax.set_ylim(v_min, v_max)
	ax.set_title(title)
	fig.colorbar(im, ax=ax, label="Intensity [W/m^2]")
	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _save_intensity_map_png(
	intensity: np.ndarray,
	edges_u: np.ndarray,
	edges_v: np.ndarray,
	out_png: Path,
	title: str,
	axis_limits: tuple[float, float, float, float] | None = None,
) -> None:
	fig, ax = plt.subplots(figsize=(7, 5))
	im = ax.imshow(
		intensity,
		extent=[edges_u[0], edges_u[-1], edges_v[0], edges_v[-1]],
		origin="lower",
		cmap="inferno",
		interpolation="nearest",
	)
	ax.set_xlabel("u")
	ax.set_ylabel("v")
	if axis_limits is not None:
		u_min, u_max, v_min, v_max = axis_limits
		ax.set_xlim(u_min, u_max)
		ax.set_ylim(v_min, v_max)
	ax.set_title(title)
	fig.colorbar(im, ax=ax, label="Intensity [W/m^2]")
	fig.tight_layout()
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _save_average_with_single_map_png(
	single_intensity: np.ndarray,
	average_intensity: np.ndarray,
	edges_u: np.ndarray,
	edges_v: np.ndarray,
	out_png: Path,
	single_title: str,
	average_title: str,
	axis_limits: tuple[float, float, float, float] | None = None,
) -> None:
	if single_intensity.shape != average_intensity.shape:
		raise RuntimeError("Single-sample and average map shapes must match")

	vmin = float(min(np.min(single_intensity), np.min(average_intensity)))
	vmax = float(max(np.max(single_intensity), np.max(average_intensity)))
	if math.isclose(vmin, vmax):
		pad = max(1e-12, abs(vmin) * 1e-6)
		vmin -= pad
		vmax += pad

	extent = [edges_u[0], edges_u[-1], edges_v[0], edges_v[-1]]
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
	im = axes[0].imshow(
		single_intensity,
		extent=extent,
		origin="lower",
		cmap="inferno",
		interpolation="nearest",
		vmin=vmin,
		vmax=vmax,
	)
	axes[0].set_title(single_title)
	axes[0].set_xlabel("u")
	axes[0].set_ylabel("v")

	axes[1].imshow(
		average_intensity,
		extent=extent,
		origin="lower",
		cmap="inferno",
		interpolation="nearest",
		vmin=vmin,
		vmax=vmax,
	)
	axes[1].set_title(average_title)
	axes[1].set_xlabel("u")

	if axis_limits is not None:
		u_min, u_max, v_min, v_max = axis_limits
		axes[0].set_xlim(u_min, u_max)
		axes[0].set_ylim(v_min, v_max)

	fig.colorbar(im, ax=axes, label="Intensity [W/m^2]")
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def _save_focused_comparison_png(
	projection_uv: np.ndarray,
	focus_uv: np.ndarray,
	energies: np.ndarray,
	bin_size_u: float,
	bin_size_v: float,
	out_png: Path,
	axis_limits: tuple[float, float, float, float] | None = None,
	edges_u: np.ndarray | None = None,
	edges_v: np.ndarray | None = None,
) -> None:
	if projection_uv.shape[0] == 0 or focus_uv.shape[0] == 0:
		raise RuntimeError("Cannot save focused comparison map for empty point set.")

	if edges_u is None or edges_v is None:
		shared_uv = np.vstack((projection_uv, focus_uv))
		if edges_u is None:
			edges_u = _build_edges(shared_uv[:, 0], bin_size_u)
		if edges_v is None:
			edges_v = _build_edges(shared_uv[:, 1], bin_size_v)

	projection_intensity, u_edges, v_edges = _intensity_map_from_edges(
		projection_uv,
		energies,
		edges_u,
		edges_v,
	)
	focus_intensity, _, _ = _intensity_map_from_edges(
		focus_uv,
		energies,
		edges_u,
		edges_v,
	)

	vmin = float(min(np.min(projection_intensity), np.min(focus_intensity)))
	vmax = float(max(np.max(projection_intensity), np.max(focus_intensity)))
	if math.isclose(vmin, vmax):
		pad = max(1e-12, abs(vmin) * 1e-6)
		vmin -= pad
		vmax += pad

	extent = [u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]]
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
	im = axes[0].imshow(
		projection_intensity,
		extent=extent,
		origin="lower",
		cmap="inferno",
		interpolation="nearest",
		vmin=vmin,
		vmax=vmax,
	)
	axes[0].set_title("Projection plane")
	axes[0].set_xlabel("u")
	axes[0].set_ylabel("v")

	axes[1].imshow(
		focus_intensity,
		extent=extent,
		origin="lower",
		cmap="inferno",
		interpolation="nearest",
		vmin=vmin,
		vmax=vmax,
	)
	axes[1].set_title("Focus plane")
	axes[1].set_xlabel("u")

	if axis_limits is not None:
		u_min, u_max, v_min, v_max = axis_limits
		axes[0].set_xlim(u_min, u_max)
		axes[0].set_ylim(v_min, v_max)

	fig.colorbar(im, ax=axes, label="Intensity [W/m^2]")
	fig.savefig(out_png, dpi=200)
	plt.close(fig)


def run(config_path: Path) -> None:
	cfg = load_config(config_path)
	mask = _build_mask(cfg)

	cfg.output.output_folder.mkdir(parents=True, exist_ok=True)

	print("Configured projection ray generation")
	print(f"Config: {config_path}")
	print(f"Emission mode: {cfg.emission.origin_type}")
	print(f"Focus mode: {cfg.focus.mode}")
	print(f"Mask: {cfg.mask.type}")
	print(f"Mask area: {mask.area():.6e}")
	print(f"Base intensity: {cfg.emission.base_intensity:.6e}")
	print(f"Batch count: {cfg.output.batch_count}")
	if cfg.emission.origin_type == "pixelated" and cfg.pixel is not None:
		print(
			"Projector random shift max [pixels]: "
			f"({cfg.pixel.projector_shift_max_pixels[0]:.3f}, {cfg.pixel.projector_shift_max_pixels[1]:.3f})"
		)

	avg_projection_intensity: np.ndarray | None = None
	avg_projection_edges_u: np.ndarray | None = None
	avg_projection_edges_v: np.ndarray | None = None
	single_projection_intensity: np.ndarray | None = None

	avg_focus_intensity: np.ndarray | None = None
	avg_focus_edges_u: np.ndarray | None = None
	avg_focus_edges_v: np.ndarray | None = None
	single_focus_intensity: np.ndarray | None = None

	avg_plane_intensity_world: np.ndarray | None = None
	avg_edges_u_world: np.ndarray | None = None
	avg_edges_v_world: np.ndarray | None = None
	avg_count = 0

	for batch_idx in range(cfg.output.batch_count):
		seed = None if cfg.output.seed is None else cfg.output.seed + batch_idx
		rng = np.random.default_rng(seed)
		projector_shift_uv = (0.0, 0.0)

		if cfg.emission.origin_type == "continuous":
			origins, directions, energies, focus_uv = _generate_continuous_rays(cfg, mask, rng)
		else:
			projector_shift_uv = _sample_projector_plane_shift(cfg, rng)
			origins, directions, energies, focus_uv = _generate_pixelated_rays(
				cfg,
				mask,
				rng,
				projector_shift_uv=projector_shift_uv,
			)

		out_path = _output_path_for_batch(cfg.output, batch_idx)
		if out_path.exists() and not cfg.output.overwrite:
			raise FileExistsError(
				f"Output file already exists and overwrite=false: {out_path}"
			)

		_write_ry(out_path, origins, directions, energies)

		projection_uv = _world_to_local_uv(cfg.projection, origins)
		bin_u, bin_v = _map_bin_size(cfg, mask, origins.shape[0])
		plane_u_min, plane_u_max, plane_v_min, plane_v_max = _projection_plane_limits(cfg.projection)
		axis_limits = (plane_u_min, plane_u_max, plane_v_min, plane_v_max)
		aligned_anchor_u = None
		aligned_anchor_v = None
		if cfg.emission.origin_type == "pixelated":
			aligned_anchor_u, aligned_anchor_v = _pixel_bin_anchors(cfg)

		shared_edges_u = None
		shared_edges_v = None
		if cfg.focus.mode == "focused":
			max_spread = _focused_max_spread(cfg)
			min_bin = min(bin_u, bin_v)
			if max_spread < 0.5 * min_bin:
				print(
					"Warning: focused spread in projection plane is smaller than map bin size; "
					"projection/focus PNGs may look identical. "
					"Increase focus.cone_half_angle_deg or use smaller pixel/bin size."
				)
			if aligned_anchor_u is not None and aligned_anchor_v is not None:
				shared_edges_u = _build_edges_for_limits(plane_u_min, plane_u_max, bin_u, aligned_anchor_u)
				shared_edges_v = _build_edges_for_limits(plane_v_min, plane_v_max, bin_v, aligned_anchor_v)
			else:
				shared_edges_u = _build_edges_for_limits(plane_u_min, plane_u_max, bin_u)
				shared_edges_v = _build_edges_for_limits(plane_v_min, plane_v_max, bin_v)

		projection_edges_u = None
		projection_edges_v = None
		if aligned_anchor_u is not None and aligned_anchor_v is not None:
			projection_edges_u = _build_edges_for_limits(plane_u_min, plane_u_max, bin_u, aligned_anchor_u)
			projection_edges_v = _build_edges_for_limits(plane_v_min, plane_v_max, bin_v, aligned_anchor_v)
		else:
			projection_edges_u = _build_edges_for_limits(plane_u_min, plane_u_max, bin_u)
			projection_edges_v = _build_edges_for_limits(plane_v_min, plane_v_max, bin_v)

		projection_png = out_path.with_name(f"{out_path.stem}_projection.png")
		comparison_png = out_path.with_name(f"{out_path.stem}_projection_focus.png")

		projection_intensity, map_edges_u, map_edges_v = _intensity_map_from_edges(
			projection_uv,
			energies,
			projection_edges_u,
			projection_edges_v,
		)

		focus_intensity: np.ndarray | None = None
		avg_batch_intensity = projection_intensity
		avg_batch_edges_u = map_edges_u
		avg_batch_edges_v = map_edges_v
		avg_batch_points_world = _local_uv_to_world_plane_coords(
			cfg.projection,
			projection_uv[:, 0],
			projection_uv[:, 1],
		)
		avg_batch_edges_u_world = avg_batch_edges_u + cfg.projection.center[0]
		avg_batch_edges_v_world = avg_batch_edges_v + cfg.projection.center[1]
		if cfg.focus.mode == "focused":
			if shared_edges_u is None or shared_edges_v is None:
				raise RuntimeError("Focused averaging requires shared map edges")
			focus_intensity, avg_batch_edges_u, avg_batch_edges_v = _intensity_map_from_edges(
				focus_uv,
				energies,
				shared_edges_u,
				shared_edges_v,
			)
			avg_batch_intensity = focus_intensity
			avg_batch_points_world = _local_uv_to_world_plane_coords(
				cfg.projection,
				focus_uv[:, 0],
				focus_uv[:, 1],
			)
			avg_batch_edges_u_world = avg_batch_edges_u + cfg.projection.center[0]
			avg_batch_edges_v_world = avg_batch_edges_v + cfg.projection.center[1]

		avg_batch_intensity_world, map_edges_u_world, map_edges_v_world = _intensity_map_from_edges(
			avg_batch_points_world,
			energies,
			avg_batch_edges_u_world,
			avg_batch_edges_v_world,
		)

		if avg_projection_intensity is None:
			avg_projection_intensity = np.zeros_like(projection_intensity)
			avg_projection_edges_u = map_edges_u.copy()
			avg_projection_edges_v = map_edges_v.copy()
			single_projection_intensity = projection_intensity.copy()
		else:
			if avg_projection_edges_u is None or avg_projection_edges_v is None:
				raise RuntimeError("Projection average intensity accumulator is in invalid state")
			if projection_intensity.shape != avg_projection_intensity.shape:
				raise RuntimeError("Projection map shape changed across batches; cannot compute average map")
			if not np.allclose(map_edges_u, avg_projection_edges_u) or not np.allclose(map_edges_v, avg_projection_edges_v):
				raise RuntimeError("Projection map bins changed across batches; cannot compute average map")
		avg_projection_intensity += projection_intensity

		if cfg.focus.mode == "focused":
			if focus_intensity is None:
				raise RuntimeError("Focused averaging map is missing")
			if avg_focus_intensity is None:
				avg_focus_intensity = np.zeros_like(focus_intensity)
				avg_focus_edges_u = avg_batch_edges_u.copy()
				avg_focus_edges_v = avg_batch_edges_v.copy()
				single_focus_intensity = focus_intensity.copy()
			else:
				if avg_focus_edges_u is None or avg_focus_edges_v is None:
					raise RuntimeError("Focus average intensity accumulator is in invalid state")
				if focus_intensity.shape != avg_focus_intensity.shape:
					raise RuntimeError("Focus map shape changed across batches; cannot compute average map")
				if not np.allclose(avg_batch_edges_u, avg_focus_edges_u) or not np.allclose(avg_batch_edges_v, avg_focus_edges_v):
					raise RuntimeError("Focus map bins changed across batches; cannot compute average map")
			avg_focus_intensity += focus_intensity

		if avg_plane_intensity_world is None:
			avg_plane_intensity_world = np.zeros_like(avg_batch_intensity_world)
			avg_edges_u_world = map_edges_u_world.copy()
			avg_edges_v_world = map_edges_v_world.copy()
		else:
			if avg_plane_intensity_world is None or avg_edges_u_world is None or avg_edges_v_world is None:
				raise RuntimeError("World-coordinate average intensity accumulator is in invalid state")
			if avg_batch_intensity_world.shape != avg_plane_intensity_world.shape:
				raise RuntimeError("World-coordinate map shape changed across batches; cannot compute average map")
			if not np.allclose(map_edges_u_world, avg_edges_u_world) or not np.allclose(map_edges_v_world, avg_edges_v_world):
				raise RuntimeError("World-coordinate map bins changed across batches; cannot compute average map")
		avg_plane_intensity_world += avg_batch_intensity_world
		avg_count += 1

		if cfg.focus.mode == "focused":
			_save_focused_comparison_png(
				projection_uv,
				focus_uv,
				energies,
				bin_u,
				bin_v,
				comparison_png,
				axis_limits,
				shared_edges_u,
				shared_edges_v,
			)
			print(f"Saved: {comparison_png}")
		else:
			_save_intensity_map_png(
				projection_intensity,
				map_edges_u,
				map_edges_v,
				projection_png,
				"Intensity map in projection plane",
				axis_limits=axis_limits,
			)
			print(f"Saved: {projection_png}")

		if cfg.emission.origin_type == "pixelated":
			print(
				f"batch {batch_idx}: projector shift [u, v] = "
				f"({projector_shift_uv[0]:.6f}, {projector_shift_uv[1]:.6f})"
			)

		print(
			f"Wrote {out_path} | nrays={origins.shape[0]} | "
			f"energy sum={energies.sum():.6e}"
		)

	if avg_projection_intensity is not None and avg_count > 0:
		if avg_projection_edges_u is None or avg_projection_edges_v is None or single_projection_intensity is None:
			raise RuntimeError("Projection average post-processing failed due to missing map state")
		if avg_plane_intensity_world is None or avg_edges_u_world is None or avg_edges_v_world is None:
			raise RuntimeError("World-coordinate average post-processing failed due to missing bin edges")

		single_projection_for_png = single_projection_intensity
		single_focus_for_png = single_focus_intensity
		single_projection_title = "Single sample projection intensity (batch 0)"
		single_focus_title = "Single sample focus intensity (batch 0)"
		if cfg.emission.origin_type == "pixelated":
			centered_seed = cfg.output.seed
			centered_rng = np.random.default_rng(centered_seed)
			centered_origins, _, centered_energies, centered_focus_uv = _generate_pixelated_rays(
				cfg,
				mask,
				centered_rng,
				projector_shift_uv=(0.0, 0.0),
			)
			centered_projection_uv = _world_to_local_uv(cfg.projection, centered_origins)
			single_projection_for_png, _, _ = _intensity_map_from_edges(
				centered_projection_uv,
				centered_energies,
				avg_projection_edges_u,
				avg_projection_edges_v,
			)
			single_projection_title = "Centered projection intensity (no shift)"
			if cfg.focus.mode == "focused":
				if avg_focus_edges_u is None or avg_focus_edges_v is None:
					raise RuntimeError("Focus average post-processing failed due to missing bin edges")
				single_focus_for_png, _, _ = _intensity_map_from_edges(
					centered_focus_uv,
					centered_energies,
					avg_focus_edges_u,
					avg_focus_edges_v,
				)
				single_focus_title = "Centered focus intensity (no shift)"

		avg_projection_intensity /= float(avg_count)
		if avg_focus_intensity is not None:
			avg_focus_intensity /= float(avg_count)
		avg_plane_intensity_world /= float(avg_count)
		base_stem = Path(cfg.output.output_filename).stem
		projection_average_base = cfg.output.output_folder / f"{base_stem}_projection_average"
		projection_average_png = projection_average_base.with_suffix(".png")
		projection_average_npz = projection_average_base.with_suffix(".npz")
		focus_average_png: Path | None = None
		focus_average_npz: Path | None = None
		if cfg.focus.mode == "focused":
			if avg_focus_intensity is None or avg_focus_edges_u is None or avg_focus_edges_v is None or single_focus_intensity is None:
				raise RuntimeError("Focus average post-processing failed due to missing map state")
			focus_average_base = cfg.output.output_folder / f"{base_stem}_focus_average"
			focus_average_png = focus_average_base.with_suffix(".png")
			focus_average_npz = focus_average_base.with_suffix(".npz")

		average_plane_label = "focus" if cfg.focus.mode == "focused" else "projection"
		average_world_base = cfg.output.output_folder / f"{Path(cfg.output.output_filename).stem}_{average_plane_label}_average_world"
		average_world_png = average_world_base.with_suffix(".png")
		average_world_npz = average_world_base.with_suffix(".npz")
		plane_u_min, plane_u_max, plane_v_min, plane_v_max = _projection_plane_limits(cfg.projection)
		axis_limits = (plane_u_min, plane_u_max, plane_v_min, plane_v_max)
		axis_limits_world = (
			plane_u_min + cfg.projection.center[0],
			plane_u_max + cfg.projection.center[0],
			plane_v_min + cfg.projection.center[1],
			plane_v_max + cfg.projection.center[1],
		)
		if single_projection_for_png is None:
			raise RuntimeError("Projection comparison PNG is missing a single-sample map")

		_save_average_with_single_map_png(
			single_projection_for_png,
			avg_projection_intensity,
			avg_projection_edges_u,
			avg_projection_edges_v,
			projection_average_png,
			single_projection_title,
			f"Average projection intensity ({avg_count} batches)",
			axis_limits=axis_limits,
		)
		np.savez_compressed(
			projection_average_npz,
			intensity=avg_projection_intensity,
			u_edges=avg_projection_edges_u,
			v_edges=avg_projection_edges_v,
		)
		if focus_average_png is not None and focus_average_npz is not None and avg_focus_intensity is not None and avg_focus_edges_u is not None and avg_focus_edges_v is not None:
			if single_focus_for_png is None:
				raise RuntimeError("Focus comparison PNG is missing a single-sample map")
			_save_average_with_single_map_png(
				single_focus_for_png,
				avg_focus_intensity,
				avg_focus_edges_u,
				avg_focus_edges_v,
				focus_average_png,
				single_focus_title,
				f"Average focus intensity ({avg_count} batches)",
				axis_limits=axis_limits,
			)
			np.savez_compressed(
				focus_average_npz,
				intensity=avg_focus_intensity,
				u_edges=avg_focus_edges_u,
				v_edges=avg_focus_edges_v,
			)
		_save_intensity_map_png(
			avg_plane_intensity_world,
			avg_edges_u_world,
			avg_edges_v_world,
			average_world_png,
			f"Average {average_plane_label} intensity map in world coordinates ({avg_count} batches)",
			axis_limits=axis_limits_world,
		)
		np.savez_compressed(
			average_world_npz,
			intensity=avg_plane_intensity_world,
			u_world_edges=avg_edges_u_world,
			v_world_edges=avg_edges_v_world,
		)
		print(f"Saved: {projection_average_png}")
		print(f"Saved: {projection_average_npz}")
		if focus_average_png is not None and focus_average_npz is not None:
			print(f"Saved: {focus_average_png}")
			print(f"Saved: {focus_average_npz}")
		print(f"Saved: {average_world_png}")
		print(f"Saved: {average_world_npz}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate .ry ray files from TOML projection settings"
	)
	parser.add_argument(
		"--config",
		type=Path,
		required=True,
		help="Path to TOML config file",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	try:
		run(args.config.resolve())
	except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
		raise SystemExit(f"Invalid configuration or runtime setup: {exc}") from exc


if __name__ == "__main__":
	main()
