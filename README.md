# Ray Generators

This folder contains TOML-driven ray generation scripts.

## projectionImageGen.py

Unified configurable projection ray generator that writes proprietary `.ry` files.

Run:

```bash
cd /home/marwan/programs/MIRAGE/programs/raygenerators
python3 projectionImageGen.py --config projectionImageGen.example.toml
```

Generate all curated examples used below:

```bash
cd /home/marwan/programs/MIRAGE
for cfg in \
  examples/raygen/triangle_continuous.toml \
  examples/raygen/triangle_pixelated_flat.toml \
  examples/raygen/circle_antialiased_flat.toml \
  examples/raygen/annulus_pixelated_focused.toml \
  examples/raygen/rectangle_gaussian_pixelated.toml \
  examples/raygen/triangle_gaussian_antialiased.toml \
  examples/raygen/circle_gaussian_focused.toml \
  examples/raygen/annulus_gaussian_antialiased_focused.toml \
  examples/raygen/annulus_gaussian_antialiased_focused_avg_jitter2px.toml \
  examples/raygen/annulus_gaussian_center_focused_avg_jitter2px.toml; do
  python3 programs/raygenerators/projectionImageGen.py --config "$cfg"
done
```

### Supported features

- Projection setup: width, height, axis, direction sign (+/-), focus/source plane positions.
- Base intensity in W/m^2.
- Masks: rectangle, circle, triangle, annulus (circle with central hole).
- Origin types:
  - `continuous`: sample directly in mask until `n_rays` is reached.
  - `pixelated`: mask -> pixel activation -> ray emission.
- Pixel activation:
  - `center`: on/off by pixel center.
  - `antialiased`: intensity scaling by in-mask area fraction.
- Pixel intensity modes:
  - `flat`
  - `gaussian` (peak equals activated pixel intensity)
- Ray direction modes:
  - `collimated` along projection direction.
  - `focused` by sampling a cone direction and back-projecting to source plane.
- Batch generation with indexed filenames and optional deterministic seeding.
- Optional energy renormalization for pixelated outputs so batch energy matches
  `base_intensity * mask_area` exactly (`output.normalize_batch_energy_to_mask_power`).
- Optional random in-plane projector jitter for pixelated emission while keeping the
  mask fixed (`pixels.projector_shift_max_pixels`, in pixel-length units).

### Output

- Proprietary `ray file v1` format (`.ry`):
  - header lines
  - number of rays
  - `ox oy oz dx dy dz energy`
- Intensity-map PNG(s) saved next to each `.ry` with matching stem:
  - `<stem>_projection.png` when `focus.mode = "collimated"`
  - `<stem>_projection_focus.png` when `focus.mode = "focused"` (single figure with both maps)
  - focused/projection panels in the combined figure use shared u/v axis limits
  - focused/projection panels in the combined figure use a shared colorbar scale
- Average projection map across all generated batches:
  - when `focus.mode = "collimated"`: `<output_stem>_projection_average.png/.npz`
  - when `focus.mode = "focused"`: `<output_stem>_projection_average.png/.npz` and `<output_stem>_focus_average.png/.npz`
  - average PNGs are two-panel figures: `centered sample (no shift)` (left) and `batch average` (right)
  - both panels in each average PNG use the same colorbar scale for direct comparison
  - `.npz` contains averaged intensity and u/v bin edges

### Representative Configurations (Mixed Mask Showcase)

The following gallery uses configs in `examples/raygen/` and PNG assets stored in
`doc/img/raygenerators/`.
These curated pixelated examples intentionally use large pixels (`pitch = [50, 50]` on a
`600 x 600` projection plane) with high `rays_per_pixel` values.

1. Continuous sampling (`continuous`, `collimated`, mask: `triangle`)

  Config: [triangle_continuous.toml](../../examples/raygen/triangle_continuous.toml)

  <img src="../../doc/img/raygenerators/triangle_continuous_projection.png" alt="triangle continuous projection" width="420" />

2. Pixelated center activation, flat intensity (`pixelated`, `center`, `flat`, `collimated`, mask: `triangle`)

  Config: [triangle_pixelated_flat.toml](../../examples/raygen/triangle_pixelated_flat.toml)

  <img src="../../doc/img/raygenerators/triangle_pixelated_flat_projection.png" alt="triangle pixelated flat projection" width="420" />

3. Pixelated antialiased activation, flat intensity (`pixelated`, `antialiased`, `flat`, `collimated`, mask: `circle`)

  Config: [circle_antialiased_flat.toml](../../examples/raygen/circle_antialiased_flat.toml)

  <img src="../../doc/img/raygenerators/circle_antialiased_flat_projection.png" alt="circle antialiased flat projection" width="420" />

4. Pixelated center activation, flat intensity, focused (`pixelated`, `center`, `flat`, `focused`, mask: `annulus`)

  Config: [annulus_pixelated_focused.toml](../../examples/raygen/annulus_pixelated_focused.toml)

  <img src="../../doc/img/raygenerators/annulus_pixelated_focused_projection_focus.png" alt="annulus pixelated focused projection and focus" width="720" />

5. Pixelated center activation, Gaussian intensity (`pixelated`, `center`, `gaussian`, `collimated`, mask: `rectangle`)

  Config: [rectangle_gaussian_pixelated.toml](../../examples/raygen/rectangle_gaussian_pixelated.toml)

  <img src="../../doc/img/raygenerators/rectangle_gaussian_pixelated_projection.png" alt="rectangle gaussian pixelated projection" width="420" />

6. Pixelated antialiased activation, Gaussian intensity (`pixelated`, `antialiased`, `gaussian`, `collimated`, mask: `triangle`)

  Config: [triangle_gaussian_antialiased.toml](../../examples/raygen/triangle_gaussian_antialiased.toml)

  <img src="../../doc/img/raygenerators/triangle_gaussian_antialiased_projection.png" alt="triangle gaussian antialiased projection" width="420" />

7. Pixelated center activation, Gaussian intensity, focused (`pixelated`, `center`, `gaussian`, `focused`, mask: `circle`)

  Config: [circle_gaussian_focused.toml](../../examples/raygen/circle_gaussian_focused.toml)

  <img src="../../doc/img/raygenerators/circle_gaussian_focused_projection_focus.png" alt="circle gaussian focused projection and focus" width="720" />

8. Pixelated antialiased activation, Gaussian intensity, focused (`pixelated`, `antialiased`, `gaussian`, `focused`, mask: `annulus`)

  Config: [annulus_gaussian_antialiased_focused.toml](../../examples/raygen/annulus_gaussian_antialiased_focused.toml)

  <img src="../../doc/img/raygenerators/annulus_gaussian_antialiased_focused_projection_focus.png" alt="annulus gaussian antialiased focused projection and focus" width="720" />

9. Averaging case with projector jitter (`pixelated`, `antialiased`, `gaussian`, `focused`, mask: `annulus`, jitter: 5 pixels, samples: 20)

  Config: [annulus_gaussian_antialiased_focused_avg_jitter2px.toml](../../examples/raygen/annulus_gaussian_antialiased_focused_avg_jitter2px.toml)

  <img src="../../doc/img/raygenerators/annulus_gaussian_antialiased_focused_avg_jitter2px_projection_average.png" alt="annulus jittered projection-plane single sample and average intensity map" width="720" />

  <img src="../../doc/img/raygenerators/annulus_gaussian_antialiased_focused_avg_jitter2px_focus_average.png" alt="annulus jittered focus-plane single sample and average intensity map" width="720" />

10. Averaging case with projector jitter, no antialiasing (`pixelated`, `center`, `gaussian`, `focused`, mask: `annulus`, jitter: 5 pixels, samples: 20)

  Config: [annulus_gaussian_center_focused_avg_jitter2px.toml](../../examples/raygen/annulus_gaussian_center_focused_avg_jitter2px.toml)

  <img src="../../doc/img/raygenerators/annulus_gaussian_center_focused_avg_jitter2px_projection_average.png" alt="annulus jittered no-aa projection-plane single sample and average intensity map" width="720" />

  <img src="../../doc/img/raygenerators/annulus_gaussian_center_focused_avg_jitter2px_focus_average.png" alt="annulus jittered no-aa focus-plane single sample and average intensity map" width="720" />

### Intensity-map binning

- Pixelated + `flat`: default 1 bin per pixel (bin size = pixel pitch).
- Pixelated + `gaussian`: default `bins_per_pixel = [5, 5]` (5x5 bins per pixel).
- Continuous: auto bin size from ray density over mask area.

### Notes

- Image-mask import is intentionally not implemented yet.
- Relative paths in TOML are resolved from the TOML file location.
