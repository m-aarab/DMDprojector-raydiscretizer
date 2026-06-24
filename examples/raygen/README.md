# Mixed-Mask Raygen TOML Testcases

These configs target the unified TOML ray generator:

Run from repository root:

```bash
python3 programs/raygenerators/projectionImageGen.py --config examples/raygen/triangle_continuous.toml
```

Output files are written to:

- examples/raygen/output

Pixelated mode supports projector-plane jitter via `pixels.projector_shift_max_pixels` in TOML.
When batch generation is used, the generator also writes averaged intensity maps with a
centered-sample-vs-average PNG layout (shared colorbar scale across both panels):

- `<output_stem>_projection_average.png/.npz` for collimated mode
- `<output_stem>_projection_average.png/.npz` and `<output_stem>_focus_average.png/.npz` for focused mode

Included representative mixed-mask cases:

1. `triangle_continuous.toml`
2. `triangle_pixelated_flat.toml`
3. `circle_antialiased_flat.toml`
4. `annulus_pixelated_focused.toml`
5. `rectangle_gaussian_pixelated.toml`
6. `triangle_gaussian_antialiased.toml`
7. `circle_gaussian_focused.toml`
8. `annulus_gaussian_antialiased_focused.toml`
9. `annulus_gaussian_antialiased_focused_avg_jitter2px.toml`
10. `annulus_gaussian_center_focused_avg_jitter2px.toml`
