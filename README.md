# PCHE Nu-f Correlation Benchmarking Tool

Open-source benchmarking tool for Nusselt number (Nu) and friction factor (f)
correlations in printed circuit heat exchangers (PCHE), developed for
high-pressure hydrogen precooling systems (SAE J2601 refueling context).

## What this does

Compares published PCHE channel-geometry correlations against a straight-channel
baseline using the Performance Evaluation Criterion (PEC), across the hydrogen
precooling design space (temperature, pressure, Reynolds number, channel angle).

**Geometries covered:**
- Zigzag channels — Kim & No (2011), Meshram et al. (2016)
- S-shaped channels — Ngo et al. (2007)
- Straight channels — Dittus-Boelter / Blasius (baseline reference)

**Design space:**
- Temperature: 233–333 K (-40 to 60 C)
- Pressure: 350 and 700 bar
- Reynolds number: 2,000–40,000
- Zigzag channel angle: 30-60 deg

## Files

| File | Purpose |
|---|---|
| `hydrogen_properties.py` | Real-gas para-hydrogen property fits (Pr, rho, mu, k, cp) from NIST data |
| `correlations.py` | Nu/f correlation library + single-condition PEC benchmark |
| `benchmark_sweep.py` | Full T x P x angle x Re sweep, exports `benchmark_results.csv` |
| `plots.py` | Reads the CSV and generates Nu/f/PEC vs Re comparison plots |

## Usage

```bash
pip install -r requirements.txt

# Quick single-condition check
python3 correlations.py

# Full design-space sweep -> benchmark_results.csv
python3 benchmark_sweep.py

# Generate comparison plots from the sweep results
python3 plots.py
```

## Performance Evaluation Criterion (PEC)

PEC = (Nu / Nu_ref) / (f / f_ref)^(1/3)

- PEC > 1: better heat transfer per unit pumping-power penalty than the
  straight-channel baseline
- PEC = 1: equivalent to baseline
- PEC < 1: worse than baseline

## Status / roadmap

- [x] Hydrogen property fits
- [x] Correlation library (3 geometries + baseline)
- [x] Multi-condition benchmark sweep with CSV export
- [x] Comparison plots
- [ ] Validation against published experimental/CFD data points
- [ ] Extend to additional PCHE geometries (e.g. sinusoidal, airfoil-fin)
- [ ] Feed into first-author paper on Nu-f correlation benchmarking for
      high-pressure hydrogen precooling

## Author

Aakash Yadav, Mechanical Engineering, Delhi Technological University (DTU)
Supervised by Prof. Anil Kumar
