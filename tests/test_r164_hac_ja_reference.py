"""Compare the standalone r164_hac_ja_driver executable with direct NumPy sums.

Usage: python test_r164_hac_ja_reference.py path/to/r164_hac_ja_driver
"""
import pathlib
import subprocess
import sys
import tempfile
import numpy as np

def main():
    exe = pathlib.Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        for n, nc, interval in [(31, 31, 3), (31, 31, 1), (2, 2, 1)]:
            case = pathlib.Path(tmp) / f"{n}_{interval}"
            case.mkdir()
            subprocess.run([str(exe), str(n), str(nc), str(interval)], cwd=case, check=True)
            i = np.arange(n)
            mask = (i % 7 != 1) & (i % 11 != 3)
            values = np.stack([np.sin(i * (j + 1) * .031) + i * .0001 * (j + 1) for j in range(9)], axis=1)
            values[mask] -= values[mask].mean(axis=0)
            values[~mask] = 0
            for kind, name in enumerate(("ja", "hardy", "centroid")):
                data = np.loadtxt(case / ("hac_" + name + ".out"), ndmin=2, converters=lambda x: float("nan") if "nan" in str(x).lower() else float(x))
                lags = np.arange(nc // interval) * interval
                np.testing.assert_allclose(data[:, 0], (lags + (interval - 1) / 2) / 1000)
                corr, pairs = [], []
                for lag in range(nc):
                    both = mask[:n-lag] & mask[lag:]
                    pairs.append(both.sum())
                    x = values[:n-lag, kind*3:kind*3+3][both]
                    y = values[lag:, kind*3:kind*3+3][both]
                    corr.append((x.T @ y / both.sum()).ravel() if both.any() else np.full(9, np.nan))
                corr = np.array(corr)
                integ = np.zeros_like(corr)
                if not pairs[0]:
                    integ[0] = np.nan
                integ[1:] = np.cumsum(corr[:-1] + corr[1:], axis=0)
                np.testing.assert_allclose(data[:, 1], np.array(pairs)[:len(lags)*interval].reshape(-1, interval).mean(axis=1))
                np.testing.assert_allclose(data[:, 2:11], corr[:len(lags)*interval].reshape(-1, interval, 9).mean(axis=1), rtol=1e-8, atol=1e-8, equal_nan=True)
                np.testing.assert_allclose(data[:, 11:], integ[:len(lags)*interval].reshape(-1, interval, 9).mean(axis=1), rtol=1e-8, atol=1e-8, equal_nan=True)
    print("R164 HAC reference checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python test_r164_hac_ja_reference.py DRIVER")
    main()
