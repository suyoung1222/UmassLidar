"""Orthomosaic GeoTIFF tiles -> ground colour on the twin's local UTM grid.

Written for the EE2023 campus deliverable (`AllCampus_Spring2023_Edits_v3/Ortho Tiles`): 3-band
uint8 GeoTIFFs in EPSG:2249 (NAD83 / Massachusetts Mainland, US survey feet), 0.1008 ft per pixel.
No GDAL / pyproj needed: the two projections are evaluated in numpy on a coarse grid and
interpolated (both are smooth over a site of a few hundred metres), and the tiles are decoded
with Pillow (they are LZW strips, which tifffile cannot read without imagecodecs).

    idx = OrthoTiles(ortho_dir)
    rgb, valid = idx.render(origin_utm_xy, size_xy, res)      # (ny, nx, 3) uint8, row 0 = south
"""
from pathlib import Path

import numpy as np

from .resepi import utm_forward

FT_US = 1200.0 / 3937.0


def utm_inverse(e, n, zone=18, iters=6):
    """UTM easting / northing -> lat, lon (deg) by Newton iteration on utm_forward."""
    e, n = np.asarray(e, np.float64), np.asarray(n, np.float64)
    lat = n / 111132.0
    lon = (zone * 6 - 183) + (e - 500000.0) / (111320.0 * np.cos(np.radians(lat)))
    d = 1e-6
    for _ in range(iters):
        x0, y0 = utm_forward(lat, lon, zone)
        xa, ya = utm_forward(lat + d, lon, zone)
        xb, yb = utm_forward(lat, lon + d, zone)
        j11, j12, j21, j22 = (xa - x0) / d, (xb - x0) / d, (ya - y0) / d, (yb - y0) / d
        det = j11 * j22 - j12 * j21
        rx, ry = e - x0, n - y0
        lat = lat + (j22 * rx - j12 * ry) / det
        lon = lon + (-j21 * rx + j11 * ry) / det
    return lat, lon


def ma_mainland_ft(lat_deg, lon_deg):
    """lat, lon (deg) -> EPSG:2249 easting / northing in US survey feet (Lambert conformal conic 2SP,
    GRS80; standard parallels 42d41' and 41d43', origin 41N 71d30'W, false E/N 200 km / 750 km)."""
    a, f = 6378137.0, 1 / 298.257222101
    e = np.sqrt(f * (2 - f))
    r = np.radians
    p1, p2, p0, l0 = r(42 + 41 / 60), r(41 + 43 / 60), r(41.0), r(-71.5)
    m = lambda p: np.cos(p) / np.sqrt(1 - (e * np.sin(p)) ** 2)                                   # noqa: E731
    t = lambda p: np.tan(np.pi / 4 - p / 2) / ((1 - e * np.sin(p)) / (1 + e * np.sin(p))) ** (e / 2)  # noqa: E731
    nn = (np.log(m(p1)) - np.log(m(p2))) / (np.log(t(p1)) - np.log(t(p2)))
    F = m(p1) / (nn * t(p1) ** nn)
    rho = lambda p: a * F * t(p) ** nn                                                           # noqa: E731
    th = nn * (r(lon_deg) - l0)
    return (200000.0 + rho(r(lat_deg)) * np.sin(th)) / FT_US, (750000.0 + rho(p0) - rho(r(lat_deg)) * np.cos(th)) / FT_US


class OrthoTiles:
    """Index of north-up GeoTIFF tiles (pixel scale + tiepoint tags) in one projected CRS."""

    def __init__(self, folder, to_crs=ma_mainland_ft, zone=18):
        import logging

        import tifffile
        logging.getLogger("tifffile").setLevel(logging.ERROR)       # GDAL_NODATA float in an int tag: harmless
        self.to_crs, self.zone, self.tiles = to_crs, zone, []
        for f in sorted(Path(folder).expanduser().glob("*.tif")):
            with tifffile.TiffFile(f) as t:
                p = t.pages[0]
                sx, sy = p.tags["ModelPixelScaleTag"].value[:2]
                tp = p.tags["ModelTiepointTag"].value
                x0, y0 = tp[3] - tp[0] * sx, tp[4] + tp[1] * sy          # outer corner of pixel (0, 0)
                h, w = p.shape[:2]
            self.tiles.append(dict(path=f, x0=x0, y0=y0, sx=sx, sy=sy, w=w, h=h))
        if not self.tiles:
            raise FileNotFoundError(f"no GeoTIFF tiles in {folder}")

    def _crs_grid(self, origin_xy, size_xy, step=20.0):
        """Projected coordinates on a coarse local grid (rows = y, cols = x)."""
        gx = np.arange(0.0, size_xy[0] + step, step); gy = np.arange(0.0, size_xy[1] + step, step)
        X, Y = np.meshgrid(origin_xy[0] + gx, origin_xy[1] + gy)
        lat, lon = utm_inverse(X, Y, self.zone)
        return gx, gy, self.to_crs(lat, lon)

    def render(self, origin_xy, size_xy, res, shift=(0.0, 0.0), verbose=True):
        """Ortho resampled to the local grid: pixel (i, j) is centred at local ((j + .5) res, (i + .5) res).
        `shift` (m, local x / y) is added to the sample position. Returns (rgb uint8, valid bool)."""
        import cv2
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        nx, ny = int(round(size_xy[0] / res)), int(round(size_xy[1] / res))
        gx, gy, (E, N) = self._crs_grid(origin_xy, size_xy)
        # coarse grid -> per-pixel projected coordinates (bilinear; the mapping is nearly affine)
        cx = ((np.arange(nx) + 0.5) * res + shift[0]) / (gx[1] - gx[0])
        cy = ((np.arange(ny) + 0.5) * res + shift[1]) / (gy[1] - gy[0])
        out = np.zeros((ny, nx, 3), np.uint8); valid = np.zeros((ny, nx), bool)
        e_lo, e_hi, n_lo, n_hi = E.min(), E.max(), N.min(), N.max()
        for t in self.tiles:
            tx1, ty1 = t["x0"] + t["w"] * t["sx"], t["y0"] - t["h"] * t["sy"]
            if t["x0"] > e_hi or tx1 < e_lo or t["y0"] < n_lo or ty1 > n_hi:
                continue
            if verbose:
                print(f"[ortho] reading {t['path'].name[-12:]} ({t['w']}x{t['h']})", flush=True)
            img = np.asarray(Image.open(t["path"]).convert("RGB"))
            block = 1024
            # bilinear interpolation of the coarse projection grid, done by hand: cv2.remap quantises
            # coordinates to 1/32 of a (20 m) cell, which shows up as 0.6 m blocks
            Et, Nt = (E - t["x0"]).astype(np.float32), (t["y0"] - N).astype(np.float32)
            jx = np.clip(np.floor(cx).astype(int), 0, Et.shape[1] - 2); fx = (cx - jx).astype(np.float32)[None, :]
            for i0 in range(0, ny, block):
                yy = cy[i0:i0 + block]
                iy = np.clip(np.floor(yy).astype(int), 0, Et.shape[0] - 2); fy = (yy - iy).astype(np.float32)[:, None]

                def lerp(A):
                    top = A[iy][:, jx] * (1 - fx) + A[iy][:, jx + 1] * fx
                    bot = A[iy + 1][:, jx] * (1 - fx) + A[iy + 1][:, jx + 1] * fx
                    return top * (1 - fy) + bot * fy
                e, n = lerp(Et), lerp(Nt)
                u, v = e / np.float32(t["sx"]) - 0.5, n / np.float32(t["sy"]) - 0.5
                inside = (u >= 0) & (u <= t["w"] - 1) & (v >= 0) & (v <= t["h"] - 1)
                if not inside.any():
                    continue
                c0, c1 = max(int(u[inside].min()) - 2, 0), min(int(u[inside].max()) + 3, t["w"])
                r0, r1 = max(int(v[inside].min()) - 2, 0), min(int(v[inside].max()) + 3, t["h"])
                px = cv2.remap(img[r0:r1, c0:c1], u - c0, v - r0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                ok = inside & (px.max(-1) > 0) & ~valid[i0:i0 + block]      # 0,0,0 = no data in the tiles
                out[i0:i0 + block][ok] = px[ok]; valid[i0:i0 + block] |= ok
            del img
        return out, valid
