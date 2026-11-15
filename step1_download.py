"""
STEP 1 - Download features (Sentinel-2) and labels (ESA WorldCover) for one area.

Usage:
    python step1_download.py toulouse
    python step1_download.py montauban

Creates data/<area>.npz with:
    X        (height, width, 13) float32  - 10 bands + NDVI, NDWI, NDBI
    labels   (height, width) uint8        - our 5 classes (255 = ignored)
    valid    (height, width) bool         - pixel has data and no cloud
and data/<area>_rgb.png, a true-colour preview.
"""
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from common import (AREAS, BANDS, COLLECTION, DATA, DATES, MAX_CLOUD, STAC_URL,
                    FEATURES, to_classes, worldcover_url)

PIXEL = 10  # metres


def find_scene(bbox):
    """Least cloudy Sentinel-2 L2A scene covering bbox in DATES."""
    items = list(Client.open(STAC_URL).search(
        collections=[COLLECTION], bbox=bbox, datetime=DATES,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items())
    if not items:
        raise SystemExit("No scene found: widen DATES or raise MAX_CLOUD in common.py")
    # prefer scenes that fully contain the area, then the least cloudy
    best = min(items, key=lambda it: it.properties["eo:cloud_cover"])
    print(f"  {len(items)} scenes found, using {best.id} "
          f"({best.datetime:%Y-%m-%d}, {best.properties['eo:cloud_cover']:.1f}% clouds)")
    return best


def target_grid(bbox, crs):
    """A 10 m grid in the scene's projection that covers bbox."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = zip(*[tr.transform(x, y) for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])])
    left = math.floor(min(xs) / PIXEL) * PIXEL
    top = math.ceil(max(ys) / PIXEL) * PIXEL
    width = int(math.ceil((max(xs) - left) / PIXEL))
    height = int(math.ceil((top - min(ys)) / PIXEL))
    return {"crs": crs, "transform": from_origin(left, top, PIXEL, PIXEL),
            "width": width, "height": height}


def read_on_grid(href, grid, resampling):
    """Read any raster (any projection or resolution) onto our grid.
    Only the needed part of the cloud file is downloaded."""
    with rasterio.open(href) as src, WarpedVRT(
            src, crs=grid["crs"], transform=grid["transform"],
            width=grid["width"], height=grid["height"], resampling=resampling) as vrt:
        return vrt.read(1)


def scale_offset(item, asset):
    """Convert stored integers to reflectance (0-1).
    Earth Search often removes the -0.1 offset of processing baseline >= 04.00 from
    the pixels already ("earthsearch:boa_offset_applied") but still lists it in the
    metadata, so applying it again would make reflectances negative."""
    info = (asset.extra_fields.get("raster:bands") or [{}])[0]
    offset = 0.0 if item.properties.get("earthsearch:boa_offset_applied") else info.get("offset", 0.0)
    return info.get("scale", 0.0001), offset


def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def main(area):
    bbox = AREAS[area]
    DATA.mkdir(exist_ok=True)
    print(f"[{area}] searching Sentinel-2...")
    item = find_scene(bbox)

    with rasterio.open(item.assets["red"].href) as src:
        grid = target_grid(bbox, src.crs)
    print(f"  grid: {grid['width']} x {grid['height']} pixels of 10 m")

    bands = {}
    for name in BANDS:
        print(f"  reading {name}...")
        raw = read_on_grid(item.assets[name].href, grid, Resampling.bilinear).astype("float32")
        scale, offset = scale_offset(item, item.assets[name])
        band = raw * scale + offset
        band[raw == 0] = np.nan
        bands[name] = band

    # Scene Classification Layer: remove clouds, cloud shadows, cirrus, snow
    scl = read_on_grid(item.assets["scl"].href, grid, Resampling.nearest)
    cloudy = np.isin(scl, [3, 8, 9, 10, 11])

    indices = {
        "ndvi": nd(bands["nir"], bands["red"]),
        "ndwi": nd(bands["green"], bands["nir"]),
        "ndbi": nd(bands["swir16"], bands["nir"]),
    }
    X = np.stack([bands[b] for b in BANDS] + [indices[i] for i in indices], axis=-1)

    print("  reading ESA WorldCover labels...")
    lon_c, lat_c = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    worldcover = read_on_grid(worldcover_url(lon_c, lat_c), grid, Resampling.nearest)
    labels = to_classes(worldcover)

    valid = np.isfinite(X).all(axis=-1) & ~cloudy
    np.savez_compressed(DATA / f"{area}.npz", X=X.astype("float32"), labels=labels,
                        valid=valid, features=np.array(FEATURES),
                        scene=item.id, date=f"{item.datetime:%Y-%m-%d}")

    rgb = np.clip(np.stack([bands["red"], bands["green"], bands["blue"]], -1) / 0.3, 0, 1)
    plt.imsave(DATA / f"{area}_rgb.png", np.nan_to_num(rgb))

    print(f"  saved data/{area}.npz - {valid.mean():.0%} usable pixels, "
          f"{(labels != 255).mean():.0%} with a label")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in AREAS:
        raise SystemExit(f"Usage: python step1_download.py [{' | '.join(AREAS)}]")
    main(sys.argv[1])
