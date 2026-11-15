"""Shared settings and helpers for the land-cover project."""
import os
from pathlib import Path

# Faster, anonymous reads of the public cloud files (Sentinel-2 + ESA WorldCover)
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")

DATA = Path("data")
OUT = Path("outputs")

# Areas of interest: lon_min, lat_min, lon_max, lat_max
AREAS = {
    "toulouse": [1.35, 43.55, 1.52, 43.66],    # training area
    "montauban": [1.30, 43.97, 1.42, 44.05],   # test area (never seen in training)
}

# Summer 2021, to match the ESA WorldCover 2021 labels
DATES = "2021-06-01/2021-08-31"
MAX_CLOUD = 10
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Sentinel-2 bands used as features (Earth Search asset names)
BANDS = ["blue", "green", "red", "nir",                     # 10 m: B02 B03 B04 B08
         "rededge1", "rededge2", "rededge3", "nir08",        # 20 m: B05 B06 B07 B8A
         "swir16", "swir22"]                                 # 20 m: B11 B12
INDICES = ["ndvi", "ndwi", "ndbi"]
FEATURES = BANDS + INDICES

# ESA WorldCover codes -> our 5 classes
CLASSES = ["Forest", "Grass/shrub", "Cropland", "Built-up", "Water"]
WORLDCOVER_TO_CLASS = {10: 0, 20: 1, 30: 1, 40: 2, 50: 3, 80: 4}
CLASS_COLORS = ["#1a7a1a", "#a6d96a", "#f1d55e", "#d7301f", "#2166ac"]
NO_CLASS = 255


def worldcover_url(lon, lat):
    """URL of the 3x3 degree WorldCover 2021 tile that contains (lon, lat)."""
    import math
    la = int(math.floor(lat / 3) * 3)
    lo = int(math.floor(lon / 3) * 3)
    ns = f"N{la:02d}" if la >= 0 else f"S{-la:02d}"
    ew = f"E{lo:03d}" if lo >= 0 else f"W{-lo:03d}"
    return ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif")


def to_classes(worldcover):
    """Convert a WorldCover array to our class ids (255 = ignored)."""
    import numpy as np
    out = np.full(worldcover.shape, NO_CLASS, dtype="uint8")
    for code, cls in WORLDCOVER_TO_CLASS.items():
        out[worldcover == code] = cls
    return out
