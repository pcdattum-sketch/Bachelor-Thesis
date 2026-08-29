"""
features_extraction
====================

The four GEE extraction runs that build up the SOC-S2 thesis's
feature table. This is a single-file that show how all features were extracted from the S2 images, terrain, climate, peat, and soil texture. The final output is one file includes all samples and their features to be used in scripts folder.

Run each stage's main function in order, one at a time

    main_s2_bare_soil()        
    main_terrain_climate()     
    main_peat_and_lab()       
    main_legacy_soil_rasters()  
    
Configuration (paths, GEE project id, export folder names) is from local machine. Edit the
values there to match your own machine before running anything.
"""

import argparse
import json

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import seaborn as sns

# ============================================================================
# config  --  Edit the values below to match your own machine/GEE account
# ----------------------------------------------------------------------------

# --- Google Earth Engine ----
# This assumes you've already run `earthengine authenticate` once on your
# machine and have access to your own GEE project and own credentials 
GEE_PROJECT = "soc-s2"  # <-- replace with your own GEE project id

# --- Raw LUCAS survey data ---
LUCAS_CSV_PATH = "LUCAS-SOIL-2018.csv"  # <-- replace with your own path

# --- First extraction: Sentinel-2 bare soil extraction ---
# GEE exports will be in your Google Drive folder under the same project name,
# the bare-soil CSV must be downloaded from Drive by hand before
# the next step.
GEE_EXPORT_FOLDER_S2_BARE_SOIL = "SOC_S2_bare_soil"
# Where the downloaded bare-soil extraction CSV lives, use your own pathway for the file
S2_BARE_SOIL_CSV_PATH = "LUCAS_S2_bare_soil.csv"  # <-- replace with your own path

# --- Ammending terrain + climate features  ---
GEE_EXPORT_FOLDER_TERRAIN_CLIMATE = "SOC_terrain_climate"
# Where the downloaded terrain/climate extraction CSV lives, use your own pathway for the file
TERRAIN_CLIMATE_CSV_PATH = "LUCAS_terrain_climate.csv"  # <-- replace with your own path
# Output path for the merged features CSV.
SOC_TERRAIN_CLIMATE_OUTPUT_CSV_PATH = "LUCAS_SOC_terrain_climate_features.csv"

# --- Ammending organic soil (peat) feature ---
PEATMAP_SHAPEFILE_PATH = "Thuenen_Kulisse_OrgBoeden_v1"  # <-- replace with your own path
SOC_S2_TERRAIN_CLIMATE_PEAT_OUTPUT_CSV_PATH = "LUCAS_S2_terrain_climate_peat_features.csv"

# --- Ammending soil texture (clay, AWC) features ---
# Legacy soil-property rasters, point these at wherever you keep them
CLAY_RASTER_PATH = "Clay files/Clay_eu23.tif"  # <-- replace with your own path
AWC_RASTER_PATH = "AWC files/AWC_eu23.tif"  # <-- replace with your own path

# Final merged output, ready for the `scripts` repo's cleaning/splitting step.
WORKING_FILE_OUTPUT_CSV_PATH = "Working_files_full_features.csv"

# ============================================================================
# Extract bare-soil Sentinel-2 reflectance at LUCAS 2018 sampling points.
# ----------------------------------------------------------------------------
def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band (bits 10, 11). """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

  return image.updateMask(mask).divide(10000)

def add_pv_ir2(image):
    """Bare-soil index PV_IR2 threshold"""
    pv_ir2 = image.expression(
        "((B8 - B4) / (B8 + B4)) + ((B8 - B12) / (B8 + B12))",
        {
            "B8": image.select("B8"), 
            "B4": image.select("B4"), 
            "B12": image.select("B12"),
        }).rename('PV_IR2')
    return image.addBands([pv_ir2])

def add_nbr2(image): 
    """Bare-soil index BR2 threshold"""
    nbr2 = image.normalizedDifference(['B11', 'B12']).rename('NBR2')
    return image.addBands([nbr2])

def add_ndvi(image): 
    """NDVI and its inverse for qualityMosaic."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    #we need inverse value for qualityMosaic() 
    # a fuction to pick the pixel where the band chosen is highest so we inverse it so that 
    # it picks the most bare soil index 
    ndvi_inv = ndvi.multiply(-1).rename('NDVI_inv') 
    return image.addBands([ndvi, ndvi_inv])

def load_lucas_germany_agriculture(lucas_csv_path):
    """Load the raw LUCAS 2018 CSV, filter to Germany agriculture cropland, clean OC.

    filters NUTS_0 == 'DE' and LU == 'U111', keeps Depth/ POINTID/ OC/ NUTS_0/ LU/ TH_LAT/      TH_LONG, convert OC to numeric and drops rows where OC failed to parse (in the original     data this was 1 single "< LOD" (below level of detection) value).
    """
    df = pd.read_csv(lucas_csv_path)
    df_filtered = df[(df["NUTS_0"] == "DE") & (df["LU"] == "U111")]
    df_filtered = df_filtered[["Depth", "POINTID", "OC", "NUTS_0", "LU", "TH_LAT", "TH_LONG"]].copy()

    df_filtered["OC"] = pd.to_numeric(df_filtered["OC"], errors="coerce")
    n_dropped = df_filtered["OC"].isna().sum()
    if n_dropped:
        print(f"Dropping {n_dropped} row(s) with non-numeric OC")
    df_filtered = df_filtered.dropna(subset=["OC"])

    return df_filtered

def build_bare_soil_composite():
    """Cloud-masked, bare-soil-favoring Sentinel-2 composite, May-Sep 2018."""
    s2_composite = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        # Harmonized collection corrects S2B to match S2A so the whole time
        # series behaves as if it came from a single consistent sensor.
        .filterDate("2018-05-01", "2018-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B2", "B3","B4", "B5", "B6", "B7", "B8A", "B8", "B11", "B12", "QA60"])
        .map(mask_s2_clouds)
        .map(add_ndvi)
        .map(add_pv_ir2)
        .map(add_nbr2)
        .qualityMosaic("NDVI_inv")
    )
    print("Composite bands:", s2_composite.bandNames().getInfo())
    return s2_composite

def main_s2_bare_soil_filter():
    # Uses your own already-authenticated GEE project, swap this line out for your own ee.Initialize() call if you authenticate differently.
    ee.Initialize(project=GEE_PROJECT)

    df_filtered = load_lucas_germany_agriculture(LUCAS_CSV_PATH)
    print(f"{len(df_filtered)} Germany arable-cropland LUCAS points after cleaning") #should be 540

    features = []
    for _, row in df_filtered.iterrows(): #for loop in going to each sample
        point = ee.Geometry.Point([row['TH_LONG'], row['TH_LAT']])
        feature = ee.Feature(point,{'POINTID':row['POINTID'],'OC':row['OC']})
        #here P,N,K, EC, pH values can also be retained, or could do it in later steps 
        features.append(feature)

    lucas_fc = ee.FeatureCollection(features)
    print(f"FeatureCollection size: {lucas_fc.size().getInfo()}") # should be 540

    s2_composite = build_bare_soil_composite()

    #Collecting composite and attached to previous feature collection
    sampled = s2_composite.sampleRegions(
        collection=lucas_fc,
        properties=['POINTID', 'OC'],
        scale=20, 
        geometries=True
    )
    print("Sampled size:", sampled.size().getInfo()) #should be 540

    #Filtering samples with chosen threshold
    bare_soil = sampled.filter(ee.Filter.lt("NBR2", 0.16)).filter(ee.Filter.lt("PV_IR2", 0.32))
    print("Combined PV_IR2 < 0.32 AND NBR2 < 0.16:", bare_soil.size().getInfo(), "points") #should be 310 samples

    task = ee.batch.Export.table.toDrive(
        collection=bare_soil,
        description= f"LUCAS_S2_bare_soil",
        folder=GEE_EXPORT_FOLDER_S2_BARE_SOIL,
        fileFormat="CSV",
    )
    task.start()
    print(f"Export task started (status: {task.status()['state']}). ")
    #Download the CSV from the Drive folder '{GEE_EXPORT_FOLDER_S2_BARE_SOIL}'
    # Once it finishes, the S2_BARE_SOIL_CSV_PATH should be its pathway to continue running 

# ============================================================================
# Extract terrain (Copernicus GLO-30 DEM) and climate (ERA5-Land) features
# ----------------------------------------------------------------------------
          
def load_s2_bare_soil(csv_path):
    df = pd.read_csv(csv_path)
    df_filtered = df[[
        "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "OC", "POINTID",
    ]].copy()

    #Add back longtitude and latitude at each sample using ".geo" which call the shape where GEE exports the features
    lons, lats = [], []
    for geo_str in df[".geo"]:
        geo = json.loads(geo_str)
        lons.append(geo["coordinates"][0])
        lats.append(geo["coordinates"][1])
    df_filtered["lon"] = lons
    df_filtered["lat"] = lats

    return df_filtered

def build_terrain_climate_raster():
    """Terrain (elevation/slope/aspect) + climate (temp_mean/precip_mean) raster."""
    dataset = ee.ImageCollection("COPERNICUS/DEM/GLO30")
    # GLO30 exists as tiled images that must be mosaicked; otherwise Earth
    # Engine's default projection for mosaics (EPSG:4326 at 1-degree scale is used
    dem = dataset.select("DEM").mosaic()

    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope")
    aspect = terrain.select("aspect")
    elevation = dem.rename("elevation")

    terrain_raster = elevation.addBands(slope).addBands(aspect)
    print("Terrain bands:", terrain_raster.bandNames().getInfo()) 
    #should give Terrain bands: ['elevation', 'slope', 'aspect']


    # Temperature at 2m above ground, precipitation as total precipitation
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate("1991-01-01", "2021-01-01")
        .select(["temperature_2m", "total_precipitation_sum"])
    )

    # Mean annual temperature: temperature is already a per-month mean, so
    # mean() over the 30-year record gives the mean annual temperature.
    temp_mean_k = era5.select("temperature_2m").mean()
    temp_mean_c = temp_mean_k.subtract(273.15).rename("temp_mean")

    # Mean annual precipitation: total_precipitation_sum is total metres per
    # month, so sum() over the record then divide by 30 years gives the
    # annual total, converted from metres to millimetres.
    precip_annual_total_m = era5.select("total_precipitation_sum").sum().divide(30)
    precip_mean_mm = precip_annual_total_m.multiply(1000).rename("precip_mean")

    terrain_climate_raster = terrain_raster.addBands(temp_mean_c).addBands(precip_mean_mm)
    print("All bands:", terrain_climate_raster.bandNames().getInfo())
    #should give All bands: ['elevation', 'slope', 'aspect', 'temp_mean', 'precip_mean']
    return terrain_climate_raster


def main_terrain_climate():
    # Same GEE project as main_s2_bare_soil() above.
    ee.Initialize(project=GEE_PROJECT)

    df_filtered = load_s2_bare_soil(S2_BARE_SOIL_CSV_PATH)

    features = [
        ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]]), {"POINTID": row["POINTID"]})
        for _, row in df_filtered.iterrows()
    ]
    mod_fc = ee.FeatureCollection(features)
    print("FeatureCollection size:", mod_fc.size().getInfo()) #should be 310 

    terrain_climate_raster = build_terrain_climate_raster()

    # scale=30 matches the Copernicus DEM's native resolution, and is
    # consistent with the S2 extraction's 20m (no artificial upsampling; DEM
    # is the resolution-limiting layer here).
    terrain_climate_samples = terrain_climate_raster.sampleRegions(
        collection=mod_fc,
        properties=["POINTID"],
        scale=30,
        tileScale=4,  # helps avoid memory errors on large jobs
        geometries=False,
    )
    print("Sample count:", terrain_climate_samples.size().getInfo())

    task = ee.batch.Export.table.toDrive(
        collection= terrain_climate_samples,
        description="LUCAS_terrain_climate",
        folder=GEE_EXPORT_FOLDER_TERRAIN_CLIMATE,
        fileNamePrefix="LUCAS_terrain_climate",
        fileFormat="CSV",
    )
    task.start()
    print(f"Export task started. Download the CSV from the Drive folder ")
    # {GEE_EXPORT_FOLDER_TERRAIN_CLIMATE}' once it finishes, and point 
    # TERRAIN_CLIMATE_CSV_PATH at it, then re-run this script's 
    #`merge_and_save` step (or call it directly) to produce the final merged dataset

    if TERRAIN_CLIMATE_CSV_PATH:
        merge_and_save(df_filtered, TERRAIN_CLIMATE_CSV_PATH, SOC_TERRAIN_CLIMATE_OUTPUT_CSV_PATH)
    else:
        print("TERRAIN_CLIMATE_CSV_PATH not set yet, skipping merge step for now.")


def merge_and_save(df_filtered, terrain_climate_csv_path, output_csv_path):
    """
    Merge the downloaded terrain/climate CSV back onto the spectral dataframe.
    """
    terrain_climate_df = pd.read_csv(terrain_climate_csv_path)
    terrain_climate_df = terrain_climate_df[["POINTID", "elevation", "slope", "aspect", "temp_mean", "precip_mean"]].copy()

    merge_df = df_filtered.merge(terrain_climate_df, on="POINTID", how="left")
    merge_df = merge_df[[
        "POINTID", "lon", "lat", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
     "OC", "elevation", "slope", "aspect", "temp_mean", "precip_mean",
    ]]

    merge_df.to_csv(output_csv_path, index=False)
    print(f"Wrote {len(merge_df)} rows to {output_csv_path}")
    return merge_df

# ============================================================================
# peat_and_lab.py  --  Add a binary peat flag and lab-measured P/N/K to the Model 
# ----------------------------------------------------------------------------

def load_peatmap(peatmap_shapefile_path):
    df_peatmap = gpd.read_file(peatmap_shapefile_path)
    print(df_peatmap.crs) #should be EPSG:25832
    return df_peatmap


def flag_peat(df, df_peatmap):
    """
    Join LUCAS points against the peatmap, return a POINTID -> peat (0/1) table.
    """
    # Reproject the points to the shapefile's CRS (EPSG:25832)
    gdf_points = gpd.GeoDataFrame(
        df[["POINTID"]],
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25832")

    joined = gpd.sjoin(gdf_points, df_peatmap[["geometry"]], how="left", predicate="within")
    joined["peat"] = joined["index_right"].notna().astype(int)

    print(joined["peat"].value_counts())  # expect mostly 0s, small number of 1s

    return joined[["POINTID", "peat"]].drop_duplicates(subset="POINTID")


def load_lab_pnk(lucas_csv_path):
    """
    Load lab-measured P, N, K, pH_CaCl2, EC from the raw LUCAS CSV.
    """
    df_lucas = pd.read_csv(lucas_csv_path)
    return df_lucas[["POINTID", "P", "N", "K","pH_CaCl2","EC"]]


def main_peat_and_lab():
    df_peatmap = load_peatmap(PEATMAP_SHAPEFILE_PATH)
    df_rest = pd.read_csv(SOC_TERRAIN_CLIMATE_OUTPUT_CSV_PATH)

    peat_result = flag_peat(df_rest, df_peatmap)
    lab_pnk = load_lab_pnk(LUCAS_CSV_PATH)

    df_soc_peat_climate_terrain = df_rest.merge(peat_result, on="POINTID", how="left")
    df_soc_peat_climate_terrain_lab = df_soc_peat_climate_terrain.merge(lab_pnk, on="POINTID", how="left")

    df_soc_peat_climate_terrain_lab.to_csv(SOC_S2_TERRAIN_CLIMATE_PEAT_OUTPUT_CSV_PATH, index=False)
    print(f"Wrote {len(df_soc_peat_climate_terrain_lab)} rows to {SOC_S2_TERRAIN_CLIMATE_PEAT_OUTPUT_CSV_PATH}") #should be 310 
    return df_soc_peat_climate_terrain_lab

# ============================================================================
# Add legacy soil-property map (clay and available water capacity)
# ----------------------------------------------------------------------------

def _sample_raster(gdf, raster_path, column_name):
    with rasterio.open(raster_path) as src:
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        values = [v[0] for v in src.sample(coords)]
        nodata = src.nodata

    gdf[column_name] = values
    gdf[column_name] = gdf[column_name].replace(nodata, float("nan"))

    n_missing = gdf[column_name].isnull().sum()
    print(f"{column_name}: {n_missing} of {len(gdf)} points are nodata")
    print(gdf[column_name].describe())
    return gdf


def add_legacy_soil_rasters(df):
    """
    Sample clay, AWC rasters at each point's lon/lat.

    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3035")

    gdf = _sample_raster(gdf, CLAY_RASTER_PATH, "clay_map_pct")
    gdf = _sample_raster(gdf, AWC_RASTER_PATH, "AWC_pct")

    return pd.DataFrame(gdf.drop(columns="geometry"))


def main_legacy_soil_rasters():
    df_new = pd.read_csv(SOC_S2_TERRAIN_CLIMATE_PEAT_OUTPUT_CSV_PATH)
    df_new = add_legacy_soil_rasters(df_new)

    df_new.to_csv(WORKING_FILE_OUTPUT_CSV_PATH, index=False)
    print(f"Wrote {len(df_new)} rows to {WORKING_FILE_OUTPUT_CSV_PATH}") #should be 310 samples
    return df_new

