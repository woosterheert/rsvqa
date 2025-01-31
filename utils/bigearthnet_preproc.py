import os
import tqdm
import tqdm
import rasterio
from rasterio.warp import reproject, Resampling

def reduce_filenames(ben_dir):
    for tile in tqdm.tqdm(os.listdir(ben_dir)):
        if os.path.isdir(os.path.join(ben_dir, tile)):
            for patch in os.listdir(os.path.join(ben_dir, tile)):
                if os.path.isdir(os.path.join(ben_dir, tile, patch)):
                    for band in os.listdir(os.path.join(ben_dir, tile, patch)):
                        os.rename(os.path.join(ben_dir, tile, patch, band), 
                                  os.path.join(ben_dir, tile, patch, band[-7:]))
                    os.rename(os.path.join(ben_dir, tile, patch), 
                              os.path.join(ben_dir, tile, patch[-5:]))
                    
class BigEarthNetPreProcessing:
    def __init__(self, dir_data_in, dir_data_out, boi):
        self.dir_data_in = dir_data_in 
        self.dir_data_out = dir_data_out
        self.boi = boi
        self.tile_names = [tile for tile in os.listdir(self.dir_data_in) 
                           if os.path.isdir(os.path.join(self.dir_data_in, tile))]

    def process_image(self, tile_name, patch_name):
        image_folder = os.path.join(self.dir_data_in, tile_name, patch_name)
        bands = self.collect_bands(image_folder)
        metadata = self.collect_metadata(bands[0])
        self.update_metadata(metadata)
        self.create_image(tile_name, patch_name, bands, metadata)
    
    def collect_bands(self, image_folder):
        bands = []
        for band in self.boi:
            bands.append(os.path.join(image_folder, band + '.tif'))
        return bands

    def collect_metadata(self, image_file):
        with rasterio.open(image_file) as src:
            meta = src.meta
        return meta

    def update_metadata(self, metadata):
        metadata.update(count = len(self.boi))
        metadata.update(dtype = 'uint8')

    def create_image(self, tile_name, patch_name, bands, metadata):
        if not os.path.exists(os.path.join(self.dir_data_out, tile_name)):
            os.makedirs(os.path.join(self.dir_data_out, tile_name))
        with rasterio.open(os.path.join(self.dir_data_out, tile_name, patch_name + '.tif'),
                           'w', **metadata) as dst:
            for id, layer in enumerate(bands, start=1):
                with rasterio.open(layer) as src:
                  tmp = src.read(1) / 4096 * 255 #(4096 = 2**12)
                  reproject(
                            source=tmp,
                            destination=rasterio.band(dst, id),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=src.transform,
                            dst_crs=src.crs,
                            resampling=Resampling.nearest)

    def process_tile(self, tile_name):
        patch_names = [patch for patch in os.listdir(os.path.join(self.dir_data_in, tile_name)) 
                       if os.path.isdir(os.path.join(self.dir_data_in, tile_name, patch))]
        for patch_name in patch_names:
            self.process_image(tile_name, patch_name)

    def process_tiles(self):
        for tile in tqdm.tqdm(self.tile_names):
            if not os.path.exists(os.path.join(self.dir_data_out, tile)):
                self.process_tile(tile)

if __name__ == "__main__" :
    reduce_filenames('BigEarthNet-S2')
    preproc = BigEarthNetPreProcessing('BigEarthNet-S2', '6d_data', ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'])
    preproc.process_tiles()
    preproc = BigEarthNetPreProcessing('BigEarthNet-S2', 'rgb_data', ['B04', 'B03', 'B02'])
    preproc.process_tiles()