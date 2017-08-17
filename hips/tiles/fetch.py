# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import urllib.request
import concurrent.futures
from typing import Generator
from ..tiles import HipsSurveyProperties, HipsTile, HipsTileMeta

__all__ = [
    'HipsTileFetcher',
]


class HipsTileFetcher:
    """Fetch a list of HiPS tiles.

    Parameters
    ----------
    tile_indices : np.ndarray
        List of index values for HiPS tiles
    hips_order : int
        Order of the HiPS survey
    hips_survey : `~hips.HipsSurveyProperties`
        HiPS survey properties
    tile_format : {'fits', 'jpg', 'png'}
        Format of HiPS tile
    progress_bar : bool
        Show a progress bar for tile fetching and drawing
    thread_count : int
        Number of threads to use for fetching HiPS tiles
    """

    def __init__(self, tile_indices: np.ndarray, hips_order: int, hips_survey: HipsSurveyProperties,
                 tile_format: str, progress_bar: bool, thread_count: int = 10) -> None:
        self.tile_indices = tile_indices
        self.hips_order = hips_order
        self.hips_survey = hips_survey
        self.tile_format = tile_format
        self.progress_bar = progress_bar
        self.thread_count = thread_count

    def fetch_tile_threaded(self, url: str, timeout: int = 10) -> Generator:
        """Fetch a HiPS tile asynchronously."""
        with urllib.request.urlopen(url, timeout=timeout) as conn:
            return conn.read()

    @property
    def tiles(self) -> np.ndarray:
        """Generator function to fetch HiPS tiles from a remote URL."""
        tile_urls, tile_metas = [], []
        for healpix_pixel_index in self.tile_indices:
            tile_meta = HipsTileMeta(
                order=self.hips_order,
                ipix=healpix_pixel_index,
                frame=self.hips_survey.astropy_frame,
                file_format=self.tile_format,
            )
            tile_urls.append(self.hips_survey.tile_url(tile_meta))
            tile_metas.append(tile_meta)

        raw_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            future_to_url = {executor.submit(self.fetch_tile_threaded, url): url for url in tile_urls}
            if self.progress_bar:
                from tqdm import tqdm
                requests = tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url), desc='Fetching tiles')
            else:
                requests = concurrent.futures.as_completed(future_to_url)

            for future in requests:
                url = future_to_url[future]
                raw_responses.append(future.result())

        tiles = []
        for idx, raw_data in enumerate(raw_responses):
            tiles.append(HipsTile(tile_metas[idx], raw_data))
        return tiles

    async def fetch_tile_threaded_aiohttp(self, url: str, session) -> Generator:
        """Fetch a HiPS tile asynchronously using aiohttp."""
        async with session.get(url) as response:
            return await response.read()

    @property
    async def fetch_tiles_aiohttp(self) -> np.ndarray:
        """Generator function to fetch HiPS tiles from a remote URL using aiohttp."""
        import aiohttp, asyncio
        tile_urls, tile_metas = [], []
        for healpix_pixel_index in self.tile_indices:
            tile_meta = HipsTileMeta(
                order=self.hips_order,
                ipix=healpix_pixel_index,
                frame=self.hips_survey.astropy_frame,
                file_format=self.tile_format,
            )
            tile_urls.append(self.hips_survey.tile_url(tile_meta))
            tile_metas.append(tile_meta)

        tasks = []
        async with aiohttp.ClientSession() as session:
            for idx, url in enumerate(tile_urls):
                task = asyncio.ensure_future(self.fetch_tile_threaded_aiohttp(url.format(idx), session))
                tasks.append(task)

            if self.progress_bar:
                from tqdm import tqdm
                raw_responses = []
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Fetching tiles'):
                    raw_responses.append(await f)
            else:
                raw_responses = await asyncio.gather(*tasks)

        tiles = []
        for idx, raw_data in enumerate(raw_responses):
            tiles.append(HipsTile(tile_metas[idx], raw_data))
        return tiles

    @property
    def tiles_aiohttp(self) -> np.ndarray:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.fetch_tiles_aiohttp)
