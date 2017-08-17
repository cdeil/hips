# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import remote_data
from numpy.testing import assert_allclose
from ..fetch import HipsTileFetcher
from ..survey import HipsSurveyProperties

class TestHipsTileFetcher:
    @classmethod
    def setup_class(cls):
        url = 'http://alasky.unistra.fr/DSS/DSS2Merged/properties'
        hips_survey = HipsSurveyProperties.fetch(url)
        cls.fetcher = HipsTileFetcher(tile_indices=[69623, 69627, 69628, 69629, 69630, 69631], hips_order=7,
                                      hips_survey=hips_survey, tile_format='fits', progress_bar=True)

    @remote_data
    def test_tiles(self):
        tiles = self.fetcher.tiles
        assert_allclose(tiles[0].data[0][5:10], [1871, 1921, 2064, 2215, 1810])

    @remote_data
    def test_tiles_aiohttp(self):
        tiles = self.fetcher.tiles_aiohttp
        assert_allclose(tiles[0].data[0][5:10], [1871, 1921, 2064, 2215, 1810])
