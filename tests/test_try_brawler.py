from pathlib import Path

import cv2
import pytest

from brawl_stars_gym.try_brawler import TryBrawler


@pytest.mark.parametrize(
    "image_filename, expected_damage_per_second",
    [
        ("region_1615323288.933967.png", 1104),
        ("region_1615323287.451309.png", 1104),
        ("region_1615323285.9551656.png", 690),
        ("region_1615323282.927077.png", 345),
        ("region_1615323281.4278924.png", 0),
        ("region_1615323279.9353352.png", 0),
        ("region_1615323278.4275265.png", 0),
        ("region_1615323276.8998601.png", 0),
        ("region_1615323275.4353292.png", 345),
        ("region_1615323273.9144456.png", 345),
        ("region_1615323272.4369917.png", 0),
        ("region_1615323270.9134169.png", 1188),
        ("region_1615323269.429769.png", 1215),
        ("region_1615323267.9092262.png", 1564),
        ("region_1615323266.4272156.png", 2415),
        ("region_1615323264.9300942.png", 345),
        ("region_1615323263.4112701.png", 790),
        ("region_1615323261.890822.png", 1473),
        ("region_1615323260.4179654.png", 2176),
        ("region_1615323258.9119284.png", 1847),
        ("region_1615323257.4330688.png", 2727),
        ("region_1615323255.900718.png", 2008),
        ("region_1615323254.388059.png", 2321),
        ("region_1615323252.9088092.png", 1571),
        ("region_1615323251.4210038.png", 803),
        ("region_1615323249.912249.png", 948),
        ("region_1615323248.3961856.png", 1436),
        ("region_1615323246.9156165.png", 1573),
        ("region_1615323245.4008174.png", 995),
        ("region_1615323243.9200182.png", 1327),
        ("region_1615323242.4288852.png", 854),
        ("region_1615323240.921151.png", 1405),
        ("region_1615323239.4166396.png", 1533),
        ("region_1615323237.9028618.png", 1035),
        ("region_1615323236.4090006.png", 0),
        ("region_1615323234.914815.png", 0),
    ],
)
def test_damage_per_second(image_filename, expected_damage_per_second):
    image_filepath = Path(__file__).parent / "data" / image_filename
    image = cv2.imread(str(image_filepath))

    assert TryBrawler.damage_per_second(image) == expected_damage_per_second
