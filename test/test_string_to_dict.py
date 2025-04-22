import sys
import os

root_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'
    )
)
sys.path.append(root_dir)
from src.utils.data_util import post_processing
from src.utils.constant import TOKEN_MAP
from pprint import pprint
import unittest

class TestTokenize(unittest.TestCase):
    def test_case_1(self):
        input_data = "<chart_type>vbar</chart_type><plot_bb><x0> 65</x0><y0>41</y0><x2>464</x2><y2>347</y2></plot_bb><data_series><x>2017</x><y>74.2%</y><color>lapis</color><x>2018</x><y>73%</y><color>lapis</color><x>2017</x><y>25.8%</y><color>vampire_black</color><x>2018</x><y>27%</y><color>vampire_black</color></data_series><text_display><polygon><x0>157</x0><y0>356</y0><x2>374</x2><y2>367</y2></polygon><text>2017,2018</text><role>x_axis</role><colors>None</colors><polygon><x0>25</x0><y0>150</y0><x2>374</x2><y2>367</y2></polygon><text>Share of internet sales</text><role>y_title</role><colors>None</colors><polygon><x0>41</x0><y0>27</y0><x2>59</x2><y2>366</y2></polygon><text>0,1,2,3,4,5</text><role>y_axis</role><colors>None</colors><polygon><x0>160</x0><y0>382</y0><x2>325</x2><y2>413</y2></polygon><text>North America Rest of the world</text><role>legend</role><colors>lapis,vampire_black</colors></text_display>"

        output = {
            "chart_type": "vbar",
            "plot_bb": {
                "x0": '65',
                "y0": '41',
                "x2": '464',
                "y2": '347'
            },
            "data_series": [
                {"x": '2017', "y": "74.2%", "color": "lapis"},
                {"x": '2018', "y": "73%", "color": "lapis"},
                {"x": '2017', "y": "25.8%", "color": "vampire_black"},
                {"x": '2018', "y": "27%", "color": "vampire_black"}
            ],
            "text_display": [
                {
                    "polygon": {"x0": '157', "y0": '356', "x2": '374', "y2": '367'},
                    "text": ["2017","2018"],
                    "role": "x_axis",
                    "colors": 'None'
                },
                {
                    "polygon": {"x0": '25', "y0": '150', "x2": '374', "y2": '367'},
                    "text": "Share of internet sales",
                    "role": "y_title",
                    "colors": 'None'
                },
                {
                    "polygon": {"x0": '41', "y0": '27', "x2": '59', "y2": '366'},
                    "text": ["0", "1", "2", "3", "4", "5"],
                    "role": "y_axis",
                    "colors": 'None'
                },
                {
                    "polygon": {"x0": '160', "y0": '382', "x2": '325', "y2": '413'},
                    "text": "North America Rest of the world",
                    "role": "legend",
                    "colors": ["lapis", "vampire_black"]
                }
            ]
        }
        result = post_processing(input_data,TOKEN_MAP)
        pprint(result)
        self.assertDictEqual(
            result, output
        )
if __name__ == '__main__':
    unittest.main()
