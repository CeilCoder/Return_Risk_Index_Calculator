import unittest
from ..calculator import ReturnRiskIndexCalculator
from ..data_parser import parse_input_to_series
from ..config import INPUT_STR

class TestReturnRiskIndexCalculator(unittest.TestCase):
    def setUp(self):
        self.series = parse_input_to_series(INPUT_STR)
        self.calculator = ReturnRiskIndexCalculator(self.series)

    def test_annualized_return(self):
        df = self.calculator.annualized_return()
        self.assertIn('d_curr', df.columns)

    def test_count_valuation(self):
        df = self.calculator.count_valuation()
        self.assertTrue(len(df) > 0)

if __name__ == '__main__':
    unittest.main()