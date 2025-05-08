import unittest
from ..calculator import ReturnRiskIndexCalculator
from ..data_parser import parse_input_to_series
from ..config import INPUT_STR

class TestReturnRiskIndexCalculator(unittest.TestCase):
    def setUp(self):
        self.series = parse_input_to_series(INPUT_STR)
        self.calculator = ReturnRiskIndexCalculator(self.series)

    def test_batch_calculate_returns(self):
        returns = self.calculator.batch_calculate_returns()
        self.assertTrue(len(returns) > 0)

    def test_annualized_return(self):
        df = self.calculator.annualized_return()
        self.assertIn('d_curr', df.columns)

if __name__ == '__main__':
    unittest.main()