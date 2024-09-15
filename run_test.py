import unittest

from tests.test_calculation import TestCalculation
from tests.test_predictor import TestPredictor
from tests.test_coordinate_detection import TesttestCoordinateDetection

if __name__ == "__main__":
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    suite_test = unittest.TestSuite()
    suite_test.addTests(
        [
            loader.loadTestsFromTestCase(TestCalculation),
            loader.loadTestsFromTestCase(TestPredictor),
            loader.loadTestsFromTestCase(TesttestCoordinateDetection),
        ]
    )
    test_result = runner.run(suite_test)
