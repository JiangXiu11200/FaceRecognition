import unittest

from tests_core.test_calculation import TestCalculation
from tests_core.test_coordinate_detection import TesttestCoordinateDetection
from tests_core.test_predictor import TestPredictor

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
