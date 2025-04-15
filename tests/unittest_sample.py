# run with: ~/$ py unittest_sample.py
# validation_test_cases

import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """
    Suppress the print statements form source code.
    """
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

import unittest
from test_cases import test_cases
with suppress_stdout():
    from source_code import Solution

# class TestContainsDuplicate(unittest.TestCase):
#     def setUp(self):
#         self.solution = Solution()
# 
#     def test_1(self):
#         self.assertFalse(self.solution.containsDuplicate([1, 2, 3]))
# 
#     def test_2(self):
#         self.assertFalse(self.solution.containsDuplicate([1, 2, 3, 4]))
# 
#     def test_3(self):
#         self.assertTrue(self.solution.containsDuplicate(
#             [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))

# print(Solution().containsDuplicate([1, 2, 3]), False)
# print(Solution().containsDuplicate([1, 2, 3, 4]), False)
# print(Solution().containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]), True)

class TestCase(unittest.TestCase):
    def setUp(self):
        self.solution = Solution()  # operations
        # self.test_cases = [
        #     ([1, 2, 3], False),
        #     ([1, 2, 3, 4], False),
        #     ([1, 1, 1, 3, 3, 4, 3, 2, 4, 2], True)
        # ]
        self.test_cases = test_cases

    def test_case(self):
        for arguments, expected_output in self.test_cases:
            with self.subTest(arguments=arguments, expected_output=expected_output):
                self.assertEqual(self.solution.containsDuplicate(arguments), expected_output)


# if __name__ == "__main__":
unittest.main()

