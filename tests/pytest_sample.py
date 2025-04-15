# run with: ~/$ pytest pytest_sample.py

import pytest
from source_code import Solution


class Solution:
    def containsDuplicate(self, numbers: list[int]) -> bool:
        """
        Time complexity: O(n)
        Auxiliary space complexity: O(n)
        """
        unique_numbers = set()

        for number in numbers:
            if number in unique_numbers:
                return True  # early exit
            else:
                unique_numbers.add(number)

        return False


@pytest.fixture
def solution():
    return Solution()

# print(Solution().containsDuplicate([1, 2, 3]), False)
# print(Solution().containsDuplicate([1, 2, 3, 4]), False)
# print(Solution().containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]), True)

test_data = [
    pytest.param([1, 2, 3], False, id="no_duplicates"),
    pytest.param([1, 2, 3, 4], False, id="longer_no_duplicates"),
    pytest.param([1, 1, 1, 3, 3, 4, 3, 2, 4, 2],
                 True, id="multiple_duplicates"),
    pytest.param([], False, id="empty_list"),
    pytest.param([1], False, id="single_element"),
    pytest.param([1, 1], False, id="minimal_duplicates")
]

# Test data with test IDs.
test_data = [
    ([1, 2, 3], False),
    ([1, 2, 3, 4], False),
    ([1, 1, 1, 3, 3, 4, 3, 2, 4, 2], True),
    ([], False),  # Edge case
    ([1], False),  # Single element
    ([1, 1], True)  # Minimal duplicates
]


@pytest.mark.parametrize("numbers, expected_output", test_data)
def test_contains_duplicate(solution, numbers, expected_output):
    actual_output = solution.containsDuplicate(numbers)
    assert actual_output == expected_output, f"Failed for {numbers}. Expected {expected_output}, got {actual_output}"
