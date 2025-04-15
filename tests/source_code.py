# Python (3.8.1)

from typing import Optional, List  # Use types from typing

class Solution:
    def containsDuplicate(self, numbers: List[int]) -> bool:
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

solution = Solution()
print(Solution().containsDuplicate([1, 2, 3]), False)
print(Solution().containsDuplicate([1, 2, 3, 4]), False)
print(Solution().containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]), True)
print(Solution().containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]), True)