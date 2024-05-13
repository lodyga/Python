class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums) - 1

        while True:
            if r - l < 2:
                if nums[l] == target:
                    return l
                elif nums[r] == target:
                    return r
                else:
                    return -1

            mid = (l + r)//2

            if nums[mid] == target:
                return mid
            elif nums[mid] < nums[r]:
                if target < nums[mid]:
                    r = mid
                else:
                    l = mid
            elif nums[mid] > nums[r]:
                if target < nums[r]:
                    l = mid
                else:
                    r = mid

        return None


(Solution().search([3, 5, 1], 1), 2)
(Solution().search([4, 5, 6, 7, 0, 1, 2], 0), 4)
(Solution().search([4, 5, 6, 7, 0, 1, 2], 3), -1)
(Solution().search([1], 0), -1)


(Solution().search([1, 2, 3, 4]), 1)
(Solution().search([4, 1, 2, 3]), 1)
(Solution().search([2, 3, 4, 1]), 1)
(Solution().search([3, 4, 1, 2]), 1)

(Solution().search([4, 5, 1, 2, 3]), 1)
(Solution().search([5, 1, 2, 3, 4]), 1)
(Solution().search([1, 2, 3, 4, 5]), 1)

(Solution().search([2, 3, 4, 5, 1]), 1)
(Solution().search([3, 4, 5, 1, 2]), 1)

(Solution().search([4, 5, 6, 7, 0, 1, 2]), 0)
(Solution().search([11, 13, 15, 17]), 11)
(Solution().search([1]), 1)
(Solution().search([3, 1, 2]), 1)
