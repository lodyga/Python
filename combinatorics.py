# [3, 2, 1]
# permutations with repetition
# (1, 1, 1, 1)                       .
# (1, 1, 2)              /           |             \
# (1, 2, 1)             3            2              1
# (1, 3)              /           /     \       /   |   \
# (2, 1, 1)          1           2       1     1    2    3
# (2, 2)                               /      / \  /
# (3, 1)                              1     1   2  1
#                                          /
#                                        1     
#                      .
#          /           |             \
#         3            2              1
#       /           /     \       /   |   \
#      4           4       3     2    3    4
#                        /      / \  /
#                       4     3   4  4
#                            /
#                          4     

# [3, 2, 1]
# combinations with repetition
# (1, 1, 1, 1)                      .
# (1, 1, 2)              /          |              \
# (1, 3)               3            2              1
# (2, 2)              /           /    \          /
#                    1          1       2        1
#                             /                 / 
#                            1                 1 
#                                             /
#                                            1
#                     .
#          /          |              \
#        3            2              1
#       /           /    \          /
#      4          3       4        2
#               /                 / 
#              4                 3 
#                               /
#                              4


# permutations with repetition with constraint 
# O(n^(target/min(numbers))), O(target/min(numbers))
"""
permutations with repetition
O(n^k)
"""
class Solution:
    def permutationsWithRepetition(self, numbers: list[int], target: int) -> list[list[int]]:
        # numbers.sort(reverse=True)
        permutation = []
        permutation_list = []

        def dfs(target):
            if target < 0:
                return
            elif target == 0:
                permutation_list.append(permutation.copy())
                return

            for number in numbers:
                permutation.append(number)
                dfs(target - number)
                permutation.pop()

        dfs(target)
        return permutation_list

print(Solution().permutationsWithRepetition([1, 2, 3], 4))



# combinations with repetition with constraint 
# O((n+target-1)!/((n-1)!target!)), O(target/min(numbers))
"""
combinations with repetition
O((n+k-1)!/((n-1)!k!))
"""
class Solution:
    def combinationsWithRepetition(self, numbers: list[int], target: int) -> list[list[int]]:
        # numbers.sort(reverse=True)
        combination = []
        combination_list = []

        def dfs(target, start):
            if target < 0:
                return
            elif target == 0:
                combination_list.append(combination.copy())
                return

            for index in range(start, len(numbers)):
                number = numbers[index]
                combination.append(number)
                dfs(target - number, index)  # Only use numbers[index:] to prevent reordering
                combination.pop()

        dfs(target, 0)
        return combination_list

print(Solution().combinationsWithRepetition([1, 2, 3], 4))


# permutations with repetition with constraint 
# O(n2), O(n)
# dp, bottom-up
class Solution:
    def countPermutationsWithRepetition(self, numbers: list[int], target: int) -> list[list[int]]:
        cache = [0] * (target + 1)
        cache[0] = 1

        for index in range(1, len(cache)):
            for number in numbers:
                if index - number >= 0:
                    cache[index] += cache[index - number]
        
        return cache
# [0, 1, 2, 3, 4]  # index
# [1, 1, 2, 4, 7]  # number of ways to get to the target (index)


print(Solution().countPermutationsWithRepetition([1, 2, 3], 4), 7)
print(Solution().countPermutationsWithRepetition([9], 3), 0)
print(Solution().countPermutationsWithRepetition([2, 3], 7), 3)
print(Solution().countPermutationsWithRepetition([4, 2, 1], 32), 39882198)


# combinations with repetition with constraint 
# O(n2), O(n)
# dp, bottom-up
class Solution:
    def countCombinationsWithRepetition(self, numbers: list[int], target: int) -> list[list[int]]:
        cache = [0] * (target + 1)
        cache[0] = 1

        for number in numbers:
            for index in range(number, len(cache)):
                cache[index] += cache[index - number]
        
        return cache

# [0, 1, 2, 3, 4]  # index
# [1, 1, 2, 3, 4]  # number of ways to get to the target (index)

print(Solution().countCombinationsWithRepetition([1, 2, 3], 4), 4)
print(Solution().countCombinationsWithRepetition([9], 3), 0)
print(Solution().countCombinationsWithRepetition([2, 3], 7), 1)
print(Solution().countCombinationsWithRepetition([4, 2, 1], 32), 81)







# combinations without repetition
# O(n!/((n-k)!k!))
