# 1 Say "Hello, World!" With Python
print("Hello, World!")



# 2 Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
    if (n % 2 == 1) or 6 <= n <= 20:
        print("Weird")
    else:
        print("Not Weird")



# 3 Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a + b, end="\n")
    print(a - b, end="\n")
    print(a * b)



# 4 Divison
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print('{} \n{}'.format((a//b), (a/b)))



# Loops
if __name__ == '__main__':
    n = int(input())
    print(*[i**2 for i in range(n)], sep="\n")



# Write a function
def is_leap(year):
    leap = False
    if year % 400 == 0:
        leap = True
    elif year % 4 == 0 and not year % 100 == 0:
        leap = True
    return leap
year = int(input())
print(is_leap(year))


def is_leap(year):
    return year % 4 == 0 and (year % 400 == 0 or year % 100 != 0)

year = int(input())
print(is_leap(year))



# Print Function
if __name__ == '__main__':
    n = int(input())
    print(*[i for i in range(1, n + 1)], sep="")

if __name__ == '__main__':
    n = int(input())
    print(*list(range(1, n + 1)), sep="")



# List Comprehensions
if __name__ == '__main__':
    x, y, z, n = (int(input()) for _ in range(4))
    print([[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n])



# Find the Runner-Up Score!
if __name__ == '__main__':
    # n = int(input()) # na co to?
    arr = map(int, input().split())
    arr = list(arr)
    # max_count = arr.count(max(arr))
    [arr.remove(max(arr)) for i in range(arr.count(max(arr)))]
    print(max(arr))

if __name__ == '__main__':
    # n = int(input())
    arr = map(int, input().split())
    arr = sorted(list(set(arr)))[-2]
    print(arr)



# Nested Lists
if __name__ == '__main__':
    nasc = {}
    nasc2 = {}
    names = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nasc[name] = score
    for key, val in nasc.items():
        if val != min(nasc.values()):
            nasc2[key] = val
    for key, val in nasc2.items():
        if val == min(nasc2.values()):
            names.append(key)
    for name in sorted(names):
        print(name, end="\n")



