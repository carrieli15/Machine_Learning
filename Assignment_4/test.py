import math
import os
import random
import re
import sys


def subsetA(arr):
    # Write your code here
    arr.sort()
    n = len(arr)
    sum_a = 0
    sum_b = 0
    for i in range(n):
        if i % 2 == 0:
            sum_a += arr[i]
        else:
            sum_b += arr[i]
    return sum_a, sum_b


if __name__ == "__main__":
    fptr = open("output.txt", "w")

    arr_count = int(input().strip())

    arr = []

    for _ in range(arr_count):
        arr_item = int(input().strip())
        arr.append(arr_item)

    result = subsetA(arr)

    fptr.write("\n".join(map(str, result)))
    fptr.write("\n")

    fptr.close()
