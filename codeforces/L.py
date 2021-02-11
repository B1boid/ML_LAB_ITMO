import math


def main():
    n = int(input())
    sum_x, sum_y = 0, 0
    data = []
    for _ in range(n):
        x, y = map(int, input().split())
        sum_x += x
        sum_y += y
        data.append((x, y))

    x_m = sum_x / n
    y_m = sum_y / n

    sum_numenator, sum_denominator_1, sum_denominator_2 = 0, 0, 0
    for (x, y) in data:
        sum_numenator += (x - x_m) * (y - y_m)
        sum_denominator_1 += pow((x - x_m), 2)
        sum_denominator_2 += pow((y - y_m), 2)
    if sum_denominator_1 == 0 or sum_denominator_2 == 0:
        k_pirson = 0
    else:
        k_pirson = sum_numenator/math.sqrt(sum_denominator_1*sum_denominator_2)
    print(k_pirson)


main()
