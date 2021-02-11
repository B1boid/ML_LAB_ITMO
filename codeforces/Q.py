import math


def main():
    k1, k2 = map(int, input().split())
    n = int(input())
    m = {}
    tmp_m1, tmp_m2 = [0 for _ in range(k1)], [0 for _ in range(k2)]
    for _ in range(n):
        x1, x2 = map(int, input().split())
        if (x1 - 1, x2 - 1) in m:
            m[(x1 - 1, x2 - 1)] += 1
        else:
            m[(x1 - 1, x2 - 1)] = 1
        tmp_m1[x1 - 1] += 1
        tmp_m2[x2 - 1] += 1

    h = 0
    _log = math.log
    for el in m.keys():
        h += m[el] / n * _log(m[el] / (tmp_m1[el[0]]))

    print(-h)


main()
