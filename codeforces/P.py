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

    xi_2 = n
    for el in m.keys():
        expected_val = tmp_m1[el[0]] * tmp_m2[el[1]] / n
        if expected_val != 0:
            xi_2 += (m[el] - expected_val) ** 2 / expected_val - expected_val

    print(xi_2)


main()
