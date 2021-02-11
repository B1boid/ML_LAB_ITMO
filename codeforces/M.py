def main():
    n = int(input())
    xs, ys = [], []
    for i in range(n):
        x, y = map(int, input().split())
        xs.append((x, i))
        ys.append((y, i))
    tmp_xs, tmp_ys = sorted(xs), sorted(ys)
    rank_xs, rank_ys = [0] * n, [0] * n
    for i in range(n):
        rank_xs[tmp_xs[i][1]] = i
        rank_ys[tmp_ys[i][1]] = i

    d = sum((rank_xs[i] - rank_ys[i]) ** 2 for i in range(n))
    k_spirman = 1 - 6 * d / (n ** 3 - n)
    print(k_spirman)


main()
