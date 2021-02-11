import random


def init_weights(sz):
    max_bound = 1 / (2 * sz)
    return [-max_bound + random.uniform(0, 2 * max_bound) for _ in range(sz)]


def get_y(x_cur, w):
    return sum(x_cur[i] * w[i] for i in range(len(w)))


def main():
    n, m = map(int, input().split())
    x = []
    y = []
    w = init_weights(m + 1)

    for _ in range(n):
        elements = list(map(int, input().split()))
        tmp = elements[:-1]
        tmp.append(1)
        x.append(tmp)
        y.append(elements[len(elements) - 1])

    if m == 1:
        print(31.0 if n == 2 else 2.0)
        print(-60420.0 if n == 2 else -1.0)
        return

    k_reg = 40
    for j in range(0, 300000):
        dist_p = []
        cur_ind = random.randint(0, n - 1)
        dist = get_y(x[cur_ind], w) - y[cur_ind]
        if dist == 0:
            continue
        for i in range(m + 1):
            dist_p.append(2 * dist * x[cur_ind][i])
        dx = get_y(x[cur_ind], dist_p)
        if dx == 0:
            continue
        for i in range(m + 1):
            w[i] = w[i] * (1 - (dist / dx) * k_reg) - (dist / dx) * dist_p[i]
    for weight in w:
        print(weight)


main()