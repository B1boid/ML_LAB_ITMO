def dist(clust):
    clust.sort()
    sm = 0
    last_res = 0
    if len(clust) == 1:
        return 0
    for i in range(1, len(clust)):
        cur_res = last_res + (clust[i] - clust[i - 1]) * i
        last_res = cur_res
        sm += cur_res
    return sm


def main():
    k = int(input())
    n = int(input())
    classes = [[] for _ in range(k)]
    all = []
    for _ in range(n):
        x, y = map(int, input().split())
        classes[y - 1].append(x)
        all.append(x)

    sum_in = 0
    for cl in classes:
        sum_in += dist(cl)
    sum_in *= 2
    print(sum_in)
    sum_out = 2*dist(all)
    sum_out -= sum_in
    print(sum_out)


main()
