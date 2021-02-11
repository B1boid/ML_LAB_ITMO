def main():
    n, m, k = map(int, input().split())
    array = input().split()
    print(array)
    result = cross_validation(k, array)
    for arr in result:
        print(len(arr), ' '.join(map(str, arr)))


def cross_validation(k, array):
    result = [[] for _ in range(k)]
    c = [(array[i], i + 1) for i in range(len(array))]
    c.sort()
    ind = 0
    for i in range(len(c)):
        result[ind % k].append(c[i][1])
        ind += 1
    return result


main()