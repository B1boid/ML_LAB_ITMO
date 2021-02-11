def main():
    k = int(input())
    n = int(input())
    E_yy_x = 0
    tmp_y_x = [0] * k
    cnt_y = [0] * k
    for _ in range(n):
        x, y = map(int, input().split())
        E_yy_x += y * y / n
        tmp_y_x[x - 1] += y / n
        cnt_y[x - 1] += 1 / n

    E_y_x = sum((tmp_y_x[i] ** 2) / cnt_y[i] if cnt_y[i] != 0 else 0 for i in range(k))

    D_y_x = E_yy_x - E_y_x
    print(D_y_x)


main()
