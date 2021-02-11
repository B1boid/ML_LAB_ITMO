def f_rel(x, alpha):
    return x / alpha if x < 0 else x


def d_f_rel(x, alpha):
    return 1 / alpha if x < 0 else 1


def relu(network, n, d, alpha):
    return [[[f_rel(network[i1][i2][i3], alpha) for i3 in range(n)] for i2 in range(n)] for i1 in range(d)]


def d_relu(d_net, network, n, d, alpha):
    return [[[d_net[i1][i2][i3] * d_f_rel(network[i1][i2][i3], alpha)
              for i3 in range(n)] for i2 in range(n)] for i1 in range(d)]


# # TODO faster
# def f_pool(matrix, i2, i3, s):
#     return max([matrix[i][j] for j in range(s * i3, s * (i3 + 1)) for i in range(s * i2, s * (i2 + 1))])
#
#
# def pool(network, n, d, s):
#     print(n,(n + s - 1) // s,d)
#     return [[[f_pool(network[i1], i2, i3, s) for i3 in range((n + s - 1) // s)] for i2 in range((n + s - 1) // s)]
#             for i1 in range(d)]
#
#
# def d_pool(d_net, l_network, network, n, d, s):
#     res = [[[0 for _ in range(len(l_network[0]))] for _ in range(len(l_network[0]))] for _ in range(len(l_network))]
#     for dim in range(d):
#         for i in range(n):
#             for j in range(n):
#                 target = network[dim][i][j]
#                 for r_i in range(s * i, s * (i + 1)):
#                     for r_j in range(s * j, s * (j + 1)):
#                         if l_network[dim][r_i][r_j] == target:
#                             res[dim][r_i][r_j] = d_net[dim][i][j]
#     return res

def pool(network, n, d, s):
    res = [[[0 for _ in range((n + s - 1) // s)] for _ in range((n + s - 1) // s)] for _ in range(d)]
    d_res = [[[0 for _ in range((n + s - 1) // s)] for _ in range((n + s - 1) // s)] for _ in range(d)]
    for dim in range(d):
        for i in range((n + s - 1) // s):
            for j in range((n + s - 1) // s):
                mx, d_mx = None, None
                for i_r in range(i * s, (i + 1) * s):
                    for j_r in range(j * s, (j + 1) * s):
                        if i_r < n and j_r < n:
                            if mx is None or network[dim][i_r][j_r] > mx:
                                mx = network[dim][i_r][j_r]
                                d_mx = [(i_r, j_r)]
                            elif network[dim][i_r][j_r] == mx:
                                d_mx.append((i_r, j_r))
                res[dim][i][j] = mx
                d_res[dim][i][j] = d_mx
    return res, d_res


def d_pool(d_net, d_res, n, d, n2, d2):
    res = [[[0 for _ in range(n2)] for _ in range(n2)] for _ in range(d2)]
    for i1 in range(d):
        for i2 in range(n):
            for i3 in range(n):
                for i_r, j_r in d_res[i1][i2][i3]:
                    res[i1][i_r][j_r] = d_net[i1][i2][i3]
    return res


def bias(network, n, d, b):
    return [[[network[i1][i2][i3] + b[i1] for i3 in range(n)] for i2 in range(n)] for i1 in range(d)]


def d_bias(d_net, n, d):
    return d_net, [sum([d_net[i1][i2][i3] for i3 in range(n) for i2 in range(n)]) for i1 in range(d)]


# TODO faster
def f_cnv(data, d, k, s, a, i1, i2, i3):
    return sum([data[dim][i2 * s + shift_i][i3 * s + shift_j] * a[i1][dim][shift_i][shift_j]
                for shift_j in range(k) for shift_i in range(k) for dim in range(d)])


def cnv(cur_f, network, n, d, h, k, s, p, a):
    r_n = 1 + (2 * p + n - k) // s
    data, d_dat = pre_cnv(cur_f, network, n, d, p)
    return [[[f_cnv(data, d, k, s, a, i1, i2, i3) for i3 in range(r_n)] for i2 in range(r_n)] for i1 in
            range(h)], d_dat, data


def pre_cnv(cur_f, network, n, d, p):
    res = []
    d_res = None
    #print(2 * p + n,n,p,network)
    for dim in range(d):
        tmp1 = []
        d_res = []
        for i in range(2 * p + n):
            tmp2 = []
            d_tmp = []
            for j in range(2 * p + n):
                r_i, r_j = cur_f(i, n, p), cur_f(j, n, p)
                d_tmp.append((r_i, r_j))
                tmp2.append(network[dim][r_i][r_j])
            tmp1.append(tmp2)
            d_res.append(d_tmp)
        res.append(tmp1)
    return res, d_res


def cnve(i, n, p):
    if i < p:
        return 0
    res = i - p
    if res > n - 1:
        res = n - 1
    return res


def cnvm(i, n, p):
    res = i - p
    if res < 0:
        res = -res
    if res > n - 1:
        res = 2 * (n - 1) - res
    return res


def cnvc(i, n, p):
    return (2 * n + i - p) % n


def d_cnv(u_data, d_res, cur_d_net, n, d, h, k, s, p, a):
    r_n = 1 + (2 * p + n - k) // s
    d_net = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(d)]
    d_print = [[[[0 for _ in range(k)] for _ in range(k)] for _ in range(d)] for _ in range(h)]
    for ih in range(h):
        for i in range(r_n):
            for j in range(r_n):
                for shift_i in range(k):
                    for shift_j in range(k):
                        for dim in range(d):
                            d_print[ih][dim][shift_i][shift_j] += \
                                u_data[dim][i * s + shift_i][j * s + shift_j] * cur_d_net[ih][i][j]
                            d_net[dim][d_res[i * s + shift_i][j * s + shift_j][0]][
                                d_res[i * s + shift_i][j * s + shift_j][1]] += \
                                cur_d_net[ih][i][j] * a[ih][dim][shift_i][shift_j]

    return d_net, d_print


def main():
    input_str = list(map(int, input().split()))
    n, d = input_str[0], input_str[1]
    tmp_vec = input_str[2:]
    start_network = [[[tmp_vec[i1 * n * n + i2 * n + i3] for i3 in range(n)] for i2 in range(n)] for i1 in range(d)]
    L = int(input())
    networks = [start_network]
    inpts = [None]
    all_d = [None]
    u_datas = [None]

    for _ in range(L):
        tmp_input = input().split()
        inpts.append(tmp_input)
        key = tmp_input[0]
        cur_network = networks[-1]
        cur_n, cur_d = len(cur_network[0]), len(cur_network)
        d_res, u_data = None, None
        if key == "relu":
            alpha = int(tmp_input[1])
            networks.append(relu(cur_network, cur_n, cur_d, alpha))
        elif key == "pool":
            s = int(tmp_input[1])
            res, d_res = pool(cur_network, cur_n, cur_d, s)
            networks.append(res)
        elif key == "bias":
            b = list(map(int, tmp_input[1:]))
            networks.append(bias(cur_network, cur_n, cur_d, b))
        elif str(key).find("cnv") > -1:
            args = list(map(int, tmp_input[1:]))
            h, k, s, p = args[0], args[1], args[2], args[3]
            a = [[[[args[4:][i4 + i3 * k + i2 * pow(k, 2) + i1 * pow(k, 2) * cur_d]
                    for i4 in range(k)] for i3 in range(k)] for i2 in range(cur_d)] for i1 in range(h)]
            if key == "cnvm":
                res, d_res, u_data = cnv(cnvm, cur_network, cur_n, cur_d, h, k, s, p, a)
            elif key == "cnve":
                res, d_res, u_data = cnv(cnve, cur_network, cur_n, cur_d, h, k, s, p, a)
            elif key == "cnvc":
                res, d_res, u_data = cnv(cnvc, cur_network, cur_n, cur_d, h, k, s, p, a)
            else:
                res = None
                #print("err", 5 / 0)
            networks.append(res)
        all_d.append(d_res)
        u_datas.append(u_data)

    #print(networks)
    last_n, last_d = len(networks[-1][0]), len(networks[-1])
    print(" ".join(map(str, [networks[-1][i1][i2][i3]
                             for i1 in range(last_d)
                             for i2 in range(last_n)
                             for i3 in range(last_n)
                             ])))

    d_tmp = list(map(int, input().split()))
    d_last = [[[d_tmp[i1 * last_n * last_n + i2 * last_n + i3]
                for i3 in range(last_n)] for i2 in range(last_n)] for i1 in range(last_d)]
    # print(d_last)
    # print("--")
    d_net = [d_last]
    d_prints = []
    for ii in range(L):
        ind = L - ii
        key = inpts[ind][0]
        cur_network = networks[ind]
        cur_n, cur_d = len(cur_network[0]), len(cur_network)
        if key == "relu":
            alpha = int(inpts[ind][1])
            d_net.append(d_relu(d_net[-1], networks[ind - 1], cur_n, cur_d, alpha))
            # print(d_relu(d_net[-1], networks[ind - 1], cur_n, cur_d, alpha))
        elif key == "pool":
            #s = int(inpts[ind][1])
            #print(d_pool(d_net[-1], networks[ind - 1], cur_network, cur_n, cur_d, s))
            #d_net.append(d_pool(d_net[-1], networks[ind - 1], cur_network, cur_n, cur_d, s))
            #print(d_pool(d_net[-1], all_d[ind], cur_n, cur_d, len(networks[ind - 1][0]), len(networks[ind - 1])))
            d_net.append(d_pool(d_net[-1], all_d[ind], cur_n, cur_d, len(networks[ind - 1][0]), len(networks[ind - 1])))
        elif key == "bias":
            cur_d_net, d_print = d_bias(d_net[-1], cur_n, cur_d)
            d_net.append(cur_d_net)
            d_prints.append(["bias", d_print, [cur_d]])
            # print("b", d_print)
        elif key.find("cnv") > -1:
            args = list(map(int, inpts[ind][1:]))
            h, k, s, p = args[0], args[1], args[2], args[3]
            a = [[[[args[4:][i4 + i3 * k + i2 * pow(k, 2) + i1 * pow(k, 2) * len(networks[ind - 1])]
                    for i4 in range(k)] for i3 in range(k)] for i2 in range(len(networks[ind - 1]))] for i1 in range(h)]
            cur_d_net, d_print = d_cnv(u_datas[ind], all_d[ind], d_net[-1], len(networks[ind - 1][0]),
                                       len(networks[ind - 1]), h, k, s, p, a)
            d_net.append(cur_d_net)
            # print(cur_d_net)
            d_prints.append(["cnv", d_print, [h, len(networks[ind - 1]), k]])

    last_n, last_d = len(d_net[-1][0]), len(d_net[-1])
    #print(d_net[-1],last_n,last_d)
    print(" ".join(map(str, [d_net[-1][i1][i2][i3]
                             for i1 in range(last_d)
                             for i2 in range(last_n)
                             for i3 in range(last_n)
                             ])))
    # print("---")
    #print(d_net)
    #
    # print("pr---")
    # print(d_prints)
    ll = len(d_prints)
    for i in range(ll):
        dp = d_prints[ll - i - 1]
        # print(dp[1])
        # print(dp[2])
        if dp[0] == "cnv":
            print(" ".join(map(str, [dp[1][i0][i1][i2][i3]
                                     for i0 in range(dp[2][0])
                                     for i1 in range(dp[2][1])
                                     for i2 in range(dp[2][2])
                                     for i3 in range(dp[2][2])
                                     ])))
        elif dp[0] == "bias":
            print(" ".join(map(str, [dp[1][i0]
                                     for i0 in range(dp[2][0])
                                     ])))
        else:
            print("err", 5 / 0)


main()

