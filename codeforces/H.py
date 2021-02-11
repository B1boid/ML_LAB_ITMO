import math


def gen_args(m, m_2):
    args = []
    for i in range(0, m_2):
        cur_arg = []
        num = i
        for _ in range(m):
            cur_arg.append(num % 2)
            num = num // 2
        args.append(cur_arg)
    return args


def last_str(p, sz):
    print(' '.join([str(p) for _ in range(sz)]), -0.5)


def main():
    m = int(input())
    m_2 = int(math.pow(2, m))
    args_f = gen_args(m, m_2)
    neurons = []

    for i in range(m_2):
        val = int(input())
        if len(neurons) == 512:
            continue
        if val == 0:
            continue
        neuron = [0.5, []]
        for j in range(m):
            if args_f[i][j] == 0:
                neuron[1].append("-1.0")
                continue
            neuron[1].append("1.0")
            neuron[0] -= 1
        neurons.append(neuron)

    cnt = len(neurons)
    if cnt == 0:
        print(1)
        print(1)
        cnt = m
    else:
        print(2)
        print(len(neurons),1)

    for neuron in neurons:
        print(' '.join(neuron[1]),neuron[0])

    last_str(1, cnt)


main()
