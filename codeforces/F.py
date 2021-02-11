def main():
    cnst_min, cnst_max = 1e-200, 1e+200
    k = int(input())
    lambdas = list(map(int, input().split()))
    class_count = [0 for _ in range(k)]
    alpha = int(input())
    n_train = int(input())
    words = {}
    for _ in range(n_train):
        tmp = list(input().split())
        class_count[int(tmp[0]) - 1] += 1
        tmp_Dict = {}
        for word in tmp[2:]:
            if word.strip() not in words:
                words[word.strip()] = [0 for _ in range(k)]
            if word.strip() not in tmp_Dict:
                words[word.strip()][int(tmp[0]) - 1] += 1
            tmp_Dict[word.strip()] = 1

    matrix = {}
    for word in words:
        tmp_word_classes = []
        for cur_cl in range(k):
            tmp_word_classes.append((words[word][cur_cl] + alpha) / (class_count[cur_cl] + 2 * alpha))
        matrix[word] = tmp_word_classes

    n_test = int(input())
    for _ in range(n_test):
        tmp = list(input().split())
        tmp_Dict = {}
        for word in tmp[1:]:
            tmp_Dict[word.strip()] = 1

        sum_p_x_cl = 0
        all_p_x_cl = []
        for cur_cl in range(k):
            p_x_cl = class_count[cur_cl] / n_train
            for word in words:
                p_x_cl *= matrix[word][cur_cl] if (word in tmp_Dict) else (1 - matrix[word][cur_cl])
            p_x_cl *= lambdas[cur_cl]
            sum_p_x_cl += p_x_cl
            all_p_x_cl.append(p_x_cl)
        if sum_p_x_cl != 0:
            print(' '.join([str(round((all_p_x_cl[i]) / sum_p_x_cl, 10)) for i in range(k)]))
            continue

        all_p_x_cl = []
        cnt = []

        for i in range(k):
            p_x_cl = class_count[i] / n_train
            p_x_cl *= lambdas[i]
            cnt.append(0)
            for word in words:
                p_x_cl *= matrix[word][i] if (word in tmp_Dict) else (1 - matrix[word][i])
                if p_x_cl < cnst_min:
                    p_x_cl *= cnst_max
                    cnt[i] += 1
            all_p_x_cl.append(p_x_cl)

        max_cnt = max(cnt)
        for i in range(k):
            while max_cnt != cnt[i]:
                cnt[i] += 1
                all_p_x_cl[i] *= cnst_max

        sum_p_x_cl = sum(all_p_x_cl)

        print(' '.join([str((all_p_x_cl[i]) / sum_p_x_cl) for i in range(k)]))


main()
