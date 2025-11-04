import numpy as np

def sub_iden(identification_matrix, sub_num, state_num):
    Identification_accuracy = np.empty((state_num, state_num), dtype=object)

    for i in range(state_num):
        for j in range(state_num):
            if i == j:
                Identification_accuracy[i, j] = 1
                continue
            else:
                iden = []
                for k in range(sub_num * i, sub_num * (i + 1)):
                    R_train = identification_matrix[:, k]
                    all_correlation = []
                    for l in range(sub_num * j, sub_num * (j + 1)):
                        E_test_all = identification_matrix[:, l]
                        r = np.corrcoef(R_train, E_test_all)
                        all_correlation.append(r[0, 1])

                    # 计算 R_train 和对应的 E_test_all 的相关性
                    r_1 = np.corrcoef(R_train, identification_matrix[:, k + (sub_num * (j - i))])
                    pei_dui = r_1[0, 1]

                    if max(all_correlation) == pei_dui:
                        iden.append(1)
                    else:
                        iden.append(0)

                Identification_accuracy[i, j] = np.sum(iden) / len(iden)

    return Identification_accuracy
