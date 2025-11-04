import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr

os.system('cls' if os.name == 'nt' else 'clear')

data = ['REST', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'TWMEMORY']

char_head = '/data2/home/data_for_research/sub_net_divide/Primary Visual/'
Identification_1 = np.zeros((8, 8))

for i in range(8):
    char = data[i]
    char_new = os.path.join(char_head, char)
    train = [f for f in os.listdir(char_new) if f not in ('.', '..')]
    for j in range(8):
        if i == j:
            continue
        else:
            iden = []
            char_1 = data[j]
            char_new_1 = os.path.join(char_head, char_1)
            test = [f for f in os.listdir(char_new_1) if f not in ('.', '..')]
            for k in range(len(train)):
                str_train = train[k]
                str_sub = str_train[5:11]
                for l in range(len(test)):
                    test_file = test[l]
                    a = test_file.find(str_sub)
                    if a == 5:
                        train_path = os.path.join(char_new, train[k])
                        train_data = loadmat(train_path)
                        R_train = train_data['S1']
                        test_path = os.path.join(char_new_1, test_file)
                        test_data = loadmat(test_path)
                        E_test = test_data['S1']
                        flat_R = R_train.flatten()
                        flat_E = E_test.flatten()
                        pei_dui, _ = pearsonr(flat_R, flat_E)
                        all_correlation = []
                        for m in range(len(test)):
                            all_test_path = os.path.join(char_new_1, test[m])
                            all_test_data = loadmat(all_test_path)
                            E_test_all = all_test_data['S1']
                            flat_E_all = E_test_all.flatten()
                            r, _ = pearsonr(flat_R, flat_E_all)
                            all_correlation.append(r)
                        if max(all_correlation) == pei_dui:
                            iden.append(1)
                        else:
                            iden.append(0)
                        break
            if len(iden) > 0:
                Identification_1[i, j] = np.sum(iden) / len(iden)
            else:
                Identification_1[i, j] = 0

np.savetxt('/data2/home/HCP_research/Primary Visual.csv', Identification_1, delimiter=',')
print('1.Primary Visual finished')
