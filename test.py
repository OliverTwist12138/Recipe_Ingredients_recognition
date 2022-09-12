import numpy as np
with open('./model/training_curves_google.txt', 'r') as f:
    loss_cache_train = f.readline()
    # acc_cache_train = f.readline()
    # loss_cache_val = f.readline()
    # acc_cache_val = f.readline()

cache = loss_cache_train[1:-1].split('][')
loss_cache_train = cache[0].split(',')
acc_cache_train = cache[1].split(',')
loss_cache_val = cache[2].split(',')
acc_cache_val = cache[3].split(',')

loss_cache_train = [float(i) for i in loss_cache_train]
acc_cache_train = [float(i) for i in acc_cache_train]
loss_cache_val = [float(i) for i in loss_cache_val]
acc_cache_val = [float(i) for i in acc_cache_val]
print(acc_cache_train,'\n', len(acc_cache_train))

# import matplotlib.pyplot as plt
#
# _, axes = plt.subplots(1, 2, figsize=(20, 5))
# train_scores_mean = np.mean(acc_cache_train)
# train_scores_std = np.std(acc_cache_train)
# test_scores_mean = np.mean(acc_cache_val)
# test_scores_std = np.std(acc_cache_val)
num_epochs = 9
import matplotlib.pyplot as plt

plt.style.use('seaborn')

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 7))
plt.grid()
plt.subplot(1, 2, 1)

plt.plot(range(1, num_epochs + 1), np.array(loss_cache_train),  label='train', linewidth=2)
plt.plot(range(1, num_epochs + 1), np.array(loss_cache_val), label='val', linewidth=2)
plt.xlabel('$Epochs$', size=15)
plt.ylabel('$Loss$', size=15)
plt.legend(loc='best', fontsize=15)

plot2 = plt.subplot(1, 2, 2)
plot2.plot(range(1, num_epochs + 1), np.array(acc_cache_train),  label='train', linewidth=2)
plot2.plot(range(1, num_epochs + 1), np.array(acc_cache_val),  label='val', linewidth=2)
plot2.set_xlabel('$Epochs$', size=15)
plot2.set_ylabel('$Acc$', size=15)
plot2.legend(loc='best', fontsize=15)
plot2.grid(True)

plt.show()
