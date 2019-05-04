import numpy as np
import matplotlib.pyplot as plt 

plt.style.use('seaborn-darkgrid')


train_loss = np.load('results/loss_train.npy')

train_acc = np.load('results/accuracy_train.npy')

x = np.arange(start=0, stop=len(train_loss) * 50, step=50)

plt.figure(0)
plt.plot(x, train_acc)
# plt.hlines(0.46, xmin=0, xmax=(FLAGS.max_steps / FLAGS.eval_freq), label='Accuracy Goal: 0.46')
# plt.legend(loc=4)
plt.title('Accuracy of the LSTM (Default Parameters)')
plt.xlabel('Number of Evaluations (Eval Freq = ' + str(50) + ')')
plt.ylabel('Accuracy')
plt.savefig('LSTM_Part2_acc.png')

plt.figure(1)
plt.plot(x, train_loss)
# plt.hlines(0.46, xmin=0, xmax=(FLAGS.max_steps / FLAGS.eval_freq), label='Accuracy Goal: 0.46')
# plt.legend(loc=4)
plt.title('Loss of the LSTM (Default Parameters)')
plt.xlabel('Number of Evaluations (Eval Freq = ' + str(50) + ')')
plt.ylabel('Loss')
plt.savefig('LSTM_Part2_loss.png')