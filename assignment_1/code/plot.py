import numpy as np
import matplotlib.pyplot as plt 

# train_loss = np.load('../np_results/np_loss_train.npy')
# test_loss = np.load('../np_results/np_loss_test.npy')

# train_acc = np.load('../np_results/np_accuracy_train.npy')
# test_acc = np.load('../np_results/np_accuracy_test.npy')

# plt.plot(train_loss)
# plt.plot(test_loss)
# plt.title('Loss curves for numpy-MLP: default parameters')
# plt.xlabel('Number of epochs x100')
# plt.ylabel('Loss on the mini-batch')
# plt.legend(['train loss', 'test loss'], loc='upper right')
# plt.show()

# plt.plot(train_acc)
# plt.plot(test_acc)
# plt.title('Accuracy curves for numpy-MLP: default parameters')
# plt.xlabel('Number of epochs x100')
# plt.ylabel('Accuracy on the mini-batch')
# plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
# plt.show()

train_loss = np.load('../torch_results/torch_loss_train.npy')
test_loss = np.load('../torch_results/torch_loss_test.npy')

train_acc = np.load('../torch_results/torch_accuracy_train.npy')
test_acc = np.load('../torch_results/torch_accuracy_test.npy')

plt.plot(train_loss)
plt.plot(test_loss)
plt.title('Loss curves for pytorch-MLP: default parameters')
plt.xlabel('Number of epochs x100')
plt.ylabel('Loss on the mini-batch')
plt.legend(['train loss', 'test loss'], loc='upper right')
plt.show()

plt.plot(train_acc)
plt.plot(test_acc)
plt.title('Accuracy curves for pytorch-MLP: default parameters')
plt.xlabel('Number of epochs x100')
plt.ylabel('Accuracy on the mini-batch')
plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
plt.show()