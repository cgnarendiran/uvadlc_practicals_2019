import cifar10_utils


batch_size = 100

cifar10 =  cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
x , y = cifar10 ['train'].next_batch(batch_size)

