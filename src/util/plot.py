import matplotlib.pyplot as plt


def plot(train_loss, test_loss):
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.savefig('loss.png')
