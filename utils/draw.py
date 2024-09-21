import matplotlib
import matplotlib.pyplot as plt


def list_to_draw(listinfo, filename, title, xlabel, ylabel, figsize=(10, 7)):
    plt.figure(figsize=figsize)
    plt.plot(listinfo, color='green', label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    # plt.show()

if __name__ == '__main__':
    acc = [56.24, 67.64, 86.75, 89.12, 90.30, 96.36]
    list_to_draw(acc, filename='../chart/acc_{}.png'.format(name), title='acc', xlabel='epochs', ylabel='acc_num')

