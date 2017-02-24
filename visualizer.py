import matplotlib.pyplot as plt
def visualize_two_hidden(nn):
    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax1.set_title('Input Weights', y=1.09)
    plt.axis('off')
    ax1.matshow(nn.nnwi)

    ax2 = fig.add_subplot(132)
    ax2.set_title('Hidden Weights', y=1.09)
    plt.axis('off')
    ax2.matshow(nn.nnwh)

    ax3 = fig.add_subplot(133)
    ax3.set_title('Output Weights', y=1.09)
    ax3.matshow(nn.nnwo)

    plt.axis('off')
    plt.show()
