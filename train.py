import torch
from torch import nn
from tqdm import tqdm

from dataset import load_data_db
from config import *
from utils import *

def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    # print(pred)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, net_name, train_iter, test_iter, loss, trainer, num_epochs,
               devices=try_all_gpus()):
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in tqdm(range(num_epochs), desc="train epoch"):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
            f'{   metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
            f'{str(devices)}')
    plt.title(f'{net_name}')
    plt.savefig(os.path.join(LOG_DIR, f'{net_name}.png'))
    plt.show()
    torch.save(net.state_dict(), os.path.join(WEIGHTS_DIR, f'{net_name}.pt'))


def get_net(net_name, vocab):

    if net_name == "BiRNN":
        embed_size, num_hiddens, num_layers = EMBED_SIZE, NUM_HIDDEN, NUM_LAYERS
        net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.LSTM:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])
        net.apply(init_weights)
        return net

    elif net_name == "TextCNN":
        embed_size, kernel_sizes, nums_channels = EMBED_SIZE, KERNEL_SIZES, NUMS_CHANNELS 
        net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
        def init_weights(m):
            if type(m) in (nn.Linear, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
        return net

        

if __name__ == '__main__':
        
    batch_size = BATCH_SIZE
    train_iter, test_iter, vocab = load_data_db(batch_size) 



    print("\n\n"+"="*100+"\n"+"="*100+"\n\n")

    devices = DEVICES
    net_name = NET_NAME
    net = get_net(net_name, vocab)
    lr, num_epochs = LR, NUM_EPOCHS
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")

    print(f"batch_size: \t\t{batch_size}")
    print(f"lr: \t\t\t{lr}")
    print(f"num_epochs: \t\t{num_epochs}")
    print(net) 
    print("\n\n"+"="*100+"\n"+"="*100+"\n\n")



    print(f"training {net_name} on {devices}...")
    train(net, net_name, train_iter, test_iter, loss, trainer, num_epochs, devices)
