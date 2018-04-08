import argparse
import torch
import pandas as pd
from cnn import CNN_1D
from modelwrapper import ModelWrapper
from dataset import getTrainDev, GenomicsDataset
from torch.utils.data import DataLoader


def do_train(cnn, args):
    train_data, dev_data = getTrainDev('train.csv')
    train_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                         num_workers=4)
    dev_data = DataLoader(dev_data, batch_size=args.batch_size, num_workers=4)

    print()
    try:
      cnn.train(train_data, dev_data=dev_data, args=args)
    except KeyboardInterrupt:
      print('\n' + '-' * 89)
      print('Exiting from training early')


def do_pred(cnn, args):
    df = pd.read_csv('test.csv')
    df['prediction'] = 0
    ix = 0
    for feature in GenomicsDataset(df, labels=False):
        feature = feature.unsqueeze(0)
        label = cnn.predict(feature, args)
        df.iloc[ix, df.columns.get_loc('prediction')] = int(label)
        ix += 1
    df[['id','prediction']].to_csv('output.csv', index=False)


def do_test(cnn, args):
    df = pd.read_csv('train.csv')
    dataset = GenomicsDataset(df, labels=True)
    dataset = DataLoader(dataset, batch_size=args.batch_size,
                         num_workers=4)
    cnn.eval(dataset, args)


def main():
    parser = argparse.ArgumentParser(description='Kaggle Genomics Classifier')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('-batch-size', type=int, default=10, help='batch size')
    parser.add_argument('-log-interval', type=int, default=1, help='logging rate')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', action='store_true', default=False, help='predict the results for the dataset')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

    args = parser.parse_args()

    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]
    args.num_classes = 2
    cnn = ModelWrapper(CNN_1D(args), args)

    if args.predict is not False:
        do_pred(cnn, args)
    elif args.test:
        do_test(cnn, args)
    else:
        do_train(cnn, args)


if __name__ == "__main__":
    main()
