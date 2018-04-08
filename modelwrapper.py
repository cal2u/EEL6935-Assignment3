import datetime
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys

class ModelWrapper():
    def __init__(self, model, args):
        self.model = model
        self.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if args.snapshot is not None:
            print('\nLoading model from {}...'.format(args.snapshot))
            self._loadfrom(args.snapshot)

        if args.cuda:
            torch.cuda.set_device(args.device)
            self.model.cuda()

    def train(self, dataset, dev_data=None, args=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        steps = 0
        best_acc = 0
        last_step = 0
        self.model.train()
        for epoch in range(1, args.epochs+1):
            for batch in dataset:
                feature, target = batch
                feature, target = Variable(feature), Variable(target)
                if args.cuda:
                    feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                logit = self.model(feature)

                loss = F.cross_entropy(logit, target)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % args.log_interval == 0:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                    accuracy = 100.0 * corrects/args.batch_size
                    sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                 loss.data[0],
                                                                                 accuracy,
                                                                                 corrects,
                                                                                 args.batch_size))
                if steps % args.test_interval == 0:
                    dev_acc = self.eval(dev_data, args)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        last_step = steps
                        if args.save_best:
                            self._save(args.save_dir, 'best', steps)
                    else:
                        if steps - last_step >= args.early_stop:
                            print('early stop by {} steps.'.format(args.early_stop))
                elif steps % args.save_interval == 0:
                    self._save(args.save_dir, 'snapshot', steps)

    def eval(self, dataset, args):
        self.model.eval()
        corrects, avg_loss = 0, 0
        for batch in dataset:
            feature, target = batch
            feature, target = Variable(feature), Variable(target)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = self.model(feature)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.data[0]
            corrects += (torch.max(logit, 1)
                         [1].view(target.size()).data == target.data).sum()

        size = len(dataset) * args.batch_size
        avg_loss /= size
        accuracy = 100.0 * corrects/size
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size))
        return accuracy

    def predict(self, feature, args):
        self.model.eval()

        feature = Variable(feature)
        if args.cuda:
            feature = feature.cuda()

        value, index = torch.max(self.model(feature)[0], 0)
        return index


    def _save(self, save_dir, save_prefix, steps):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
        torch.save(self.model.state_dict(), save_path)

    def _loadfrom(self, snapshot):
        self.model.load_state_dict(torch.load(snapshot))
