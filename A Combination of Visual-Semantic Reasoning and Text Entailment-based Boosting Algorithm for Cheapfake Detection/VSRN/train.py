import pickle, os, time
import shutil, json
import random
import logging
import argparse
import torch
from utils import get_data_loader

from data import PrecompDataset, collate_fn
from vocab import Vocabulary  # NOQA
from model import VSRN
from torch.autograd import Variable
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import tensorboard_logger as tb_logger
import numpy as np


def train(opt, json_train, json_val, vocab, model, epoch, best_rsum):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()

    rand = np.arange(17).tolist()
    random.shuffle(rand)

    for i_s in rand:
        # if opt.reset_train:
            # Always reset to train mode, this is not the default behavior

        # dset = PrecompDataset('/media/vinh/Not_mix2/Research/ACMMM/Data_Npy/Train', json_train, vocab, i_s, opt)
        dset = PrecompDataset(opt.data_path + 'Data_Npy/Train/train_', json_train, vocab, 16, opt)
        train_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

        for i, train_data in enumerate(train_loader):

            model.train_start()

            data_time.update(time.time() - end)

            model.logger = train_logger

            model.train_emb(*train_data)

            batch_time.update(time.time() - end)
            end = time.time()

            if model.Eiters % opt.log_step == 0:
                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, e_log=str(model.logger)))

            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)
            # if model.Eiters % opt.val_step == 0:
            #     # validate(opt, val_loader, model)

            #     # evaluate on validation set
            #     rsum = validate(opt, json_val, vocab, model)

            #     # remember best R@ sum and save checkpoint
            #     is_best = rsum > best_rsum
            #     best_rsum = max(rsum, best_rsum)
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'model': model.state_dict(),
            #         'best_rsum': best_rsum,
            #         'opt': opt,
            #         'Eiters': model.Eiters,
            #     }, is_best, prefix=opt.logger_name + '/')



    return best_rsum

def validate(opt, json_val, vocab, model):
    # compute the encoding for all the validation images and captions

    currscore = r1 = r5 = r10 = medr = meanr = r1i = r5i = r10i = medri = meanri = 0

    # for i in range(4):
    #     dset = PrecompDataset(opt.data_path + 'Data_Npy/Val/val_', json_val, vocab, i, opt)
    #     val_loader = torch.utils.data.DataLoader(dataset=dset,
    #                                           batch_size=opt.batch_size,
    #                                           shuffle=False,
    #                                           pin_memory=True,
    #                                           collate_fn=collate_fn)

    #     img_embs, cap_embs = encode_data(
    #         model, val_loader, opt.log_step, logging.info)

    #     (r1_, r5_, r10_, medr_, meanr_) = i2t(img_embs, cap_embs, measure=opt.measure)

    #     (r1i_, r5i_, r10i_, medri_, meanri_) = t2i(
    #         img_embs, cap_embs, measure=opt.measure)
    #     r1 += r1_
    #     r5 += r5_
    #     r10 += r10_
    #     medr += medr_
    #     meanr += meanr_

    #     r1i += r1i_
    #     r5i += r5i_
    #     r10i += r10i_
    #     medri += medri_
    #     meanri += meanri_

    #     currscore += r1_ + r5_ + r1i_ + r5i_ 

    # logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
    #              (r1/4, r5/4, r10/4, medr/4, meanr/4))
    # logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
    #              (r1i/4, r5i/4, r10i/4, medri/4, meanr/4))


    # tb_logger.log_value('r1', r1/4, step=model.Eiters)
    # tb_logger.log_value('r5', r5/4, step=model.Eiters)
    # tb_logger.log_value('r10', r10/4, step=model.Eiters)
    # tb_logger.log_value('medr', medr/4, step=model.Eiters)
    # tb_logger.log_value('meanr', meanr/4, step=model.Eiters)
    # tb_logger.log_value('r1i', r1i/4, step=model.Eiters)
    # tb_logger.log_value('r5i', r5i/4, step=model.Eiters)
    # tb_logger.log_value('r10i', r10i/4, step=model.Eiters)
    # tb_logger.log_value('medri', medri/4, step=model.Eiters)
    # tb_logger.log_value('meanr', meanr/4, step=model.Eiters)
    # tb_logger.log_value('rsum', currscore/4, step=model.Eiters)

    dset = PrecompDataset(opt.data_path + 'Data_Npy/Val/val_', json_val, vocab, 4, opt)
    val_loader = torch.utils.data.DataLoader(dataset=dset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          collate_fn=collate_fn)

    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)

    (r1_, r5_, r10_, medr_, meanr_) = i2t(img_embs, cap_embs, measure=opt.measure)

    (r1i_, r5i_, r10i_, medri_, meanri_) = t2i(
        img_embs, cap_embs, measure=opt.measure)
    r1 += r1_
    r5 += r5_
    r10 += r10_
    medr += medr_
    meanr += meanr_

    r1i += r1i_
    r5i += r5i_
    r10i += r10i_
    medri += medri_
    meanri += meanri_

    currscore += r1_ + r5_ + r1i_ + r5i_ 

    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1/4, r5/4, r10/4, medr/4, meanr/4))
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i/4, r5i/4, r10i/4, medri/4, meanr/4))


    tb_logger.log_value('r1', r1/4, step=model.Eiters)
    tb_logger.log_value('r5', r5/4, step=model.Eiters)
    tb_logger.log_value('r10', r10/4, step=model.Eiters)
    tb_logger.log_value('medr', medr/4, step=model.Eiters)
    tb_logger.log_value('meanr', meanr/4, step=model.Eiters)
    tb_logger.log_value('r1i', r1i/4, step=model.Eiters)
    tb_logger.log_value('r5i', r5i/4, step=model.Eiters)
    tb_logger.log_value('r10i', r10i/4, step=model.Eiters)
    tb_logger.log_value('medri', medri/4, step=model.Eiters)
    tb_logger.log_value('meanr', meanr/4, step=model.Eiters)
    tb_logger.log_value('rsum', currscore/4, step=model.Eiters)

    return currscore

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/media/vinh/Not_mix3/Research/ACMMM/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco_precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=2048, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=10000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')

    ###caption parameters
    parser.add_argument(
        '--dim_vid',
        type=int,
        default=2048,
        help='dim of features of video frames')
    parser.add_argument(
        '--dim_hidden',
        type=int,
        default=512,
        help='size of the rnn hidden layer')
    parser.add_argument(
        "--bidirectional",
        type=int,
        default=0,
        help="0 for disable, 1 for enable. encoder/decoder bidirectional.")
    parser.add_argument(
        '--input_dropout_p',
        type=float,
        default=0.2,
        help='strength of dropout in the Language Model RNN')
    parser.add_argument(
        '--rnn_type', type=str, default='gru', help='lstm or gru')

    parser.add_argument(
        '--rnn_dropout_p',
        type=float,
        default=0.5,
        help='strength of dropout in the Language Model RNN')

    parser.add_argument(
        '--dim_word',
        type=int,
        default=300,  # 512
        help='the encoding size of each token in the vocabulary, and the video.'
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=350,
        help='max length of captions(containing <sos>,<eos>)')

    opt = parser.parse_args()

    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    # dset = PrecompDataset(opt.data_path, 'dev', vocab, opt)

    # data_loader = torch.utils.data.DataLoader(dataset=dset,
    #                                         batch_size=8,
    #                                         shuffle=True,
    #                                         pin_memory=True,
    #                                         collate_fn=collate_fn)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    model = VSRN(opt)
    model.load_state_dict(torch.load('/media/vinh/Not_mix3/Research/ACMMM/pretrain_model/coco/model_coco_1.pth.tar')['model'])
    best_rsum = 0

    f_train = open(opt.data_path + 'cosmos_anns/train_data.json')
    labels = []

    f_val = open(opt.data_path + 'cosmos_anns/val_data.json')
    validation = []

    for line in f_train:
        labels.append(json.loads(line))

    for line in f_val:
        validation.append(json.loads(line))

    for epoch in range(opt.num_epochs):

        adjust_learning_rate(opt, model.optimizer, epoch)

        # best_rsum = train(opt, labels, validation, vocab, model, epoch, best_rsum)

        rsum = validate(opt, validation, vocab, model)
        print(rsum)
        raise Exception

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


if __name__ == '__main__':
    main()
