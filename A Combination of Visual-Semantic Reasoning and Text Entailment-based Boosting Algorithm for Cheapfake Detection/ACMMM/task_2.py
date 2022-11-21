import numpy as np
import argparse
import pickle, os
import json
import nltk
import torch
import csv
import spacy
import random

import torch
from torch import nn

from vocab import Vocabulary
from model import VSRN
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../',
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

opt, unknown = parser.parse_known_args()

vocab = pickle.load(open(os.path.join(
    opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
opt.vocab_size = len(vocab)



def process_text(text, vocab):
    tokens = nltk.tokenize.word_tokenize(
            str(text).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)
    targets = torch.zeros(1, len(target)).long()
    end = len(target)
    targets[0, :end] = target[:end]
    lengths = [end]
    return targets, lengths



def image_score_test(images, idx, cap1, test, model, vocab):
    image = images[idx]
    image = torch.Tensor(image).unsqueeze(0)

    doc = nlp(cap1)
    for ent in doc.ents:
        cap1 = cap1.replace(ent.text, ent.label_)

    print(cap1)
    cap1, len1 = process_text(cap1, vocab)

    image = Variable(image)
    cap1 = Variable(cap1)
    image = image.cuda()
    cap1 = cap1.cuda()

    img_emb, GCN_img_emd = model.img_enc(image)
    img_emb = img_emb.data.cpu().numpy().copy()

    cap1_emb = model.txt_enc(cap1, len1)
    cap1_emb = cap1_emb.data.cpu().numpy().copy()
    print(img_emb.shape, cap1_emb.shape)

    score1 = np.dot(img_emb, cap1_emb.T)

    score1 = score1[0][0]

    mean = 0

    range_img = list(range(0, idx)) + list(range(idx + 1, 100))

    for i in range(10):
        rand = random.choice(range_img)
        img = images[rand]
        img = torch.Tensor(img).unsqueeze(0)
        img = Variable(img)
        img = img.cuda()
        img, GCN_img_emd = model.img_enc(img)
        img = img.data.cpu().numpy().copy()

        rand = random.choice(range_img)

        cap = test[rand]['caption']

        doc = nlp(cap)
        for ent in doc.ents:
            cap = cap.replace(ent.text, ent.label_)
        # print(cap +'\n')


        cap, len = process_text(cap, vocab)
        cap = Variable(cap)
        cap = cap.cuda()
        cap_emb = model.txt_enc(cap, len)
        cap_emb = cap_emb.data.cpu().numpy().copy()

        score = np.dot(img, cap1_emb.T)
        score = score[0][0]/20
        mean += score

        score = np.dot(img_emb, cap_emb.T)
        score = score[0][0]/20
        mean += score

    return score1 - mean


nlp = spacy.load('en_core_web_sm')

model = VSRN(opt)
model.load_state_dict(torch.load('runs/runX/model_best.pth.tar', map_location='cpu')['model'])
model.val_start()

f = open('task_2.json')
test = json.load(f)

images = np.load(opt.data_path + 'Data_Npy/task_2.npy')

with open('task_2.csv', 'w') as f:
    writer = csv.writer(f)
    for i in range(100):
        score = image_score_test(images, i, test[i]['caption'], test, model, vocab)
        input = [test[i]['img_local_path'], score]

        print(test[i]['img_local_path'])

        writer.writerow(input)

# score = image_score_test(images, 0, test[0]['caption'], test, model, vocab)