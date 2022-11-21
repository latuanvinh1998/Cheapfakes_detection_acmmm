from vocab import Vocabulary
from model import VSRN
import numpy as np
import argparse
import json
import nltk


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

model = VSRN(opt)
model.load_state_dict(torch.load('/media/vinh/Not_mix3/Research/ACMMM/pretrain_model/coco/model_coco_1.pth.tar')['model'])

images = np.load('/media/vinh/Not_mix3/Research/ACMMM/Data_Npy/Test/test.npy')

f = open('../Data/mmsys_anns/public_test_mmsys_final.json')
labels = []

# f_val = open('../Data/mmsys_anns/val_data.json')
# validation = []

for line in f:
	labels.append(json.loads(line))

text = labels[0]['caption1'].strip()

tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
caption = []
caption.append(vocab('<start>'))
caption.extend([vocab(token) for token in tokens])
caption.append(vocab('<end>'))
target = torch.Tensor(caption)

targets = torch.zeros(len(target), 350).long()
end = len(target)
targets[i, :end] = cap[:end]