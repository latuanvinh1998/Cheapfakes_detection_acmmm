import difflib
import random
import torch
import nltk
import numpy as np
from WK_Sbert.utils_sbert import *
from torch.autograd import Variable
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Module


choice = [' this is fake.', ' this claim is false.', ' this claim is not genuine.', 'this is not truth.', 'thus claim is false.']

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

def take_cap(train, captions, idx):
    label = random.randrange(2)
    cap_1 = train[idx]["articles"][0]['caption_modified'].replace('\n','')

    find = 0
    for j in range(1, len(train[idx]["articles"])):
        cap = train[idx]["articles"][j]['caption_modified'].replace('\n','')
        if difflib.SequenceMatcher(None, cap_1, cap).ratio() < 0.5:
            cap_2 = cap
            find = 1
            break

    if find == 0:
        cap_2 = captions[idx]

    if label == 1:
        cap_1 += random.choice(choice)

    return cap_1, cap_2, label

def ssim_sbert(sentences, model, tokenizer, device):

    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > 128:
            sent_ids = sent_ids[: 128]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = 128 - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
        # Length Check
        assert len(sent_ids) == 128
        assert len(sent_mask) == 128

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)

    features_mask = np.array(features_mask)

    batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
    batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
    model.zero_grad()

    with torch.no_grad():
        features = model(**inputs)[1]

    all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

    embed_method = generate_embedding("ave_last_hidden", features_mask)
    embedding = embed_method.embed("ave_last_hidden", all_layer_embedding)

    similarity = (embedding[0].dot(embedding[1])
        / np.linalg.norm(embedding[0])
        / np.linalg.norm(embedding[1]))

    return similarity


def image_score(images, idx, idx_, cap1, cap2, train, model, vocab):
    image = images[idx]
    image = torch.Tensor(image).unsqueeze(0)
    cap1, len1 = process_text(cap1, vocab)
    cap2, len2 = process_text(cap1, vocab)

    image = Variable(image)
    cap1 = Variable(cap1)
    cap2 = Variable(cap2)
    image = image.cuda()
    cap1 = cap1.cuda()
    cap2 = cap2.cuda()

    img_emb, GCN_img_emd = model.img_enc(image)
    img_emb = img_emb.data.cpu().numpy().copy()

    cap1_emb = model.txt_enc(cap1, len1)
    cap1_emb = cap1_emb.data.cpu().numpy().copy()

    cap2_emb = model.txt_enc(cap2, len1)
    cap2_emb = cap2_emb.data.cpu().numpy().copy()

    score1 = np.dot(img_emb, cap1_emb.T)
    score2 = np.dot(img_emb, cap2_emb.T)

    score1 = score1[0][0]
    score2 = score2[0][0]

    mean = 0

    range_img = list(range(0, idx)) + list(range(idx + 1, 10000))
    range_cap = list(range(0, idx_)) + list(range(idx_, 161754))

    for i in range(10):
        rand = random.choice(range_img)
        img = images[rand]
        img = torch.Tensor(img).unsqueeze(0)
        img = Variable(img)
        img = img.cuda()
        img, GCN_img_emd = model.img_enc(img)
        img = img.data.cpu().numpy().copy()

        rand = random.choice(range_cap)
        cap = train[rand]["articles"][0]['caption_modified'].replace('\n','')
        cap, len = process_text(cap, vocab)
        cap = Variable(cap)
        cap = cap.cuda()
        cap_emb = model.txt_enc(cap, len)
        cap_emb = cap_emb.data.cpu().numpy().copy()

        score = np.dot(img, cap1_emb.T)
        score = score[0][0]/30
        mean += score

        score = np.dot(img, cap2_emb.T)
        score = score[0][0]/30
        mean += score

        score = np.dot(img_emb, cap_emb.T)
        score = score[0][0]/30
        mean += score

    return score1 - mean, score2 - mean

def image_score_test(images, idx, cap1, cap2, test, model, vocab):
    image = images[idx]
    image = torch.Tensor(image).unsqueeze(0)
    cap1, len1 = process_text(cap1, vocab)
    cap2, len2 = process_text(cap1, vocab)

    image = Variable(image)
    cap1 = Variable(cap1)
    cap2 = Variable(cap2)
    image = image.cuda()
    cap1 = cap1.cuda()
    cap2 = cap2.cuda()

    img_emb, GCN_img_emd = model.img_enc(image)
    img_emb = img_emb.data.cpu().numpy().copy()

    cap1_emb = model.txt_enc(cap1, len1)
    cap1_emb = cap1_emb.data.cpu().numpy().copy()

    cap2_emb = model.txt_enc(cap2, len1)
    cap2_emb = cap2_emb.data.cpu().numpy().copy()

    score1 = np.dot(img_emb, cap1_emb.T)
    score2 = np.dot(img_emb, cap2_emb.T)

    score1 = score1[0][0]
    score2 = score2[0][0]

    mean = 0

    range_img = list(range(0, idx)) + list(range(idx + 1, 1700))

    for i in range(10):
        rand = random.choice(range_img)
        img = images[rand]
        img = torch.Tensor(img).unsqueeze(0)
        img = Variable(img)
        img = img.cuda()
        img, GCN_img_emd = model.img_enc(img)
        img = img.data.cpu().numpy().copy()

        rand = random.choice(range_img)
        rand_ = random.randrange(2)
        if rand_ == 0:
            cap = test[rand]['caption1_modified']
        else:
            cap = test[rand]['caption2_modified']
        cap, len = process_text(cap, vocab)
        cap = Variable(cap)
        cap = cap.cuda()
        cap_emb = model.txt_enc(cap, len)
        cap_emb = cap_emb.data.cpu().numpy().copy()

        score = np.dot(img, cap1_emb.T)
        score = score[0][0]/30
        mean += score

        score = np.dot(img, cap2_emb.T)
        score = score[0][0]/30
        mean += score

        score = np.dot(img_emb, cap_emb.T)
        score = score[0][0]/30
        mean += score

    return score1 - mean, score2 - mean

# class Neural(Module):
#     def __init__(self, in_features):
#         super(Neural, self).__init__()
#         self.nn_1 = Linear(in_features, 16)
#         self.nn_2 = Linear(16, 2)
#         self.prelu = PReLU(16)

#     def forward(self, input_):
#         emb = self.nn_1(input_)
#         emb = self.prelu(emb)
        
#         out = self.nn_2(emb)
#         return out

num_features = 8
class Neural(Module):
    def __init__(self, in_features):
        super(Neural, self).__init__()
        self.nn_1 = Linear(2, num_features)
        self.nn_2 = Linear(1, num_features)
        self.nn_3 = Linear(3, num_features)
        # self.prelu = PReLU(64)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=num_features, nhead=2, batch_first=True), num_layers=2)
        self.out = Linear(num_features, 2)

    def forward(self, input_):
        emb_1 = self.nn_1(input_[:, 0:2].unsqueeze(1))
        emb_2 = self.nn_2(input_[:, 2:3].unsqueeze(1))
        emb_3 = self.nn_3(input_[:, 3:6].unsqueeze(1))
        emb = torch.cat((emb_1, emb_2, emb_3,), 1)

        emb = self.transformer(emb)
        emb = emb.mean(dim = 1)

        out = self.out(emb)

        return out