import torch
import numpy as np
import nltk
# def processing_json(text):
# 	caption = 

def get_data_loader(img_npy, idxes, i, f_json, vocab, opt):
	images = []
	targets = []
	lengths = []
	indexes = []
	caption_labels = []
	caption_masks = []
	tokens_list = []

	for idx in idxes:
		idx_ = idx + i*10000
		caption = []
		img = torch.Tensor(img_npy[idx,:,:])
		cap = f_json[idx_]["articles"][0]['caption_modified']
		tokens = nltk.tokenize.word_tokenize(str(cap.strip()).lower())
	
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))

		target = torch.Tensor(caption)

		mask = np.zeros(opt.max_len + 1)
		gts = np.zeros((opt.max_len + 1))

		# print(tokens)
		cap_caption = ['<start>'] + tokens + ['<end>']
		# print(cap_caption)
		if len(cap_caption) > opt.max_len - 1:
			cap_caption = cap_caption[:opt.max_len]
			cap_caption[-1] = '<end>'
			
		for j, w in enumerate(cap_caption):
			gts[j] = vocab(w)

		non_zero = (gts == 0).nonzero()


		mask[:int(non_zero[0][0]) + 1] = 1




		caption_label = torch.from_numpy(gts).type(torch.LongTensor)
		caption_mask = torch.from_numpy(mask).type(torch.FloatTensor)

		images.append(img)
		lengths.append(len(target))
		tokens_list.append(target)
		indexes.append(idx_)
		caption_labels.append(caption_label)
		caption_masks.append(caption_mask)

	max_length = max(lengths)
	batch_targets = torch.zeros(len(tokens_list), max(lengths)).long()

	for i, cap in enumerate(tokens_list):
		end = lengths[i]
		batch_targets[i, :end] = cap[:end]

	batch_images = torch.stack([img for img in images])
	# batch_targets = torch.stack([target for target in targets])
	batch_caption_labels = torch.stack([caption_label for caption_label in caption_labels])
	batch_caption_masks = torch.stack([caption_mask for caption_mask in caption_masks])

	return batch_images, batch_targets, lengths, indexes, batch_caption_labels, batch_caption_masks