from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_train_valid_loader
import random
import argparse
import math

import densenet

from tensorboardX import SummaryWriter

def _calc_precision(y, y_pred, label):#返回tp和fp
	gt = y[y_pred==label]#预测为label的相应的y，返回true positive个数和被预测为positive的个数
	return torch.sum(gt==label), len(gt)-torch.sum(gt==label)

def  _calc_recall(y, y_pred, label):#返回tp和fn
	pred = y_pred[y==label]#实际为label的相应的预测值，返回true positive个数和实际为positive的个数
	return torch.sum(pred==label), len(pred)-torch.sum(pred==label)



# sample data to balance data size of different classses
def _sample_db(input_db, random_seed=None, classes=(0,1)):
	background_db = list(filter(lambda item:item['class']==0, input_db))
	abnormal_db = list(filter(lambda item:item['class']==1, input_db))
	if random_seed!=None:
		random.seed(random_seed)
	background_db_sampled = random.sample(background_db, len(abnormal_db))
	balance_db = background_db_sampled + abnormal_db
	random.shuffle(balance_db)
	return balance_db

# generate data loader
# for the inbalence of data size, negtive data is sampled to be equal to postive data 


# train net
def adjust_learning_rate(optimizer, decay=0.1):
	for param_group in optimizer.param_groups:
		param_group['lr'] = decay * param_group['lr']

def train(criterion, num_train, epoch, writer, net, dataloader, optimizer, batch_size, lr_decay_period=100, check_point_dir='./cell/checkpoints', gpu=True):

	train_total = num_train
	net.cuda()
	net.train()

	print('''
	Starting training:
		Epochs: {}
		Batch size: {}
		Checkpoints: {}
		CUDA: {}
	'''.format(epoch, batch_size, check_point_dir, str(gpu)))

	epoch_loss = 0
	epoch_err = 0
	epoch_tp, epoch_fp, epoch_fn = 0, 0, 0

	if epoch % lr_decay_period == 0:
		adjust_learning_rate(optimizer)

	for index_batch, (img, label) in enumerate(dataloader):
		if gpu:
			X = Variable(img).cuda()
			y = Variable(label).cuda()
		else:
			X = Variable(img)
			y = Variable(label)

		predict = net(X)
		loss = criterion(predict, y)
		epoch_loss += loss.data[0]


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		y_predict = F.softmax(predict)
		y_predict = torch.max(y_predict.data, dim=1)[1]
		epoch_err += torch.sum(y_predict!=y.data)
		batch_err = 1.*torch.sum(y_predict!=y.data)/len(y_predict)
		tp, fp = _calc_precision(y.data, y_predict, 1)
		tp, fn = _calc_recall(y.data, y_predict, 1)
		epoch_tp += tp
		epoch_fp += fp
		epoch_fn += fn

		if index_batch%(math.ceil(len(dataloader)/10)) == 0:
			print('{0:.4f} --- loss: {1:.6f}\t err: {2:.6f}'.format((index_batch+1)/len(dataloader), loss.data[0], batch_err))

	writer.add_scalar('train_loss', loss.data[0], epoch)
	writer.add_scalar('train_err', epoch_err/train_total, epoch)
	writer.add_scalar('train_precision', 1.*epoch_tp/(epoch_tp+epoch_fp), epoch)
	writer.add_scalar('train_recall', 1.*epoch_tp/(epoch_tp+epoch_fn), epoch)

	log_tmp = 'Train Epoch: {0:.1f}\t Loss: {1:.6f}\t Error: {2:.6f}\tPrecision:{3:.6f}\t' \
			  'recall:{4:.6f}\t,'.format(epoch, epoch_loss/train_total, epoch_err/train_total
																	,1.*epoch_tp/(epoch_tp+epoch_fp),
																	1. * epoch_tp / (epoch_tp + epoch_fn))
	print(log_tmp)
	file_name = 'log_finetune'
	with open("./log/{}.txt".format(file_name), "a") as log:
		log.write('{}\n'.format(log_tmp))

	print('Epoch {} finished ! Train Loss: {}'.format(epoch, loss.data[0]))
	if epoch%5 == 0:
		torch.save(net.state_dict(), check_point_dir + '/CP{}.pth'.format(epoch+1))

def val(criterion, num_valid, epoch, writer, net, dataloader, gpu=True):
	val_total = num_valid
	net.eval()
	acc = 0
	epoch = epoch


	#batch_idx, (data, target) in enumerate(train_loader)
	for index_batch, (img, label) in enumerate(dataloader):
		if gpu:
			X = Variable(img).cuda()
			y = Variable(label).cuda()
		else:
			X = Variable(img)
			y = Variable(label)

		predict = net(X)


		y_predict = F.softmax(predict)
		y_predict = torch.max(y_predict.data, dim=1)[1]

		acc +=torch.sum(y_predict==y.data)


	print('val_acc:{:.6f}'.format(acc/num_valid))

	log_tmp = 'Val Epoch: {0:.1f}\t Accuracy:{1:.6f}\t'.format(epoch, acc/num_valid)
	print(log_tmp)
	file_name = 'log_finetune'
	with open("./log/{}.txt".format(file_name), "a") as log:
		log.write('{}\n'.format(log_tmp))

if __name__ == '__main__':
	device_ids = [0, 1]
	parser = argparse.ArgumentParser(description='Train a DenseNet.')
	parser.add_argument('--batch_size', dest='batch_size', help='batch size', type=int, default=8)
	parser.add_argument('--epoch', dest='epoch', help='num epoch', type=int, default=300)
	parser.add_argument('--save', dest='save', help='check_point_dir', type=str, default='checkpoints/')
	parser.add_argument('--gpu', dest='gpu', help='gpu', type=bool, default=False)
	parser.add_argument('--lr', dest='lr', help='learning rate', type=float, default=0.001)
	parser.add_argument('--load', dest='load', help='load', type=bool, default=False)
	args = parser.parse_args()
	batch_size = args.batch_size
	epoch = args.epoch
	check_point_dir = args.save
	gpu = args.gpu
	lr = args.lr
	load = args.load


	#gpu = True
	net = densenet.DenseNet(
			num_classes=2,
			depth=46,
			growthRate=12,
			compressionRate=2,
			dropRate=0
			)
	writer = SummaryWriter('log/')
	criterion = nn.CrossEntropyLoss()
	if gpu:
		#net.cuda()
		#net = nn.DataParallel(net)
		net = nn.DataParallel(net,device_ids=device_ids)
		net = net.cuda()
		#net.cuda(device_ids[0])
		#net = net.cuda(device_ids[0])
		#criterion.cuda()

	if load:
		net.load_state_dict(torch.load('checkpoints/CP26.pth'))

	train_loader, valid_loader, num_train, num_valid = get_train_valid_loader(
		data_dir='inflammation/',
		batch_size =batch_size,
		num_workers=0,
		pin_memory=False
		)

	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.85, weight_decay=0.0005)
	'''
	for index_batch, (img, label) in enumerate(train_loader):
		print(index_batch)
		print(img.size())
		print(label.size())

	train(criterion, epoch, writer, net, train_loader, optimizer, batch_size=batch_size,
		  check_point_dir=check_point_dir, gpu=gpu)
'''

	for epoch in range(27,40):
		train(criterion,num_train, epoch, writer, net,train_loader,optimizer,batch_size=batch_size, check_point_dir=check_point_dir,gpu=gpu)
		val(criterion,num_valid, epoch, writer,net,valid_loader,gpu=gpu)

	writer.close()
