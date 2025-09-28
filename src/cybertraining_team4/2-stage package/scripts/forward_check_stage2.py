#!/usr/bin/env python3
import argparse
import os
import sys
import torch

a = '/anvil/scratch/x-jliu7'
if a not in sys.path:
	sys.path.append(a)

from src.data.stage2_dataset import Stage2Dataset
from src.models.siamese_stage2 import SiameseDamageModel


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', default='/anvil/scratch/x-jliu7/test_stage2/stage2_samples_floods.csv')
	parser.add_argument('--backbone', default='convnext_tiny')
	parser.add_argument('--classes', type=int, default=4)
	args = parser.parse_args()

	model = SiameseDamageModel(backbone_name=args.backbone, num_classes=args.classes)
	model.eval()

	ds = Stage2Dataset(args.csv) if os.path.isfile(args.csv) else None
	if ds is None or len(ds) == 0:
		print('Dataset empty or missing; running dummy forward...')
		B, C, H, W = 2, 3, 256, 256
		x_pre = torch.randn(B, C, H, W)
		x_post = torch.randn(B, C, H, W)
		m = (torch.rand(B, 1, H, W) > 0.8).float()
		r = (torch.rand(B, 1, H, W) > 0.8).float()
		with torch.no_grad():
			out = model(x_pre, x_post, m, r)
		print({k: tuple(v.shape) for k, v in out.items()})
		return

	# Load first few samples
	pre_list, post_list, m_list, r_list = [], [], [], []
	Y = []
	for i in range(min(2, len(ds))):
		pre, post, m, r, y = ds[i]
		pre_list.append(pre)
		post_list.append(post)
		m_list.append(m)
		r_list.append(r)
		Y.append(y)

	x_pre = torch.stack(pre_list)
	x_post = torch.stack(post_list)
	m = torch.stack(m_list)
	r = torch.stack(r_list)

	with torch.no_grad():
		out = model(x_pre, x_post, m, r)
	print({k: tuple(v.shape) for k, v in out.items()})


if __name__ == '__main__':
	main()
