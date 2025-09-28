#!/usr/bin/env python3
import argparse
import os
import sys
import torch

sys.path.append('/anvil/scratch/x-jliu7')
from src.data.stage2_dataset import Stage2Dataset


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', default='/anvil/scratch/x-jliu7/test_stage2/stage2_samples_floods.csv')
	args = parser.parse_args()

	ds = Stage2Dataset(args.csv, return_meta=True)
	print(f'Dataset size: {len(ds)}')
	for i in range(min(3, len(ds))):
		pre, post, m, r, y, meta = ds[i]
		print('Sample', i, {
			'shape_pre': tuple(pre.shape),
			'shape_post': tuple(post.shape),
			'shape_m': tuple(m.shape),
			'shape_r': tuple(r.shape),
			'label': int(y),
			'event_id': meta['event_id'],
			'tile_id': meta['tile_id'],
			'bldg_uid': meta['bldg_uid'],
		})


if __name__ == '__main__':
	main()


