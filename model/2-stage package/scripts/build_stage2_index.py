#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Any, Tuple


def list_label_pairs(labels_dir: str) -> Dict[str, Tuple[str, str]]:
	pairs: Dict[str, Tuple[str, str]] = {}
	for name in os.listdir(labels_dir):
		if not name.endswith('.json'):
			continue
		path = os.path.join(labels_dir, name)
		if name.endswith('_pre_disaster.json'):
			base = name[:-len('_pre_disaster.json')]
			pairs.setdefault(base, [None, None])
			pairs[base][0] = path
		elif name.endswith('_post_disaster.json'):
			base = name[:-len('_post_disaster.json')]
			pairs.setdefault(base, [None, None])
			pairs[base][1] = path
	# keep only complete pairs
	return {k: (v[0], v[1]) for k, v in pairs.items() if v[0] and v[1]}


def load_json(path: str) -> Dict[str, Any]:
	with open(path, 'r') as f:
		return json.load(f)


def uid_to_xy_wkt(features: Dict[str, Any]) -> Dict[str, str]:
	result: Dict[str, str] = {}
	for geom in features.get('xy', []):
		props = geom.get('properties', {})
		if props.get('feature_type') != 'building':
			continue
		uid = props.get('uid')
		wkt = geom.get('wkt')
		if uid and wkt:
			result[uid] = wkt
	return result


def iterate_post_buildings(features: Dict[str, Any]):
	for geom in features.get('xy', []):
		props = geom.get('properties', {})
		if props.get('feature_type') != 'building':
			continue
		yield props, geom.get('wkt')


LABEL_MAP = {
	'no-damage': 0,
	'minor-damage': 1,
	'major-damage': 2,
	'destroyed': 3,
}


def build_index(root: str, out_csv: str) -> int:
	images_dir = os.path.join(root, 'images')
	labels_dir = os.path.join(root, 'labels')
	assert os.path.isdir(images_dir), f"Missing images dir: {images_dir}"
	assert os.path.isdir(labels_dir), f"Missing labels dir: {labels_dir}"

	pairs = list_label_pairs(labels_dir)
	rows = []
	for base, (pre_json_path, post_json_path) in pairs.items():
		pre = load_json(pre_json_path)
		post = load_json(post_json_path)
		meta = post.get('metadata', {})
		if meta.get('disaster_type') != 'flooding':
			continue
		event_id = meta.get('disaster')
		width = int(meta.get('width', 1024))
		height = int(meta.get('height', 1024))

		pre_img = os.path.join(images_dir, f"{base}_pre_disaster.png")
		post_img = os.path.join(images_dir, f"{base}_post_disaster.png")
		if not (os.path.isfile(pre_img) and os.path.isfile(post_img)):
			continue

		uid2pre = uid_to_xy_wkt(pre.get('features', {}))
		for props, post_xy_wkt in iterate_post_buildings(post.get('features', {})):
			uid = props.get('uid')
			subtype = props.get('subtype')
			if subtype not in LABEL_MAP:
				# skip un-classified or unlabeled
				continue
			cls = LABEL_MAP[subtype]
			pre_xy_wkt = uid2pre.get(uid, None)
			rows.append({
				'event_id': event_id,
				'tile_id': base,
				'width': width,
				'height': height,
				'pre_image': pre_img,
				'post_image': post_img,
				'pre_json': pre_json_path,
				'post_json': post_json_path,
				'bldg_uid': uid,
				'damage_subtype': subtype,
				'damage_class': cls,
				'polygon_wkt_xy_pre': pre_xy_wkt or '',
				'polygon_wkt_xy_post': post_xy_wkt or '',
			})

	# ensure output directory exists
	os.makedirs(os.path.dirname(out_csv), exist_ok=True)
	fieldnames = [
		'event_id', 'tile_id', 'width', 'height',
		'pre_image', 'post_image', 'pre_json', 'post_json',
		'bldg_uid', 'damage_subtype', 'damage_class',
		'polygon_wkt_xy_pre', 'polygon_wkt_xy_post',
	]
	with open(out_csv, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in rows:
			writer.writerow(r)
	return len(rows)


def main():
	parser = argparse.ArgumentParser(description='Build Stage-2 (flood-only) index CSV from xBD/xView2-style test set.')
	parser.add_argument('--root', default='/anvil/scratch/x-jliu7/test', help='Root of test dataset containing images/, labels/')
	parser.add_argument('--out', default='/anvil/scratch/x-jliu7/test/stage2_index_floods.csv', help='Output CSV path')
	args = parser.parse_args()
	count = build_index(args.root, args.out)
	print(f'Wrote {count} building rows to {args.out}')


if __name__ == '__main__':
	main()


