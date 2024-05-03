import os
import cv2
import json
import numpy as np

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def load_coco(coco_path):
    with open(coco_path) as file:
        coco = json.load(file)
    return coco

def save_coco(coco_path, coco):
    with open(coco_path, 'w') as outfile:
        json.dump(coco, outfile)

def read_anno(coco):
	maps = {}
	for i, annotation in enumerate(coco['annotations']):
		image_id = annotation['image_id']
		if not image_id in maps:
			maps[image_id] = [i]
		else:
			maps[image_id].append(i)
	print('[ Finished Read Annotations ]')
	return maps

def check_border(x, y, w, h, img_w, img_h):
	margin = 5
	return int(
		x < margin or
		y < margin or
		x + w > img_w - margin or
		y + h > img_h - margin
	)

def condition(annotation=None): ##### copy condition like 2_manage
    if not annotation:
         return[
             {
                "supercategory": 'Cell', # cell, nucleus, properties of objects
                "id": 1, # order of class
                "name": 'Infected_cells', # class name
            },
            {
                "supercategory": 'Cell',
                "id": 2,
                "name": 'Uninfected_cells'
            },
            {
                "supercategory": 'Cell',
                "id": 3,
                "name": 'Divided_cells',
            },
            {
                "supercategory": 'Cell',
                "id": 4,
                "name": 'Border_cells',
            },
        ]
    else:

        channel = annotation['channel']
        divide = annotation['divide']
        border = annotation['border']
        infect = annotation['infect']

        if channel == 'nucleus':
            if divide: return 3
            elif border: return 4
            elif infect == "non-infected": return 2
            else: return 1


if __name__ == "__main__":

	base = '/share/NAS/MM_software/mmdetection/work_dirs/New_OCT/ATSS_R50_InfectNuc/'

	# load ground truth
	gts_path = '/share/NAS/Benz_Cell/cellLabel-main/Coco_File/InfectTotal_TestNuc_April.json' ### edit
	gts = load_coco(gts_path)
	gts_maps = read_anno(gts)

	# make prediction
	# pds_path = 'DCN_101_CellNucNo_MaxDets.json' ### expected output_file of Each model
	pds = gts.copy()
	pds['annotations'] = []

	# model
	config = base+'atss_r50_InfectNucNo.py'
	checkpoint = base+'best_bbox_mAP_epoch_11.pth'
	model = init_detector(config, checkpoint, device='cuda:0')

	categories = condition()
	scale = 3/4 #### check image scale in config file
	count_annotation = 0
	images = gts['images']
	total = len(images)

	# loop for each images
	for index in range(total):

		print('[ Processing {} of {} | {} ]'.format(index, total, gts['images'][index]['file_name']))

		# load image
		image_path = gts['images'][index]['file_name']
		cv2image = cv2.imread(image_path)

		image = mmcv.imread(image_path)
		image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
		results = inference_detector(model, image)

		img_h = gts['images'][index]['height']
		img_w = gts['images'][index]['width']

		for c in range(len(results)):

			for result in results[c]:

				x, y, w, h, s = result
				x = int(result[0] / scale)
				y = int(result[1] / scale)
				w = int((result[2] - result[0]) / scale)
				h = int((result[3] - result[1]) / scale)
				s = str(result[4])
				pds['annotations'].append({
					'id': count_annotation,
					'image_id': gts['images'][index]['id'],
					'channel': 'cell',
					'divide': int(c == 1),
					'infect': '',
					'border': check_border(x, y, w, h, img_w, img_h),
					'bbox': [x, y, w, h],
					'category_id': c+1,
					'score': s
				})
				count_annotation += 1

	
	pds_path = base+'Model_prediction.json' ### expected output_file of Each model
	save_coco(pds_path, pds)
	print('[ Finish! ]')
