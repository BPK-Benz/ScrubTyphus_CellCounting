import os
import json
import pandas as pd
# from 5_predict import condition 


def load_coco(coco_path):
    with open(coco_path) as file:
        coco = json.load(file)
    return coco

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

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def checksum(scores):
	total = 0
	for name in scores:
		total += scores[name]['tp']
		total += scores[name]['fn']
	return total

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
            }
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
	
	# base = '/share/NAS/MM_software/mmdetection/work_dirs/New_OCT/ATSS_R50_InfectNuc/'
	base = '/share/NAS/Benz_Cell/cellLabel-main/Coco_File/'

	# load ground truth
	gts_path = '/share/NAS/Benz_Cell/cellLabel-main/Coco_File/InfectTotal_TestNuc_3class.json' ### edit
	gts = load_coco(gts_path)
	gts_maps = read_anno(gts)

	# image processing
	# pds_path = 'image_processing.json'
	pds_path = base+'InfectTotal_TestNuc_3class.json'
	pds = load_coco(pds_path)
	pds_maps = read_anno(pds)

	images = gts['images']
	total = len(images)
	total_instances = 0

	categories = condition()
	columns = ['file_name']
	scores = {}
	for c in categories:
		scores[c['name']] = { 'tp': 0, 'fp': 0, 'fn': 0 }
		columns += [
			c['name'] + '_tp',
			c['name'] + '_fp',
			c['name'] + '_fn',
		]
	df = pd.DataFrame(columns=columns)

	threshold_iou = 0.3

	# loop for each images
	for index in range(total):

		# print progress
		# print('[ Processing {} of {} ]'.format(index, total))
		row = {}
		row['file_name'] = gts['images'][index]['file_name']

		# get all sample
		gt = []
		if gts['images'][index]['id'] in gts_maps:
			for j in gts_maps[gts['images'][index]['id']]:
				d = gts['annotations'][j]
				d['class'] = condition(d)
				gt.append(d)
		total_instances += len(gt)

		# get all predict
		pd = []
		if pds['images'][index]['id'] in pds_maps:
			for j in pds_maps[pds['images'][index]['id']]:
				d = pds['annotations'][j]
				if 'score' in d:
					d['class'] = d['category_id']
				else:
					d['class'] = condition(d)
				pd.append(d)

		for c in categories:


			filtered_gt = [o for o in gt if o['class'] == c['id']]
			filtered_pd = [o for o in pd if o['class'] == c['id']]


			matched_gt = []
			matched_pd = []

			for i in range(len(filtered_gt)):
				bbox1 = {
					'x1': filtered_gt[i]['bbox'][0],
					'y1': filtered_gt[i]['bbox'][1],
					'x2': filtered_gt[i]['bbox'][0] + filtered_gt[i]['bbox'][2],
					'y2': filtered_gt[i]['bbox'][1] + filtered_gt[i]['bbox'][3],
				}
				for j in range(len(filtered_pd)):
					bbox2 = {
						'x1': filtered_pd[j]['bbox'][0],
						'y1': filtered_pd[j]['bbox'][1],
						'x2': filtered_pd[j]['bbox'][0] + filtered_pd[j]['bbox'][2],
						'y2': filtered_pd[j]['bbox'][1] + filtered_pd[j]['bbox'][3],
					}
					iou = get_iou(bbox1, bbox2)
					if iou >= threshold_iou:
						matched_gt += [i]
						matched_pd += [j]

			tps = set(matched_gt)
			fps = [j for j in range(len(filtered_pd)) if not j in matched_pd]
			fns = [i for i in range(len(filtered_gt)) if not i in matched_gt]

			count_tp = len(tps)
			count_fp = len(fps) + len(matched_gt) - len(tps)
			count_fn = len(fns)

			scores[c['name']]['tp'] += count_tp
			scores[c['name']]['fp'] += count_fp
			scores[c['name']]['fn'] += count_fn

			row[c['name'] + '_tp'] = count_tp
			row[c['name'] + '_fp'] = count_fp
			row[c['name'] + '_fn'] = count_fn
		df = df.append(row, ignore_index=True)

	df_path = base+'Compare.csv' # output per image csv: total instance = tp + fn
	df.to_csv(df_path)  

	for c in categories:

		count_tp = scores[c['name']]['tp']
		count_fp = scores[c['name']]['fp']
		count_fn = scores[c['name']]['fn']
		precision = count_tp / (count_tp + count_fp)
		recall = count_tp / (count_tp + count_fn)
		f1 = 2 * precision * recall / (precision + recall)

		print('\t', c['name'])
		print('\t\tTrue Positive  =', count_tp)
		print('\t\tFalse Positive =', count_fp)
		print('\t\tFalse Negative =', count_fn)
		print('\t\tPrecision      =', round(precision, 3))
		print('\t\tRecall         =', round(recall, 3))
		print('\t\tF1 score       =', round(f1, 3))

	print('total_instances', total_instances)
	print('checksum_scores', checksum(scores))