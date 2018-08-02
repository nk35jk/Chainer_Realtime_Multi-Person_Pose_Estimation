import os
import sys
import cv2
import argparse
import numpy as np

from pycocotools.coco import COCO

from entity import params
from coco_data_loader import CocoDataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                        help='visualize annotations and ignore masks')
    args = parser.parse_args()

    for mode in ['train', 'val']:
        coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
        data_loader = CocoDataLoader(params['coco_dir'], coco, params['insize'],
                                     mode=mode, use_all_images=False, use_ignore_mask=True,
                                     augment_data=False, resize_data=False)

        save_dir = os.path.join(params['coco_dir'], 'labels_{}2017'.format(mode))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        for i in range(len(data_loader)):
            img, img_id, annotations, ignore_mask, _ = data_loader.get_img_annotation(ind=i)

            print('{}/{}, img_id: {}'.format(i+1, len(data_loader), img_id))

            resized_img, pafs, heatmaps, ignore_mask = data_loader.gen_labels(img, annotations, ignore_mask)

            concat_data = np.concatenate([pafs, heatmaps], axis=0)

            shorter_len = 100
            h, w = concat_data.shape[1:]
            if h > shorter_len and w > shorter_len:
                shape = (shorter_len, int(h*shorter_len/w)) if h > w else (int(w*shorter_len/h), shorter_len)
                concat_data = cv2.resize(concat_data.transpose(1, 2, 0), shape, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)

            print(concat_data.shape)

            if args.dry_run:
                img_to_show = resized_img.copy()
                img_to_show = data_loader.overlay_pafs(img_to_show, pafs, .2, .8)
                img_to_show = data_loader.overlay_heatmap(img_to_show, heatmaps[:-1].max(axis=0), .5, .5)
                img_to_show = data_loader.overlay_ignore_mask(img_to_show, ignore_mask, .5, .5)

                cv2.imshow('image', np.hstack([resized_img, img_to_show]))

                k = cv2.waitKey(1)
                if k == ord('q'):
                    exit()
                elif k == ord('d'):
                    import ipdb; ipdb.set_trace()
            else:
                save_path = os.path.join(save_dir, str(img_id).zfill(12))
                np.save(save_path, concat_data)
