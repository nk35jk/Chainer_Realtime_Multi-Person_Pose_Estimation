import subprocess
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    parser.add_argument('result')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    args = parser.parse_args()

    for i in range(1, 30+1):
         model_path = '{}/model_iter_{}'.format(args.result, i*10000)
         if os.path.exists(model_path):
             subprocess.call('python evaluate_coco.py {} {} -g{}'.format(args.arch, model_path, args.gpu), shell=True)
