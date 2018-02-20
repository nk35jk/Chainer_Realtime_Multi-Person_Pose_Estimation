import subprocess
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arch')
    parser.add_argument('result')
    parser.add_argument('--start', '-s', type=int, default=10000)
    parser.add_argument('--end', '-e', type=int, default=300000)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    args = parser.parse_args()

    for i in range(args.start, args.end+1, args.step):
        print(i)
         model_path = '{}/model_iter_{}'.format(args.result, i)
         if os.path.exists(model_path):
             subprocess.call('python evaluate_coco.py {} {} -g{}'.format(
                args.arch, model_path, args.gpu), shell=True)
