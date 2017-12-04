import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args()

    df = pd.read_json(args.log_file)

    plt.figure(figsize=(8, 4))
    plt.plot(df['iteration'][1:], df['main/loss'][1:], linewidth=1, label='train')
    plt.plot(df['iteration'][df['val/loss'].notnull()], df['val/loss'][df['val/loss'].notnull()], linewidth=1, label='validation')
    plt.legend(loc='best')
    # plt.ylim(0, 0.1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    save_dir = '/'.join(args.log_file.split('/')[:-1])
    plt.savefig(os.path.join(save_dir, 'loss_history.png'))
