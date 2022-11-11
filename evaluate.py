from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
from pathlib import Path
from math import ceil
from time import perf_counter
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision.models import inception_v3

from progan_modules import Generator, Discriminator
from utils.evaluation import imagenet2012_normalize, inception_score, fid
from utils.device import AutoDevice
import utils.dwt as dwt
import utils.dct as dct
import torchvision.datasets as dset


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = False


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def imagefolder_loader(path):
    def loader(transform):
        data = dset.LSUN(
            root='/home/k_yamanouchi/.datasets/vision',
            classes=['bedroom_train'],
            transform=transform)
        data_loader = DataLoader(data, shuffle=False, batch_size=batch_size,
                                 num_workers=0, drop_last=False)
        return data_loader
    return loader


def evaluate_data(dataloader):
    transform = transforms.Compose([
        transforms.CenterCrop(4 * 2 ** 6),
        transforms.Resize(4 * 2 ** args.max_step),
        transforms.ToTensor()
    ])
    loader = dataloader(transform)
    return loader


def preprocess(images: torch.Tensor) -> torch.Tensor:
    '''Inception V3用の画像の前処理を行います。

    Args:
        images: 画素値[0.0 ~ 1.0]の画像バッチ(B, C, H, W)

    Returns:
        前処理済みの画像バッチ(B, C, 299, 299)
    '''
    # Inception V3の入力(299×299)のための画像拡大モジュール
    upsample = nn.UpsamplingBilinear2d((299, 299))
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    images = imagenet2012_normalize(images, inplace=True)
    images = upsample(images)
    return images


def inception_forward(images: torch.Tensor, feature_extractor, classifier) -> Tuple[np.ndarray, np.ndarray]:
    '''FID Inception Scoreの算出に必要な計算結果を取得する。

    Args:
        前処理済みの画像バッチ(B, C, 299, 299)

    Returns:
        (最終層の特徴量(B, 2048)，各クラスに属する確率(B, 1000))
    '''
    features = feature_extractor(images).view(-1, 2048)
    logits = classifier(features)
    return features.cpu().numpy(), logits.cpu().numpy()


def train(generator):
    torch_fix_seed(args.seed)

    from datetime import datetime
    import os
    date_time = datetime.now()
    post_fix = '%s_%s_%d_%d.txt' % (trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'eval_%s_%s_%d_%d' % (trial_name, date_time.date(), date_time.hour, date_time.minute)

    os.mkdir(log_folder)
    os.mkdir(log_folder + '/sample')

    config_file_name = os.path.join(log_folder, 'train_config_' + post_fix)
    config_file = open(config_file_name, mode='w', encoding='utf-8')
    config_file.write(str(args))
    config_file.close()

    evaluate_file_name = os.path.join(log_folder, 'evaluate_' + post_fix)
    evaluate_file = open(evaluate_file_name, mode='w', encoding='utf-8')
    evaluate_file.write(f'乱数生成器のシード値: {args.seed}\n')
    evaluate_file.close()

    from shutil import copy
    copy('train.py', log_folder + '/train_%s.py' % post_fix)
    copy('progan_modules.py', log_folder + '/model_%s.py' % post_fix)

    with torch.no_grad():
        for i in range(10):
            images = generator(torch.randn(1 * 1, input_code_size).to(device), step=args.max_step, alpha=1).data.cpu()

            utils.save_image(
                images,
                f'{log_folder}/sample/{str(i + 1).zfill(2)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1))

    # Start evaluating
    loader = imagefolder_loader(args.path)
    data_loader = evaluate_data(loader)
    evaluate_file = open(evaluate_file_name, mode='w', encoding='utf-8')
    if device != 'cpu':
        prop = torch.cuda.get_device_properties(device)
        evaluate_file.write(
            f'使用デバイス: {prop.name}\n'
            f'  メモリ容量: {prop.total_memory // 1048576}[MiB]\n'
            f'  Compute Capability: {prop.major}.{prop.minor}\n'
            f'  ストリーミングマルチプロセッサ数: {prop.multi_processor_count}[個]\n'
        )
    else:
        evaluate_file.write('使用デバイス: CPU\n')

    generator.eval()
    # ImageNet2012訓練済みInception V3を読み込む
    inception = inception_v3(pretrained=True, aux_logits=False)
    inception_children = list(inception.children())
    # 特徴抽出器(FIDで使用)
    feature_extractor = nn.Sequential(*inception_children[:-2]).to(device)
    feature_extractor.eval()
    # クラス分類器(Inception Scoreで使用)
    classifier = nn.Sequential(
        inception_children[-1], nn.Softmax(dim=1)).to(device)
    classifier.eval()

    assets_root = Path('./assets')
    assets_dir = assets_root.joinpath('./lsun-bedroom')
    features_path = assets_dir.joinpath('features.npz')
    logits_path = assets_dir.joinpath('logits.npz')
    labels_path = assets_dir.joinpath('labels.npz')
    if (
        assets_dir.exists() and assets_dir.is_dir() and
        features_path.exists() and features_path.is_file() and
        logits_path.exists() and logits_path.is_file() and
        labels_path.exists() and labels_path.is_file()
    ):
        # 既に計算された結果がある場合は読み込み
        print("loaded")
        train_features = np.load(features_path)['features']
        train_logits = np.load(logits_path)['logits']
        train_labels = np.load(labels_path)['labels']
    else:
        features_list = []
        logits_list = []
        labels_list = []
        dataset = iter(data_loader)
        pbar = tqdm(
            enumerate(dataset),
            desc='データセット画像から特徴を抽出中...',
            total=len(dataset),
            leave=False)
        for i, (images, labels) in pbar:
            pass
            with torch.no_grad():
                images = preprocess(images)
                images = images.to(device)
                features, logits = inception_forward(images, feature_extractor, classifier)
                features_list.append(features)
                logits_list.append(logits)
                labels_list.append(labels)
        train_features = np.concatenate(features_list, axis=0)
        train_logits = np.concatenate(logits_list, axis=0)
        train_labels = np.concatenate(labels_list, axis=0)
        assets_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            assets_dir.joinpath('features.npz'), features=train_features)
        np.savez_compressed(
            assets_dir.joinpath('logits.npz'), logits=train_logits)
        np.savez_compressed(
            assets_dir.joinpath('labels.npz'), labels=train_labels)
        with open(
            assets_dir.joinpath('evaluation.txt'), mode='w', encoding='utf-8'
        ) as f:
            f.write(f'画像数: {train_labels.shape[0]}\n')
            f.write(f'Inception Score: {inception_score(train_logits)}\n')
        print(f'{assets_dir}にデータセット画像の評価を保存しました。')

    # =========================================================================== #
    # 生成画像の評価
    # =========================================================================== #
    features_list = []
    logits_list = []
    torch_fix_seed(args.seed)
    with torch.no_grad():
        for i in range(ceil(50000 / batch_size)):
            z = torch.randn(batch_size, input_code_size).to(device)
            images = generator(z, step=args.max_step, alpha=1)
            images = preprocess(images)
            features, logits = inception_forward(images, feature_extractor, classifier)
            features_list.append(features)
            logits_list.append(logits)
    features = np.concatenate(features_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    s = f'Inception Score: {inception_score(logits)}'
    print(s)
    evaluate_file.write(f'{s}\n')
    s = f'FID: {fid(features, train_features)}'
    print(s)
    evaluate_file.write(f'{s}\n')
    evaluate_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    parser.add_argument('--path', type=str, default="./image_root_folder/celeba", help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--lg', type=str, default=None, help='load generator path')
    parser.add_argument('--trial_name', type=str, default="test1", help='a brief description of the training trial')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--seed', type=int, default=42, help='random seed.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=128, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=128, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=8, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train Dhow many times while train G 1 time')
    parser.add_argument('--init_step', type=int, default=1, help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--max_step', type=int, default=6, help='max resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--total_iter', type=int, default=300000, help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=False, action="store_true", help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--tanh', default=False, action="store_true", help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')
    parser.add_argument('--dct', default=False, action="store_true", help='use disccrete cosine transform')
    parser.add_argument('--dwt', default=False, action="store_true", help='use discrete wavelete transform')

    args = parser.parse_args()

    print(str(args))

    trial_name = args.trial_name
    device = torch.device("cuda:%d"%(args.gpu_id))
    input_code_size = args.z_dim
    batch_size = args.batch_size
    n_critic = args.n_critic
    seed = args.seed

    generator = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)

    # you can directly load a pretrained model here
    # g_running.load_state_dict(torch.load('checkpoint/150000_g.model'))
    generator.load_state_dict(torch.load(args.lg))

    train(generator)
