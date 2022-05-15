# 標準モジュール
import argparse
from datetime import datetime
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from math import ceil
from pathlib import Path
import random
from time import perf_counter
from typing import Tuple
import sys

# 追加モジュール
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from tqdm import tqdm

# 自作モジュール
from utils.datasets import load_dataset
from utils.evaluation import imagenet2012_normalize, inception_score, fid
from utils.device import AutoDevice
from progan_modules import Generator


DATASETS_ROOT = './image_root_folder/celeba/img_align_celeba'
# =========================================================================== #
# コマンドライン引数の設定
# =========================================================================== #
parser = argparse.ArgumentParser(
    prog='PyTorch Generative Adversarial Network',
    description='PyTorchを用いてGANの画像生成を行います。'
)

# 訓練に関する引数

parser.add_argument(
    '-lg', '--load_generator', help='指定したパスのGeneratorのセーブファイルを読み込みます。',
    type=str, default=None
)

parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=4, metavar='B'
)

# PyTorchに関するコマンドライン引数
parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=999
)

parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')

parser.add_argument('--z_dim', type=int, default=128, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
parser.add_argument('--channel', type=int, default=128, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
parser.add_argument('--batch_size', type=int, default=4, help='how many images to train together at one iteration')
parser.add_argument('--n_critic', type=int, default=1, help='train Dhow many times while train G 1 time')

parser.add_argument(
    '--data-path', help='データセットのパスを指定します。',
    type=str, default=DATASETS_ROOT
)

parser.add_argument(
    '--info', help='ログ表示レベルをINFOに設定し、詳細なログを表示します。',
    action='store_true'
)
parser.add_argument(
    '--debug', help='ログ表示レベルをDEBUGに設定し、より詳細なログを表示します。',
    action='store_true'
)
# コマンドライン引数をパースする
args = parser.parse_args()

# 結果を出力するために起動日時を保持する
launch_datetime = datetime.now()

# ロギングの設定
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)
# 名前を指定してロガーを取得する
logger = getLogger('main')

generator_path = Path(args.load_generator)
generator_dir = generator_path.parent
input_code_size = args.z_dim
batch_size = args.batch_size
n_critic = args.n_critic

device = torch.device("cuda:%d"%(args.gpu_id))

# Generator生成
generator = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)
g_running = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)

# チェックポイントの読み込み
if args.load_generator is not None:
    generator.load_state_dict(torch.load(generator_path))
    g_running.load_state_dict(torch.load(generator_path))
else:
    print("No checkpoint")
    exit

idx = 1
output_path = generator_dir.joinpath(f'evaluation{idx}.txt')
while output_path.exists():
    idx += 1
    output_path = generator_dir.joinpath(f'evaluation{idx}.txt')
output_txt = open(output_path, mode='w', encoding='utf-8')


# =========================================================================== #
# 再現性の設定 https://pytorch.org/docs/stable/notes/randomness.html
# =========================================================================== #
random.seed(args.seed)                     # Pythonの乱数生成器のシード値の設定
np.random.seed(args.seed)                  # NumPyの乱数生成器のシード値の設定
torch.manual_seed(args.seed)               # PyTorchの乱数生成器のシード値の設定
torch.backends.cudnn.deterministic = True  # Pytorchの決定性モードの有効化
torch.backends.cudnn.benchmark = False     # Pytorchのベンチマークモードの無効化
logger.info('乱数生成器のシード値を設定しました。')
output_txt.write(f'乱数生成器のシード値: {args.seed}\n')

# デバイスについての補助クラスをインスタンス化
auto_device = AutoDevice(disable_cuda=args.disable_cuda)
logger.info('デバイスの優先順位を計算しました。')
device = auto_device()
logger.info(f'メインデバイスとして〈{device}〉が選択されました。')
if device != 'cpu':
    prop = torch.cuda.get_device_properties(device)
    output_txt.write(
        f'使用デバイス: {prop.name}\n'
        f'  メモリ容量: {prop.total_memory // 1048576}[MiB]\n'
        f'  Compute Capability: {prop.major}.{prop.minor}\n'
        f'  ストリーミングマルチプロセッサ数: {prop.multi_processor_count}[個]\n'
    )
else:
    output_txt.write('使用デバイス: CPU\n')


# =========================================================================== #
# モデルの読み込み
# =========================================================================== #
model_g = Generator(
    nz=nz, nc=nc)
model_g = model_g.to(device)
model_g.eval()
model_g.load_state_dict(state_dict)

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

# Inception V3の入力(299×299)のための画像拡大モジュール
upsample = nn.UpsamplingBilinear2d((299, 299))


def preprocess(images: torch.Tensor) -> torch.Tensor:
    '''Inception V3用の画像の前処理を行います。

    Args:
        images: 画素値[0.0 ~ 1.0]の画像バッチ(B, C, H, W)

    Returns:
        前処理済みの画像バッチ(B, C, 299, 299)
    '''
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    images = imagenet2012_normalize(images, inplace=True)
    images = upsample(images)
    return images


def inception_forward(images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    '''FID Inception Scoreの算出に必要な計算結果を取得する。

    Args:
        前処理済みの画像バッチ(B, C, 299, 299)

    Returns:
        (最終層の特徴量(B, 2048)，各クラスに属する確率(B, 1000))
    '''
    features = feature_extractor(images).view(-1, 2048)
    logits = classifier(features)
    return features.cpu().numpy(), logits.cpu().numpy()


# =========================================================================== #
# キャッシュの読み込み／訓練画像の評価
# =========================================================================== #
if nc == 1 and checkpoint['dataset'] in ['cifar10', 'stl10']:
    suffix = '_grayscale'
else:
    suffix = ''
assets_root = Path('./assets')
assets_dir = assets_root.joinpath(checkpoint['dataset'] + suffix)
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
    train_features = np.load(features_path)['features']
    train_logits = np.load(logits_path)['logits']
    train_labels = np.load(labels_path)['labels']
else:  # 無い場合は計算してその結果を保存
    # ======================================================================= #
    # データ整形
    # ======================================================================= #
    logger.info('画像に適用する変換のリストを定義します。')
    data_transforms = []

    if checkpoint['dataset'] in ['mnist', 'fashion_mnist']:
        # MNIST/Fashion MNISTは28×28画素なのでゼロパディング
        data_transforms.append(
            transforms.Pad(2, fill=0, padding_mode='constant')
        )
        logger.info('変換リストにゼロパディングを追加しました。')

    if nc == 1 and checkpoint['dataset'] in ['cifar10', 'stl10']:
        data_transforms.append(
            transforms.Grayscale(num_output_channels=1)
        )
        logger.info('変換リストにグレースケール化を追加しました。')

    data_transforms.append(transforms.ToTensor())
    logger.info('変換リストにテンソル化を追加しました。')
    # ======================================================================= #
    # データセットの読み込み／データローダーの定義
    # ======================================================================= #
    dataset = load_dataset(
        checkpoint['dataset'], root=args.data_path, transform=data_transforms)
    logger.info(f"データセット〈{checkpoint['dataset']}〉を読み込みました。")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, drop_last=False)
    logger.info('データローダを生成しました。')

    features_list = []
    logits_list = []
    labels_list = []
    pbar = tqdm(
        enumerate(dataloader),
        desc='データセット画像から特徴を抽出中...',
        total=ceil(len(dataset) / batch_size),
        leave=False)
    for i, (images, labels) in pbar:
        with torch.no_grad():
            images = preprocess(images)
            images = images.to(device)
            features, logits = inception_forward(images)
            features_list.append(features)
            logits_list.append(logits)
            labels_list.append(labels)
    logger.info('特徴抽出終了。')
    train_features = np.concatenate(features_list, axis=0)
    train_logits = np.concatenate(logits_list, axis=0)
    train_labels = np.concatenate(labels_list, axis=0)
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info('ディレクトリ作成。')
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
# 画像生成時間の計測
# =========================================================================== #
pbar = tqdm(range(10000), desc='画像生成時間を計測中...', total=10000, leave=False)
with torch.no_grad():
    begin_time = perf_counter()
    for _ in pbar:
        z = torch.randn(1, nz, device=device)
        fakes = model_g(z)
    end_time = perf_counter()
s = f'画像生成時間: {(end_time - begin_time) / 10000:.07f}[s/image]'
print(s)
output_txt.write(f'{s}\n')

# =========================================================================== #
# 生成画像の評価
# =========================================================================== #
features_list = []
logits_list = []
with torch.no_grad():
    for i in range(ceil(50000 / batch_size)):
        z = torch.randn(batch_size, nz, device=device)
        images = model_g(z)
        images = (images + 1.0) / 2  # -1.0 ~ 1.0 -> 0.0 ~ 1.0
        images = preprocess(images)
        features, logits = inception_forward(images)
        features_list.append(features)
        logits_list.append(logits)
features = np.concatenate(features_list, axis=0)
logits = np.concatenate(logits_list, axis=0)
s = f'Inception Score: {inception_score(logits)}'
print(s)
output_txt.write(f'{s}\n')
s = f'FID: {fid(features, train_features)}'
print(s)
output_txt.write(f'{s}\n')
output_txt.close()
