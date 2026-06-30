import argparse
import logging
import signal
import sys
import torch
import torch.optim as optim

from pydlshogi2.network.policy_value_resnet import build_network, LEGACY_NETWORK_CONFIG
from pydlshogi2.dataloader import HcpeDataLoader

parser = argparse.ArgumentParser(description='Train policy value network')
parser.add_argument('train_data', type=str, nargs='+', help='training data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=1024, help='Number of positions in each test mini-batch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--val_lambda', type=float, default=0.333,
                    help='weight on the game outcome for the value target '
                         '(1.0 = outcome only, <1.0 blends in the search eval)')
parser.add_argument('--eval_coef', type=float, default=600.0,
                    help='sigmoid temperature mapping eval (cp) to a win rate')
parser.add_argument('--checkpoint', default='checkpoints/checkpoint-{epoch:03}.pth', help='checkpoint file name')
parser.add_argument('--resume', '-r', default='', help='Resume from snapshot')
parser.add_argument('--eval_interval', type=int, default=100, help='evaluation interval')
parser.add_argument('--save_interval', type=int, default=0,
                    help='save a checkpoint every N steps (0 = only at epoch end); '
                         'useful for resuming after a preemption mid-epoch')
parser.add_argument('--log', default=None, help='log file path')
# ネットワーク構成 (resume時はチェックポイントの構成を優先)
parser.add_argument('--blocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--channels', type=int, default=256, help='channel width')
parser.add_argument('--fcl', type=int, default=256, help='value head fully-connected size')
parser.add_argument('--no_se', action='store_true', help='disable Squeeze-and-Excitation blocks')
# 高速化オプション
parser.add_argument('--amp', action='store_true', help='enable bfloat16 autocast (mixed precision)')
parser.add_argument('--compile', action='store_true', help='wrap the model with torch.compile')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('lr={}'.format(args.lr))

# 中断シグナル(SIGTERM/SIGINT)を受けたら現在のステップ完了後に保存して終了する
# (Vast.ai等のspotインスタンスのpreemptionに対応)。
# データ読み込み中に来ても取りこぼさないよう、早期に登録する。
interrupted = False

def _handle_interrupt(signum, frame):
    global interrupted
    interrupted = True
    logging.info('Received signal {}; will checkpoint after the current step and exit'.format(signum))

signal.signal(signal.SIGTERM, _handle_interrupt)
signal.signal(signal.SIGINT, _handle_interrupt)

# デバイス
if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

# ネットワーク構成 (resume時はチェックポイントの構成を優先)
if args.resume:
    resume_checkpoint = torch.load(args.resume, map_location=device)
    # 旧checkpointは構成情報を持たないため、load_networkと同じレガシー構成で復元する
    network_config = resume_checkpoint.get('network', LEGACY_NETWORK_CONFIG)
else:
    resume_checkpoint = None
    network_config = {'blocks': args.blocks, 'channels': args.channels,
                      'fcl': args.fcl, 'se': not args.no_se}

# モデル
model = build_network(network_config)
model.to(device)

# オプティマイザ
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

# 損失関数
cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

# チェックポイント読み込み
if resume_checkpoint is not None:
    logging.info('Loading the checkpoint from {}'.format(args.resume))
    epoch = resume_checkpoint['epoch']
    t = resume_checkpoint['t']
    model.load_state_dict(resume_checkpoint['model'])
    optimizer.load_state_dict(resume_checkpoint['optimizer'])
    # 学習率を引数の値に変更
    optimizer.param_groups[0]['lr'] = args.lr
else:
    epoch = 0
    t = 0  # total steps

# state_dict保存・export用に元モデルを保持 (torch.compileは状態辞書のキーを変えるため)
base_model = model
# torch.compile による高速化
if args.compile:
    model = torch.compile(model)

# AMP (bfloat16 autocast)
amp_enabled = args.amp and device.type == 'cuda'
amp_dtype = torch.bfloat16

# 訓練データ読み込み
logging.info('Reading training data')
train_dataloader = HcpeDataLoader(args.train_data, args.batchsize, device, shuffle=True,
                                  val_lambda=args.val_lambda, eval_coef=args.eval_coef)
# テストデータ読み込み
logging.info('Reading test data')
test_dataloader = HcpeDataLoader(args.test_data, args.testbatchsize, device,
                                 val_lambda=args.val_lambda, eval_coef=args.eval_coef)

# 読み込んだデータ数を表示
logging.info('train position num = {}'.format(len(train_dataloader)))
logging.info('test position num = {}'.format(len(test_dataloader)))

# 方策の正解率
def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

# 価値の正解率
def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

# チェックポイント保存
def save_checkpoint():
    path = args.checkpoint.format(**{'epoch':epoch, 'step':t})
    logging.info('Saving the checkpoint to {}'.format(path))
    checkpoint = {
        'epoch': epoch,
        't': t,
        'model': base_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'network': network_config,
    }
    torch.save(checkpoint, path)


# 訓練ループ
for e in range(args.epoch):
    epoch += 1
    steps_interval = 0
    sum_loss_policy_interval = 0
    sum_loss_value_interval = 0
    steps_epoch = 0
    sum_loss_policy_epoch = 0
    sum_loss_value_epoch = 0
    for x, move_label, result in train_dataloader:
        model.train()

        # 順伝播
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
            y1, y2 = model(x)
            # 損失計算
            loss_policy = cross_entropy_loss(y1, move_label)
            loss_value = bce_with_logits_loss(y2, result)
            loss = loss_policy + loss_value
        # 誤差逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # トータルステップ数に加算
        t += 1

        # 中断要求を受けていれば保存して終了する
        if interrupted:
            if args.checkpoint:
                save_checkpoint()
            sys.exit(0)

        # ステップ間隔ごとのチェックポイント保存 (preemption対策)
        if args.save_interval > 0 and t % args.save_interval == 0 and args.checkpoint:
            save_checkpoint()

        # 評価間隔ごとのステップ数カウンタと損失合計に加算
        steps_interval += 1
        sum_loss_policy_interval += loss_policy.item()
        sum_loss_value_interval += loss_value.item()

        # 評価間隔ごとに訓練損失とテスト損失・正解率を表示
        if t % args.eval_interval == 0:
            model.eval()

            x, move_label, result = test_dataloader.sample()
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                # 推論
                y1, y2 = model(x)
                # 損失計算
                test_loss_policy = cross_entropy_loss(y1, move_label).item()
                test_loss_value = bce_with_logits_loss(y2, result).item()
                # 正解率計算
                test_accuracy_policy = accuracy(y1, move_label)
                test_accuracy_value = binary_accuracy(y2, result)

                # ログ表示
                logging.info('epoch = {}, steps = {}, train loss = {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}'.format(
                    epoch, t,
                    sum_loss_policy_interval / steps_interval,
                    sum_loss_value_interval / steps_interval,
                    (sum_loss_policy_interval +
                     sum_loss_value_interval) / steps_interval,
                    test_loss_policy,
                    test_loss_value,
                    test_loss_policy + test_loss_value,
                    test_accuracy_policy,
                    test_accuracy_value))

            # エポックごとのステップ数カウンタと損失合計に加算
            steps_epoch += steps_interval
            sum_loss_policy_epoch += sum_loss_policy_interval
            sum_loss_value_epoch += sum_loss_value_interval

            # 評価間隔ごとのステップ数カウンタと損失合計をクリア
            steps_interval = 0
            sum_loss_policy_interval = 0
            sum_loss_value_interval = 0

    # エポックごとのステップ数カウンタと損失合計に加算
    steps_epoch += steps_interval
    sum_loss_policy_epoch += sum_loss_policy_interval
    sum_loss_value_epoch += sum_loss_value_interval

    # エポックの終わりにテストデータすべてを使用して評価する
    test_steps = 0
    sum_test_loss_policy = 0
    sum_test_loss_value = 0
    sum_test_accuracy_policy = 0
    sum_test_accuracy_value = 0
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
        for x, move_label, result in test_dataloader:
            y1, y2 = model(x)

            test_steps += 1
            sum_test_loss_policy += cross_entropy_loss(y1, move_label).item()
            sum_test_loss_value += bce_with_logits_loss(y2, result).item()
            sum_test_accuracy_policy += accuracy(y1, move_label)
            sum_test_accuracy_value += binary_accuracy(y2, result)

    # テストデータの検証結果をログ表示
    logging.info('epoch = {}, steps = {}, train loss avr = {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}'.format(
        epoch, t,
        sum_loss_policy_epoch / steps_epoch,
        sum_loss_value_epoch / steps_epoch,
        (sum_loss_policy_epoch + sum_loss_value_epoch) / steps_epoch,
        sum_test_loss_policy / test_steps,
        sum_test_loss_value / test_steps,
        (sum_test_loss_policy + sum_test_loss_value) / test_steps,
        sum_test_accuracy_policy / test_steps,
        sum_test_accuracy_value / test_steps))

    # チェックポイント保存
    if args.checkpoint:
        save_checkpoint()
