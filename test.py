from argparse import ArgumentParser
import os
import build

parser = ArgumentParser()

parser.add_argument('--gpu', type=str, dest='gpu', default='0', help='gpu number')
parser.add_argument('--scale', type=str, dest='scale', default='x2', help='choose scale: x2, x3, x4')
parser.add_argument('--dataset', type=str, dest='dataset', default='Set5', help='Dataset: Set5, Set14, BSDS100, Urban100')

args = parser.parse_args()



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    ckpt_path = './ckpt/' + args.scale + '/'
    model = build.Build(ckpt_path, args.scale, args.dataset)
    model.test()

if __name__ == '__main__':
    main()

