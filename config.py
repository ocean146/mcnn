import argparse


def get_args():
    parser = argparse.ArgumentParser(description='MCNN')

    parser.add_argument('--dataset',type=str,default='ShanghaiTechA')

    parser.add_argument('--dataset_path',type=str,default=r"C:\Users\ocean\Downloads\datasets\ShanghaiTech\part_A\\")

    parser.add_argument('--save_path',type=str,default='./save_file/')

    parser.add_argument('--print_freq',type=int,default=1)

    parser.add_argument('--device',type=str,default='cuda')

    parser.add_argument('--epochs',type=int,default=600)

    parser.add_argument('--batch_size',type=int,default=1)

    parser.add_argument('--lr',type=float,default=1e-3)

    parser.add_argument('--optimizer',type=str,default='Adam')

    args = parser.parse_args()
    return args