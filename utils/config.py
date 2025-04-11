import argparse
# Training settings
parser = argparse.ArgumentParser(description='UCCH implementation')

#########################
#### data parameters ####
#########################
parser.add_argument("--data_name", type=str, default="mirflickr25k", # mirflickr25k
                    help="mirflickr25k or iapr")
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='HRCH')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='HRCH')
parser.add_argument('--arch', type=str,default='HRCH')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--wd', type=float, default=1e-6)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--ls', type=str, default='linear', help='lr scheduler')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument("--seed", type=int, default=3407, help='seed for initializing training and data partition')

# ranking
parser.add_argument('--shift', type=float, default=1)
parser.add_argument('--margin', type=float, default=.2, help='ma')

# Hierarchical
parser.add_argument('--tau', type=float, default=0.12)
parser.add_argument('--taup', type=float, default=0.12)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--alpha', type=float, default=.4,help='balance the hierarchical loss and ranking loss')
parser.add_argument('--ins', type=float, default=0.8,help='weight of instance loss')
parser.add_argument('--pro', type=float, default=1.0,help='weight of prototype loss')
parser.add_argument("--beta", type=float, default=0.5,help='balance image and text loss')
parser.add_argument("--entroy", type=float, default=0.05,help='weight of entropy loss')
parser.add_argument("--qua", type=float, default=0.01,help='weight of quantization loss')
parser.add_argument('--cluster_num', type=str, default="5000,4000,3000,2000")

# RevCol
parser.add_argument("--droprate", type=float, default=0.0,help="dropout rate")
parser.add_argument('--clip', type=bool, default=True)
# parser.add_argument("--hidden", type=int, default=0)
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--ld', type=int, default=1,help="constanct C")
parser.add_argument('--layers', type=str, default="2,2,4,2")
parser.add_argument('--drop_path', type=float, default=0.01)
parser.add_argument("--feature_save", type=bool, default=True)

#path
parser.add_argument("--reversible_path",type=str,default="revcol_tiny_1k.pth")
# parser.add_argument("--test_path",type=str,default="mirflickr25k_HRCH_128_best_checkpoint.t7")
parser.add_argument("--resume",type=bool,default=False,help='path to latest checkpoint (default: none)')
parser.add_argument('--bit', type=int, default=128, help='output shape')
parser.add_argument("--test_path",type=str,default="mirflickr25k_128_checkpoint.t7",help='''--cluster_num 5000,4000,3000,2000 --layers 2,2,4,2 --resume Ture for all bits in mirflicker25k''')

args = parser.parse_args()
