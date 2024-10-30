"""Parameter parsing."""

import argparse
import torch
import os
import math

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run.")

    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--task_type",
                        nargs="?",
                        default="Multiple change",
	                help="task_type:Binary change、Multiple change")
    parser.add_argument("--dataset_name",
                        nargs="?",
                        default="USA",
	                help="dataset_name:China、USA、Hermiston、Benton、River、Bay")
    parser.add_argument("--segments_scale_init",
                        type=int,
                        default=7,
                        help="n_segments_init. Default is China:50,USA:5,Benton:5,Hermiston:10,River:10,Bay:500.")
                        # China:420*140/50,USA:307*241/5
                        # Hermiston:390*200/10,Benton:225*180/5,
                        # River:463*241/10,Bay:600*500/500.                 
    parser.add_argument("--train_ratio",
                        type=float,
                        default=0.005,
	                help="Training set ratio. Default is 0.5%.")
    parser.add_argument("--train_ratio_multi",
                        type=float,
                        default=0.005,
	                help="Training set ratio. Default is 0.5%.")
    parser.add_argument("--train_num",
                        type=float,
                        default=None,
	                help="Number of training samples per class. Default is None.")
    parser.add_argument("--layers",
                        type=int,
                        default=2,
                        help="layer number. Default is 2")
    parser.add_argument("--heads",
                        type=int,
                        default=2,
                        help="heads number. Default is 2")
    parser.add_argument("--path_result",
                        nargs="?",
                        default="./result/",
                        help="path_result.")
    
    return parser.parse_args()
