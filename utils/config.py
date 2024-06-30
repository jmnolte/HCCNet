import argparse 

def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Command line arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model architecture
    parser.add_argument("--arch", type=str, default='femto',
                        help="ConvNeXT model architecture. Choices are femto, pico, nano, tiny. Defaults to femto.")
    parser.add_argument("--use-v2", action='store_true',
                        help="Whether to use ConvNext v1 or v2.")
    parser.add_argument("--kernel-size", default=3, type=int,
                        help="Kernel size of convolutional downsampling layers. Defaults to 3.")
    parser.add_argument("--out-dim", default=4096, type=int,
                        help="Output dimensionality of projection head. Defaults to 4096")
    parser.add_argument("--seq-length", default=7, type=int,
                        help="Length of transformer encoder input sequence. Defaults to 7.")
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    
    # Training specifics
    parser.add_argument("--loss-fn", type=str, default='bce',
                        help="Loss function to use. Can dino, or BCE.")
    parser.add_argument("--distributed", action='store_true',
                        help="Whether to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Whether to enable automated mixed precision training.")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights.")
    parser.add_argument("--partial", action='store_true',
                        help="Flag to only use DINO pretrained weights.")
    parser.add_argument("--k-folds", default=5, type=int, 
                        help="Number of folds to use in cross validation. Defaults to 5.")
    parser.add_argument("--max-delta", default=3, type=int, 
                        help="Maximum time before diagnosis. Defaults to 3.")
    parser.add_argument("--num-seeds", default=3, type=int, 
                        help="Number of random seeds to run training over. Defaults to 3.")

    # Hyperparameters
    parser.add_argument("--num-steps", default=100, type=int, 
                        help="Number of steps to train for. Defaults to 100.")
    parser.add_argument("--warmup-steps", default=10, type=int, 
                        help="Number of warmup steps. Defaults to 10.")
    parser.add_argument("--log-every", default=10, type=int,
                        help="Interval to log model results. Defaults to 10.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training. Defaults to 4.")
    parser.add_argument("--effective-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
    parser.add_argument("--min-learning-rate", default=1e-6, type=float, 
                        help="Final learning rate. Defaults to 1e-6.")
    parser.add_argument("--label-smoothing", default=0.0, type=float, 
                        help="Label smoothing to use for training. Defaults to 0.")
    parser.add_argument("--gamma", type=int, default=2,
                        help="Gamma parameter to use for focal loss. Defaults to 2.")
    parser.add_argument("--weight-decay", default=0.05, type=float, 
                        help="Initial weight decay value. Defaults to 0.05.")
    parser.add_argument("--max-weight-decay", default=0.5, type=float, 
                        help="Final weight decay value. Defaults to 0.5.")
    parser.add_argument("--stochastic-depth", default=0.0, type=float, 
                        help="Stochastic depth rate to use for training. Defaults to 0.")
    parser.add_argument("--dropout", default=0.0, type=float, 
                        help="Dropout rate to use for training. Defaults to 0.")
    parser.add_argument("--epsilon", default=1e-5, type=float, 
                        help="Epsilon value to use for norm layers. Defaults to 1e-5.")
    parser.add_argument("--global-crop-size", default=72, type=int, 
                        help="Global crop size to use. Defaults to 72.")
    parser.add_argument("--local-crop-size", default=48, type=int, 
                        help="Local crop size to use. Defaults to 48.")
          
    # DINO pre-training specifics
    parser.add_argument("--norm-last-layer", action='store_true',
                        help="Whether to weight normalize the last layer of the DINO head.")
    parser.add_argument("--teacher-momentum", default=0.9995, type=float, 
                        help="Inital momentum value to update teacher network with EMA. Defaults to 0.9995.")
    parser.add_argument("--teacher-temp", default=0.04, type=float, 
                        help="Final (i.e., after warmup) teacher temperature value. Defaults to 0.04")
    parser.add_argument("--teacher-warmup-temp", default=0.04, type=float, 
                        help="Initial teacher temperature value. Defaults to 0.04")

    # File paths and auxiliaries
    parser.add_argument("--mod-list", default=MOD_LIST, nargs='+', 
                        help="List of modalities to use for training. Can be DWI_b0, DWI_b150, DWI_b400, DWI_b800, T1WI, T1A, T1D, T1V, T1W_IP, T1W_OOP, T2W_TES, T2W_TEL")
    parser.add_argument("--seed", default=1234, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--suffix", default='MedNet', type=str, 
                        help="File suffix for identification")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR, type=str, 
                        help="Path to weights directory")
    return parser.parse_args()

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MOD_LIST = ['DWI_b0','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'