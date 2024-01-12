from argparse import ArgumentParser
import pytorch_lightning as pl
#from pytorch_lightning.cli.LightningArgumentParser import LightningArgumentParser
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import yaml
import os

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_train_args():
    #parser = LightningArgumentParser(description="Training script for machine learning models.")
    parser = ArgumentParser()
    # Basic training settings
    parser.add_argument_group('Basic Training Settings')
    parser.add_argument("--load_model", default="", type=str, help="Full path to the model file (with .pth), if resuming from a previous model.")
    parser.add_argument("--wandb", default="", type=str, help="Weights & Biases project name. Leave empty if not using.")
    parser.add_argument("--proj_dir", default="out", type=str, help="Directory for project outputs.")
    parser.add_argument("--random_seed", default="-1", type=int, help="Seed for random number generation. Use -1 for non-deterministic.")

    # Data settings
    parser.add_argument_group('Data Settings')
    parser.add_argument("--data_file", default="", type=str, help="Path to the data file.")
    parser.add_argument("--data_type", default="utf-8", type=str, help="Encoding type of the data.")
    parser.add_argument("--vocab_size", default=0, type=int, help="Vocabulary size, set to 0 for automatic detection, this means auto (for char-level LM and .txt data)")

    # Model Architecture
    model_arch_group = parser.add_argument_group('Model Architecture')
    model_arch_group.add_argument("--ctx_len", default=1024, type=int, help="Context length for the model.")
    model_arch_group.add_argument("--n_layer", default=6, type=int, help="Number of layers in the model.")
    model_arch_group.add_argument("--n_embd", default=512, type=int, help="Embedding dimension size.")
    model_arch_group.add_argument("--dim_att", default=0, type=int, help="Dimension of the attention mechanism.")
    model_arch_group.add_argument("--dim_ffn", default=0, type=int, help="Dimension of the feed-forward network.")
    model_arch_group.add_argument("--pre_ffn", default=0, type=int, help="Use feed-forward network before the first attention layer. it replace first att layer by ffn (sometimes better)")
    model_arch_group.add_argument("--head_qk", default=0, type=int, help="Head query/key trick for the attention mechanism.")
    model_arch_group.add_argument("--tiny_att_dim", default=0, type=int, help="Dimension of the tiny attention mechanism.")
    model_arch_group.add_argument("--tiny_att_layer", default=-999, type=int, help="Layer to apply the tiny attention mechanism.")
    model_arch_group.add_argument("--head_size_a", default=64, type=int, help="Head size for the attention mechanism. You can try larger values for larger models")
    model_arch_group.add_argument("--head_size_divisor", default=8, type=int, help="Divisor for the head size.")
    model_arch_group.add_argument("--my_pos_emb", default=0, type=int, help="Positional embedding setting.")


    # Training Hyperparameters
    training_hyperparam_group = parser.add_argument_group('Training Hyperparameters')
    training_hyperparam_group.add_argument("--epoch_steps", default=1000, type=int, help="Steps per epoch; a mini 'epoch' has [epoch_steps] steps")
    training_hyperparam_group.add_argument("--epoch_count", default=500, type=int, help="Total number of epochs for training; train for this many 'epochs'. will continue afterwards with lr = lr_final")
    training_hyperparam_group.add_argument("--epoch_begin", default=0, type=int, help="Starting epoch for training. If you load a model trained for x 'epochs', set epoch_begin = x")
    training_hyperparam_group.add_argument("--epoch_save", default=5, type=int, help="Frequency of saving the model (in epochs); a.k.a save the model every [epoch_save] 'epochs'")
    training_hyperparam_group.add_argument("--micro_bsz", default=12, type=int, help="Micro batch size (batch size per GPU).")
    training_hyperparam_group.add_argument("--lr_init", default=6e-4, type=float, help="Initial learning rate.")# 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    training_hyperparam_group.add_argument("--lr_final", default=1e-5, type=float, help="Final learning rate.")
    training_hyperparam_group.add_argument("--warmup_steps", default=-1, type=int, help="Number of warmup steps for learning rate; try 50 if you load a model.")
    training_hyperparam_group.add_argument("--beta1", default=0.9, type=float, help="Beta1 hyperparameter for Adam optimizer.")
    training_hyperparam_group.add_argument("--beta2", default=0.99, type=float, help="Beta2 hyperparameter for Adam optimizer; use 0.999 when your model is close to convergence")
    training_hyperparam_group.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon value for Adam optimizer.")
    training_hyperparam_group.add_argument("--grad_cp", default=0, type=int, help="Gradient checkpointing to save VRAM.") # gradient checkpt: saves VRAM, but slower
    training_hyperparam_group.add_argument("--dropout", default=0, type=float, help="Dropout rate; try 0.01 / 0.02 / 0.05 / 0.1.")
    training_hyperparam_group.add_argument("--weight_decay", default=0, type=float, help="Weight decay rate; try 0.1 / 0.01 / 0.001.")
    training_hyperparam_group.add_argument("--weight_decay_final", default=-1, type=float, help="Final weight decay rate.")


    # Advanced Settings
    advanced_settings_group = parser.add_argument_group('Advanced Settings')
    advanced_settings_group.add_argument("--my_pile_version", default=1, type=int, help="Version of the special pile.")# my special pile version
    advanced_settings_group.add_argument("--my_pile_stage", default=0, type=int, help="Stage of the special pile training.")# my special pile mode
    advanced_settings_group.add_argument("--my_pile_shift", default=-1, type=int, help="Shift in the special pile text.")# my special pile mode - text shift
    advanced_settings_group.add_argument("--my_pile_edecay", default=0, type=int, help="Embedding decay in special pile training.")
    advanced_settings_group.add_argument("--layerwise_lr", default=1, type=int, help="Layer-wise learning rate adjustment.") # layerwise lr for faster convergence (but slower it/s)
    advanced_settings_group.add_argument("--ds_bucket_mb", default=200, type=int, help="Deepspeed bucket size in MB.") # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)
    advanced_settings_group.add_argument("--my_sample_len", default=0, type=int, help="Length of samples for training.")
    advanced_settings_group.add_argument("--my_ffn_shift", default=1, type=int, help="Shift in feed-forward network.")
    advanced_settings_group.add_argument("--my_att_shift", default=1, type=int, help="Shift in attention mechanism.")
    advanced_settings_group.add_argument("--load_partial", default=0, type=int, help="Load partial model weights.")
    advanced_settings_group.add_argument("--magic_prime", default=0, type=int, help="Special prime number for training settings.") # The lower prime number closer to Total tokens/2  3n+2
    advanced_settings_group.add_argument("--my_qa_mask", default=0, type=int, help="QA masking setting.")
    advanced_settings_group.add_argument("--my_random_steps", default=0, type=int, help="Random steps in training.")
    advanced_settings_group.add_argument("--my_testing", default='', type=str, help="Testing mode settings.")
    advanced_settings_group.add_argument("--my_exit", default=99999999, type=int, help="Exit condition for training.")
    advanced_settings_group.add_argument("--my_exit_tokens", default=0, type=int, help="Number of tokens for exit condition.")


    # PyTorch Lightning specific arguments
    if pl.__version__[0] == '2':
        plt_specific_group = parser.add_argument_group('PyTorch Lightning Specific Arguments')
        plt_specific_group.add_argument("--accelerator", default="gpu", type=str, help="Accelerator type (e.g., 'gpu', 'cpu').")
        plt_specific_group.add_argument("--strategy", default="auto", type=str, help="Training strategy (e.g., 'auto', 'ddp').")
        plt_specific_group.add_argument("--devices", default=1, type=int, help="Number of devices to use for training.")
        plt_specific_group.add_argument("--num_nodes", default=1, type=int, help="Number of nodes to use for distributed training.")
        plt_specific_group.add_argument("--precision", default="fp16", type=str, help="Precision for training (e.g., 'fp16', 'bf16', 'fp32').")
        #plt_specific_group.add_argument("--accumulate_grad_batches", default=1, type=int, help="Number of batches for gradient accumulation.")
        #plt_specific_group.add_argument("--max_epochs", default=-1, type=int, help="Maximum number of epochs for training.")
        #plt_specific_group.add_argument("--gradient_clip_val", default=1.0, type=float, help="Value for gradient clipping.")
        #plt_specific_group.add_argument("--check_val_every_n_epoch", default=int(1e20), type=int, help="Check validation every N epochs.")
        #plt_specific_group.add_argument("--log_every_n_steps", default=int(1e20), type=int, help="Log training progress every N steps.")
        #plt_specific_group.add_argument("--num_sanity_val_steps", default=0, type=int, help="Number of validation steps to run before training for sanity check.")
        #plt_specific_group.add_argument("--enable_checkpointing", default=False, type=bool, help="Enable model checkpointing.")
        #plt_specific_group.add_argument("--replace_sampler_ddp", default=False, type=bool, help="Replace the default DDP sampler with a custom one.")
        #plt_specific_group.add_argument("--logger", default=False, type=bool, help="Enable logging (e.g., with TensorBoard, WandB).")
    
    parser.add_argument("--yaml_config_path", default="", type=str, help="Path to the YAML configuration file.")
   
    args, unknown = parser.parse_known_args()

    # Load defaults from YAML file
    if args.yaml_config_path and os.path.exists(args.yaml_config_path):
        config = load_config(args.yaml_config_path)
        parser.set_defaults(**config)
        args = parser.parse_args(unknown)

    return parser

#rank_zero_only
def update_args_for_mypile(args):
        #if not os.path.exists(args.proj_dir):
        #    print(f"Creating project directory at: {args.proj_dir}")
        #    os.makedirs(args.proj_dir)
        if args.my_pile_stage > 0:
            magic_prime_bak = args.magic_prime

            if args.my_pile_shift < 0:
                args.my_pile_shift = 0

            if magic_prime_bak > 0:
                args.magic_prime = magic_prime_bak
            if args.my_qa_mask == 2:
                args.epoch_count = 2 * args.magic_prime // 40320
            else:
                args.epoch_count = args.magic_prime // 40320

            args.epoch_steps = 40320 // args.real_bsz
            if hasattr(args, 'epoch_steps') and hasattr(args, 'real_bsz'):
                 assert args.epoch_steps * args.real_bsz == 40320
            
            if args.my_pile_stage >= 2:  # find latest saved model
                list_p = []
                for p in os.listdir(args.proj_dir):
                    if p.startswith("rwkv") and p.endswith(".pth"):
                        p = ((p.split("-"))[1].split("."))[0]
                        if p != "final":
                            if p == "init":
                                p = -1
                            else:
                                p = int(p)
                            list_p += [p]
                list_p.sort()
                max_p = list_p[-1]
                if len(list_p) > 1:
                    args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
                if max_p == -1:
                    args.load_model = f"{args.proj_dir}/rwkv-init.pth"
                else:
                    args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                    if args.warmup_steps < 0:
                        if args.my_pile_stage == 2:
                            args.warmup_steps = 10
                        else:
                            args.warmup_steps = 30
                args.epoch_begin = max_p + 1

        return args    

#if __name__ == "__main__":
#    args = get_train_args()
#    print(args)
