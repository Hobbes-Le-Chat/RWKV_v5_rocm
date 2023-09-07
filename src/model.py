########################################################################################################
# The RWKV Language Model
# Documentation: https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# =================== IMPORTS ===================
# Standard Library Imports
import gc, math, os
from random import randint
from typing import List, Optional,Tuple

# Third-Party Libraries
import numpy as np
from packaging import version

# PyTorch Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Lightning Imports
import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

# DeepSpeed and Wandb Imports
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed.runtime.lr_schedules

# Lets import the wandb
import wandb

# =================== GLOBAL VARIABLES & SETTINGS ===================
# Define script and CUDA directories
try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    CUDA_DIR = os.path.abspath(os.path.join( SCRIPT_DIR, "../hip"))
except Exception as e:
    print(f"Error while getting directories: {e}")
    SCRIPT_DIR, CUDA_DIR = "", ""

# Let's Check the torch version rocm
def is_torch_version_above(required_version: str) -> bool:
    """Check if the installed PyTorch version is above the required version."""
    torch_version = version.parse(torch.__version__.split('+')[0])
    return torch_version >= version.parse(required_version)

IS_TORCH_2_1 = is_torch_version_above("2.0.9999")

# Get JIT / torch compile options from the environment
RWKV_JIT_ON = os.getenv("RWKV_JIT_ON", "1").lower() in ("1", "true", "yes")
RWKV_TORCH_COMPILE = os.getenv("RWKV_TORCH_COMPILE", f"0").lower() in ("1", "true", "yes")
RWKV_TORCH_RUN_MODE = None


# =================== RUNTIME SETTINGS ===================
# Set runtime mode and configuration based on environment variables

def set_no_op(x):
    """No operation function for default behavior."""
    return x

# Initialize runtime mode and settings
#RWKV_TORCH_RUN_MODE = None
JITModClass = nn.Module
JITModMethod = set_no_op  # Replacing lambda with a named function
JITFunction = set_no_op  # Replacing lambda with a named function
TCompileMax = set_no_op  # Replacing lambda with a named function
TCompileBaseline = set_no_op  # Replacing lambda with a named function
TCompileDisable = set_no_op  # Replacing lambda with a named function

# Check for torch compile support
if RWKV_TORCH_COMPILE: 
    # Now with hip it is alway recommended that you do the compilation of torch from the git and the latest rocm. 
    #However this my breake the code since the rocm compativility with torch does not always match.
    # The torch +rocm 5.4.3 is what worked for me. 
    RWKV_TORCH_RUN_MODE = "torch-compile"
    # (existing comments and code)
    TCompileMax = lambda x: torch.compile(x, fullgraph=True)
    TCompileDisable = torch._dynamo.disable
# Check for JIT support
elif RWKV_JIT_ON:
    RWKV_TORCH_RUN_MODE = "torch-jit"
    JITModClass = torch.jit.ScriptModule
    JITModMethod = torch.jit.script_method
    JITFunction = torch.jit.script
# Fallback to native torch
else:
    RWKV_TORCH_RUN_MODE = "torch-native"

print(f"[RWKV.model] Running RWKV model using '{RWKV_TORCH_RUN_MODE}' with torch '{torch.__version__}'")

# =================== DEEPSPEED CHECKPOINTING ===================
@TCompileDisable
def deepspeed_checkpoint(*args, **kwargs):
    """Wrap deepspeed checkpointing with TCompileDisable."""
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

# =================== RWKV: STATE BLOCKS ===================
class TimeMixState:
    """Stores the state for TimeMix operation."""
    
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    """Stores the state for ChannelMix operation."""
    
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:
    """Stores the states for both TimeMix and ChannelMix operations."""
    
    def __init__(self, time_mix_state: TimeMixState, channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:
    """Manages a list of BlockState instances."""

    def __init__(self, shift_states: torch.Tensor, wkv_states: torch.Tensor):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @classmethod #CHANGE
    def create(cls, N: int, B: int, C: int, n_head: int, head_size: int, device: str, dtype: str) -> 'BlockStateList':
        """Create and initialize a new BlockStateList."""
        result = cls.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result

    @classmethod
    def empty(cls, N: int, B: int, C: int, n_head: int, head_size: int, device: str, dtype: str) -> 'BlockStateList':
        """Create an empty BlockStateList."""
        wkv_states = torch.empty((N, B, n_head, head_size, head_size), device=device, dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return cls(shift_states, wkv_states)

    def __getitem__(self, layer: int) -> BlockState:
        """Retrieve the BlockState for a specific layer."""
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        """Set the BlockState for a specific layer."""
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state



# =================== RWKV: RWKV Time-mix + RWKV Channel-mix ===================

class RWKV_TimeMix(JITModClass):
    """Implements the RWKV TimeMix attention mechanism."""

    DEFAULT_CHUNK_LEN = 512  # Default optimized chunk length

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, chunk_len=DEFAULT_CHUNK_LEN):
        """Initialize the RWKV TimeMix layer."""
        super().__init__()
        self.init_parameters(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.init_fancy_init(layer_id, n_layer, n_embd, n_head)
        self.init_layers(n_embd, dim_att)
        self.chunk_len = chunk_len

    def init_parameters(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att):
        """Initialize basic parameters."""
        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id
        self.n_head = n_head
        self.head_size = head_size

    def init_fancy_init(self, layer_id, n_layer, n_embd, n_head):
        """Perform the 'fancy' initialization for this layer."""
        
        with torch.no_grad():  # fancy init
            # Calculate ratios based on layer IDs
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            
            # Create a tensor with shape (1, 1, n_embd)
            ddd = torch.ones(1, 1, n_embd)
            
            # Fill tensor based on n_embd
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # Initialize time_mix parameters
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # Initialize time_decay parameter
            decay_speed = torch.ones(n_head)
            for h in range(n_head):
                decay_speed[h] = -8 + 7 * (h / (n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # Additional parameter initializations (V5-R2 changes)
            self.time_faaaa = nn.Parameter(torch.ones(n_head) * 0.05)
        

    def init_layers(self, n_embd, dim_att):
        """Initialize the neural network layers used in this module."""
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.key = nn.Linear(n_embd, dim_att, bias=False)
        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, n_embd)

    @JITModMethod
    def _forward_rkv_chunk(self, x, B, TT, last_state: TimeMixState):
        """
        Forward pass for computing the 'r', 'k', and 'v' values.

        Parameters:
            x (torch.Tensor): Input tensor.
            B (torch.Tensor): Batch size.
            TT (torch.Tensor): Time steps.
            last_state (TimeMixState): The last time-mix state.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: 'r', 'k', and 'v' values
        """
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        
        # Compute new values for xk, xv, xr based on mixing coefficients
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Transform xr, xk, and xv using linear layers
        r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
        k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
        v = self.value(xv).view(B, TT, self.n_head, self.head_size).transpose(1, 2)                 # BTC -> BHTS

        return r, k, v

    def _forward_wkbs_chunk(self, T: int, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Calculate w, wk, wb, and ws based on the input tensors and decay settings."""
        H = self.n_head
        
        # Calculate decay weights
        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)
        
        # Additional change in V5-R2
        u = self.time_faaaa.float().unsqueeze(-1)
        
        ws = w.pow(T).reshape(1, H, 1, 1)
        ind = torch.arange(T - 1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)

        wk = w.reshape(1, H, 1, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T - 1:].reshape(1, H, T, T)

        # Ensure the dtype matches
        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        return w, wk, wb, ws

    @JITModMethod
    def _forward_state_chunk(self, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                            w: torch.Tensor, wk: torch.Tensor, wb: torch.Tensor, 
                            ws: torch.Tensor, x_l: torch.Tensor, last_state: TimeMixState):
        """Perform the state forward pass."""
        B, H, TT, S = r.size()
        T = TT
        
        s = last_state.wkv_state

        # Ensure dtype compatibility
        if r.dtype == torch.bfloat16 and s.dtype != torch.bfloat16:
            s = s.contiguous().to(torch.bfloat16)
        
        # Initialize output tensor
        x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype)
        
        for i in range(TT // T):
            rr = r[:, :, i * T:i * T + T, :]
            kk = k[:, :, :, i * T:i * T + T]
            vv = v[:, :, i * T:i * T + T, :]

            x[:, :, i * T:i * T + T, :] = ((rr @ kk) * w) @ vv + (rr @ s) * wb
            s = ws * s + (kk * wk) @ vv

        x = x.transpose(1, 2).contiguous().view(B * TT, H * S)
        x = self.ln_x(x).view(B, TT, H * S)

        return self.output(x), TimeMixState(x_l, s)

    def _forward_chunk(self, x: torch.Tensor, last_state: TimeMixState):
        """Perform the forward pass for a chunk of data."""
        B, TT, C = x.size()
        B = torch.tensor(B, device=x.device, dtype=torch.int32)
        TT = torch.tensor(TT, device=x.device, dtype=torch.int32)

        r, k, v = self._forward_rkv_chunk(x, B, TT, last_state)
        w, wk, wb, ws = self._forward_wkbs_chunk(TT, r, k, v)
        
        return self._forward_state_chunk(r, k, v, w, wk, wb, ws, x[:, -1], last_state)




    @TCompileMax
    def forward(self, x: torch.Tensor, last_state: TimeMixState) -> Tuple[torch.Tensor, TimeMixState]:
        """Forward pass for the RWKV_TimeMix block."""
        B, TT, C = x.size()# Get the x sizing
        chunk_len = self.chunk_len
        x_logits = torch.zeros(B, TT, C, device=x.device, dtype=x.dtype)

        # Process each chunk of the sequence
        for i in range(0, TT, chunk_len):# Split the input by TT chunks
            x_chunk = x[:, i:i + chunk_len, :]
            chunk_logits, last_state = self._forward_chunk(x_chunk, last_state)
            x_logits[:, i:i + chunk_len, :] = chunk_logits

        return x_logits, last_state


class RWKV_ChannelMix(JITModClass):
    """Channel mixing layer for the RWKV model."""

    def __init__(self, layer_id: int, n_layer: int, n_embd: int, dim_ffn: int):
        """Initialize the RWKV_ChannelMix layer.
        
        Parameters:
            layer_id: The ID of this layer.
            n_layer: The total number of layers.
            n_embd: The dimensionality of the embeddings.
            dim_ffn: The dimensionality of the feed-forward network.
        """
        super().__init__()

        # Fancy initialization of time_mix
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        # Initialize linear layers
        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    @JITModMethod
    @TCompileMax
    def forward(self, x: torch.Tensor, last_state: ChannelMixState) -> Tuple[torch.Tensor, ChannelMixState]:
        """Forward pass for the RWKV_ChannelMix layer.
        
        Parameters:
            x: The input tensor of shape [batch_size, sequence_length, feature_dim].
            last_state: The previous state of this layer.
        
        Returns:
            A tuple containing:
                - The output tensor
                - The new state
        """
        xx = torch.cat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(x[:, -1])


class Block(nn.Module):
        """Defines a block in the RWKV model."""

        def __init__(self, layer_id: int, n_layer: int, n_embd: int, n_head: int, 
                    head_size: int, dropout: float, dim_att: int, dim_ffn: int):
            super().__init__()
            self.layer_id = layer_id
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

            if self.layer_id == 0:
                self.ln0 = nn.LayerNorm(n_embd)

            self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
            self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)
            self.dropout = dropout

            if dropout > 0:
                self.drop0 = nn.Dropout(p=dropout)
                self.drop1 = nn.Dropout(p=dropout)

        def forward(self, x: torch.Tensor, last_state: BlockState) -> Tuple[torch.Tensor, BlockState]:
            """Forward pass for the block."""
            if self.layer_id == 0:
                x = self.ln0(x)

            att_out, att_state = self.att(self.ln1(x), last_state.time_mix_state)

            if self.dropout > 0.0:
                x = self.drop0(x + att_out)
                ffn_out, ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
                x = self.drop1(x + ffn_out)
            else:
                x = x + att_out
                ffn_out, ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
                x = x + ffn_out

            return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y, token_amount, current_mask):
        # Currently (8th July 2023), save_for_backward, causes an issue with
        # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
        # 
        # Due to L2Wrap being a major hotspot, we should monitor this for future support.
        # so that once its resolved, we can include the L2Wrap step in the torch.compile path
        #
        # See also:
        # - checkpointed_step
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        ctx.current_mask = current_mask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        token_amount = ctx.token_amount
        factor = 1e-4 / token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        gy = gy * ctx.current_mask[:, None][None, :]
        return grad_output, gy, None, None

# =================== RWKV: Core RWKV module ===================

class RWKV(L.LightningModule):


    from typing import Optional, List

class RWKV(L.LightningModule):
    def __init__(self, 
                 load_model: str, # Model file path to load from
                 # Model size settings, which we either
                 # "auto detect", or use the user specified settings
                 n_embd: int = -1,
                 n_layer: int = -1,
                 vocab_size: int = -1,
                 ctx_len: int = 2048,
                 # Context length schedule
                 ctx_len_cutoffs: List[int] = [],
                 ctx_len_warmup_steps: List[int] = [],
                 # Learning rate schedule
                 # use only target_lr_init / lr_init
                 # to configure a constant learning rate
                 lr_init: float = -1.0,
                 lr_final: float = -1.0,
                 lr_period: int = -1,
                 lr_period_type: str = 'epoch',
                 # Dropout rate
                 dropout: float = 0.0,
                 # Adam optimizer settings
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 adam_eps: float = 1.0e-08,
                 weight_decay: float = 0.01,
                 warmup_steps: int = -1,
                 # Backprop settings
                 grad_cp: bool = True,
                 bptt_learning: bool = True,
                 bptt_learning_range: int = -1,
                 bptt_truncated_learning: bool = False,
                 layerwise_lr: bool = True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 substep_cuda_cache_clear: bool = False,
                 substep_logging: bool = False,
                 torch_set_float32_matmul_precision:str = 'high'
                 ):
        
        # Lets save everything in one shot
        # (this is used for wandb logging)
        self.setup_args = locals()

        # Lets delete objects.
        del self.setup_args["self"]
        del self.setup_args["__class__"]

        # Setup the parent class
        super().__init__()
        # Step 1: Initialize Class Members
        # This method sets up the initial values of the class variables, 
        # which include model parameters and hyperparameters.
        self._initialize_class_members(load_model, n_embd, n_layer, vocab_size,ctx_len,
                                       ctx_len_cutoffs,ctx_len_warmup_steps,lr_init,lr_final,
                                       lr_period,lr_period_type,dropout,beta1,
                                       beta2,adam_eps,weight_decay,warmup_steps,grad_cp,
                                       bptt_learning,bptt_learning_range,bptt_truncated_learning,
                                       layerwise_lr,dim_att,dim_ffn,substep_cuda_cache_clear,
                                       substep_logging,torch_set_float32_matmul_precision)

        # Step 2: Load Model Weights
        # If a pre-trained model is specified, this method loads its weights.
        # If the model is being initialized from scratch, this step is skipped.
        self._load_model_weights()

        # Step 3: Compute Model Sizes
        # This method computes the sizes of the model layers. If these are not provided,
        # it automatically computes them based on the loaded pre-trained model.        
        self._compute_model_sizes()
        
        # Step 4: Build Model
        # This method constructs the neural network model using PyTorch modules.
        # It sets up the layers, embeddings, and other components.
        self._build_model()

    def _initialize_class_members(self,
                                load_model: str,
                                n_embd: int = -1,
                                n_layer: int = -1,
                                vocab_size: int = -1,
                                ctx_len: int = 2048,
                                ctx_len_cutoffs: List[int] = [],
                                ctx_len_warmup_steps: List[int] = [],
                                lr_init: float = -1.0,
                                lr_final: float = -1.0,
                                lr_period: int = -1,
                                lr_period_type: str = 'epoch',
                                dropout: float = 0.0,
                                beta1: float = 0.9,
                                beta2: float = 0.99,
                                adam_eps: float = 1.0e-08,
                                weight_decay: float = 0.01,
                                warmup_steps: int = -1,
                                grad_cp: bool = True,
                                bptt_learning: bool = True,
                                bptt_learning_range: int = -1,
                                bptt_truncated_learning: bool = False,
                                layerwise_lr: bool = True,
                                dim_att: Optional[int] = None,
                                dim_ffn: Optional[int] = None,
                                substep_cuda_cache_clear: bool = False,
                                substep_logging: bool = False,
                                torch_set_float32_matmul_precision: str = 'high'
                                ):
        """
        Initializes the class members with the provided parameters.
        """
        try: 

            self.load_model = load_model
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.vocab_size = vocab_size
            self.ctx_len = ctx_len
            self.ctx_len_cutoffs = ctx_len_cutoffs
            self.ctx_len_warmup_steps = ctx_len_warmup_steps
            self.lr_init = lr_init
            self.lr_final = lr_final
            self.lr_period = lr_period
            self.lr_period_type = lr_period_type
            self.dropout = dropout
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.weight_decay = weight_decay
            self.warmup_steps = warmup_steps
            self.grad_cp = grad_cp
            self.bptt_learning = bptt_learning
            self.bptt_learning_range = bptt_learning_range
            self.bptt_truncated_learning = bptt_truncated_learning
            self.layerwise_lr = layerwise_lr
            self.dim_att = dim_att or self.n_embd  # Default to n_embd if not provided
            self.dim_ffn = dim_ffn or self.n_embd * 4  # Default to n_embd * 4 if not provided
            self.substep_cuda_cache_clear = substep_cuda_cache_clear
            self.substep_logging = substep_logging
            self.torch_set_float32_matmul_precision = torch_set_float32_matmul_precision
            print("Initialization successful.")
            
        except Exception as e:
            print(f"Error in initialization: {e}")


    def _load_model_weights(self):
        """Load model weights if a pre-trained model is specified."""
        if self.load_model != ".//<#|=@%!$init_model$!%@=|#>//.":
            if not os.path.isfile(self.load_model):
                raise ValueError(f"load_model file '{self.load_model}' does not exist")
            self.model_weights = torch.load(self.load_model, map_location='cpu')
            self.model_keys = list(self.model_weights.keys())

    def _compute_model_sizes(self):
        """Compute or validate the sizes for the model layers and embeddings."""
        if self.n_layer < 0:
            max_block_id = max(int(x.split('.')[1]) for x in self.model_keys if 'blocks.' in x)
            self.n_layer = max_block_id + 1
        if self.n_embd < 0:
            self.n_embd = self.model_weights['head.weight'].shape[1]
        if self.vocab_size < 0:
            self.vocab_size = self.model_weights['head.weight'].shape[0]

    def _compute_head_size_and_matmul_precision(self):
        """Compute the head size and number of heads, and set matrix multiplication precision."""
        head_size = 64
        n_head = self.n_embd // head_size
        assert self.n_embd % n_head == 0, f"n_embd must be divisible by head_size ({self.head_size})"
        self.n_head = n_head
        self.head_size = head_size

        if self.torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(self.torch_set_float32_matmul_precision)
        

    def _build_model(self):
        """Construct the RWKV model."""
        self._compute_head_size_and_matmul_precision()
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.blocks = nn.ModuleList([
            Block(i, self.n_layer, self.n_embd, self.n_head, self.head_size,
                  self.dropout, self.dim_att, self.dim_ffn) for i in range(self.n_layer)
        ])
        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        if self.dropout > 0:
            self.drop0 = nn.Dropout(p=self.dropout)
        if self.model_weights is not None:
            self.load_state_dict(self.model_weights)
            del self.model_weights
            gc.collect()


    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""

        # Check if Backpropagation Through Time (BPTT) is disabled
        if not self.bptt_learning:
            if self.deepspeed_stage >= 2 or self.deepspeed_offload:
                print("[WARNING]: Enable bptt_learning with deepspeed 2/3/offloading to avoid exceptions when training with large datasets.")
        else:
            if self.trainer.num_devices > 1 and self.bptt_learning_range <= 0:
                print("[WARNING]: Consider using bptt_learning_range=1 to improve performance with multiple GPUs.")

        # Initialize learning rates
        lr_init, lr_final = self.lr_init, self.lr_final
        if lr_final < 0:
            lr_final = lr_init

        # Log learning rate settings
        if self.trainer.local_rank == 0:
            print(f"\n[RWKV.model] Configuring optimizer with\n"+
                f"    - lr_init:  {lr_init:.3e}\n"+
                f"    - lr_final: {lr_final:.3e}\n")

            # Update WANDB, if applicable
            if wandb.run is not None:
                model_args = {**self.setup_args, "__lr_init": lr_init, "__lr_final": lr_final}
                wandb.config.update({"model": model_args})

        # Setup layer-wise learning rates, if applicable
        optim_groups = self._create_optimization_groups(lr_init)

        # Create Adam optimizer
        optimizer = self._create_adam_optimizer(optim_groups, lr_init)

        # Throw if wramup_steps and lr_period are both set (not supported)
        if self.warmup_steps > 0 and self.lr_period > 0:
            raise ValueError(
                "Use either warmup_steps or lr_period, not both.")

        # Create learning rate scheduler
        lr_scheduler = self._create_lr_scheduler(optimizer, lr_init, lr_final)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }
    



    def _create_optimization_groups(self, lr_init):
        """Create optimization groups for layer-wise learning rates."""
        if self.layerwise_lr:
            lr_1x, lr_2x, lr_3x = set(), set(), set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n or "time_faaaa" in n:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)

            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "lr": 1.0 * lr_init},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "lr": 2.0 * lr_init},
            ]
        else:
            optim_groups = [
                {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0}
            ]
        return optim_groups

    def _create_adam_optimizer(self, optim_groups, lr_init):
        """Create an Adam optimizer."""
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(
                optim_groups, lr=lr_init, betas=(self.beta1, self.beta2), eps=self.adam_eps,
                bias_correction=True, adamw_mode=False, weight_decay=self.weight_decay, amsgrad=False
            )
        else:
            optimizer = FusedAdam(
                optim_groups, lr=lr_init, betas=(self.beta1, self.beta2), eps=self.adam_eps,
                bias_correction=True, adam_w_mode=False, weight_decay=self.weight_decay, amsgrad=False
            )
        return optimizer

    def _create_lr_scheduler(self, optimizer, lr_init, lr_final):
        """Create a learning rate scheduler."""
        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear'
            )
            return {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        else:
            if lr_init == lr_final:
                return None

            lr_total_step = self._calculate_total_steps_for_lr()
            
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor= lr_final / lr_init,
                total_iters=lr_total_step
            )

            return {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }

    def _calculate_total_steps_for_lr(self):
        """Calculate the total number of steps for learning rate decay."""
        if self.lr_period == -1:
            trainer_max_step = self.trainer.max_steps
            trainer_max_epoch = self.trainer.max_epochs
            if trainer_max_step > 0:
                return trainer_max_step
            elif trainer_max_epoch > 0:
                return trainer_max_epoch * self.num_step_per_epoch()
            else:
                print("Warning: max_step/max_epoch not set, assuming 10 epoch for learning rate shift.")
                return 10 * self.num_step_per_epoch()
        else:
            if self.lr_period_type == "step":
                return self.lr_period
            elif self.lr_period_type == "epoch":
                return self.lr_period * self.num_step_per_epoch()
            else:
                raise ValueError(f"lr_period_type {self.lr_period_type} not supported.")


    def num_step_per_epoch(self) -> int:
        """
        Calculate the estimated number of steps per epoch based on the training configuration.
        We have to compute the number of steps per epoch ourselves
        as this value is not provided directly by pytorch lightning
        Returns:
            int: The estimated number of steps per epoch.
        """

        # Compute the estimated total number of steps for the entire training.
        # This must be called before accessing `self.trainer.train_dataloader` to avoid a bug.
        estimated_stepping_batches = self.trainer.estimated_stepping_batches

        # Check if max_epochs is set in the trainer configuration.
        max_epochs = self.trainer.max_epochs
        if max_epochs > 0:
            return estimated_stepping_batches // max_epochs

        # If max_epochs is not set, fallback to using the length of the train dataloader.
        train_dataloader = self.trainer.train_dataloader or self.trainer.fit_loop._data_source.dataloader()

        # Calculate the number of steps per epoch based on the dataset size and configuration.
        dataset_size = len(train_dataloader)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size // (self.trainer.accumulate_grad_batches * num_devices)
        
        return num_steps

    @property
    def deepspeed_offload(self) -> bool:
        """
        Check if DeepSpeed offload is enabled in the training configuration.

        Returns:
            bool: True if DeepSpeed offload is enabled, False otherwise.
        """

        # Fetch the trainer's strategy configuration.
        strategy = self.trainer.strategy

        # Check if the strategy is of type DeepSpeedStrategy.
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "offload_optimizer" in cfg or "offload_parameters" in cfg

        return False

    @property
    def deepspeed_stage(self) -> int:
        """
        Get the DeepSpeed optimization stage from the training configuration.

        Returns:
            int: The DeepSpeed optimization stage if specified, -1 otherwise.
        """

        # Fetch the trainer's strategy configuration.
        strategy = self.trainer.strategy

        # Check if the strategy is of type DeepSpeedStrategy.
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("stage", -1)

        return -1

    @TCompileBaseline
    def forward(self, idx: torch.Tensor, 
                last_shift_states: Optional[torch.Tensor] = None,
                last_wkv_states: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.

        Parameters:
            idx (torch.Tensor): Input tensor.
            last_shift_states (torch.Tensor, optional): Previous shift states. Defaults to None.
            last_wkv_states (torch.Tensor, optional): Previous wkv states. Defaults to None.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Output tensor, new shift states, new wkv states.
        """
        # Check for valid input tensor size
        self._check_input_size(idx)
        
        # Apply embedding and initial dropout
        x = self._initial_preprocessing(idx)
        
        # Prepare block states
        cur_bs_list = self._prepare_block_states(last_shift_states, last_wkv_states, x)
        
        # Forward pass through blocks
        new_states = self._forward_through_blocks(x, cur_bs_list)
        
        # Apply output layer normalization and head
        x = self._apply_output_layers(x)
        
        return x, new_states.shift_states, new_states.wkv_states
    

    def _check_input_size(self, idx: torch.Tensor):
        """Check if the input tensor size is valid."""
        _, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

    def _initial_preprocessing(self, idx: torch.Tensor) -> torch.Tensor:
        """Apply embedding and initial dropout to the input tensor."""
        x = self.emb(idx)
        if self.dropout > 0.0:
            x = self.drop0(x)
        return x
    
    def _prepare_block_states(self, last_shift_states: torch.Tensor, last_wkv_states: torch.Tensor, x: torch.Tensor):
        """Prepare the block states for the forward pass."""
        B = x.size(0)
        if last_shift_states is None:
            return BlockStateList.create(
                self.n_layer, B, self.n_embd, 
                self.n_head, self.head_size,
                x.device, x.dtype
            )
        return BlockStateList(last_shift_states, last_wkv_states)
    
    def _forward_through_blocks(self, x: torch.Tensor, cur_bs_list):
        """Forward pass through all the blocks."""
        new_states = BlockStateList.empty(
            self.n_layer, *x.size(), self.n_head, 
            self.head_size, x.device, x.dtype
        )
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.grad_cp:
                x, new_state = deepspeed_checkpoint(
                    block, x, last_state
                )
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state
        return new_states
    
    def _apply_output_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the final layer normalization and head to the tensor."""
        x = self.ln_out(x)
        return self.head(x)
    
    def manual_backward(self, loss: torch.Tensor, *args, **kwargs):
        """
        Manually calculate the backward pass.
        
        Parameters:
            loss (torch.Tensor): Loss tensor.
        """
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            # self._verify_is_manual_optimization("manual_backward")
            self.trainer.strategy.backward(loss, None, *args, **kwargs)








    def compute_loss(self, batch, batch_idx, is_training_run: bool):
        """
        Compute the loss for a given batch.
        
        Parameters:
            batch (dict): Dictionary containing batch data.
            batch_idx (int): Batch index.
            is_training_run (bool): Flag to check if the current run is for training.
        Returns:
            The loss for the batch.    
        """
        
        seq, seq_mask = self._preprocess_input(batch)
        if is_training_run:
            seq, seq_mask = self._handle_training_cutoff(seq, seq_mask)
        
        total_loss = self._execute_forward_backward(seq, seq_mask, is_training_run, batch_idx)
        
        return total_loss

    def _preprocess_input(self, batch):
        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        seq_mask = batch.get('attention_mask', None)
        
        if seq_mask is None or seq_mask.ndim != 2:
            seq_mask = torch.ones_like(seq[:, 1:])
        
        return seq, seq_mask
    
    def _handle_training_cutoff(self, seq, seq_mask):
        prev_step = 0
        for i in range(min(len(self.ctx_len_warmup_steps), len(self.ctx_len_cutoffs))):
            step = self.ctx_len_warmup_steps[i]
            len_cut = self.ctx_len_cutoffs[i]

            if prev_step <= self.global_step < step and len_cut < seq.shape[1] - 1:
                pos = randint(0, seq.shape[1] - len_cut - 1)
                seq = seq[:, :pos + len_cut + 1]
                seq_mask = seq_mask[:, :pos + len_cut + 1]
                seq_mask[:, :pos] = 0
                break
            prev_step = step
        return seq, seq_mask

    def _execute_forward_backward(self, seq, seq_mask, is_training_run, batch_idx): 
        idx, targets = seq[:, :-1], seq[:, 1:]
        B, T = idx.shape
        C = self.n_embd
        total_mask_sum = torch.sum(seq_mask)

        if total_mask_sum == 0:
            return 0
        
        def checkpointed_step(idx, targets, mask, prev_loss, last_shift_states,
                            last_wkv_states, prev_steps):
            logits, new_shift_states, new_wkv_states = self(
                idx, last_shift_states, last_wkv_states)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.view(-1),
                                    reduction="none")

            submask = mask.view(-1)[:loss.shape[0]]
            submask_sum = torch.sum(submask)
            loss = torch.sum(loss * submask) / total_mask_sum  
            loss = L2Wrap.apply(loss, logits, total_mask_sum, submask)
            new_steps = prev_steps + submask_sum
            new_loss = prev_loss + loss
            return new_loss, new_shift_states, new_wkv_states, new_steps
        total_loss = torch.tensor(
            0, dtype=self.emb.weight.dtype).requires_grad_()
        steps = 0
        states = BlockStateList.create(self.n_layer, B, C, 
                                    self.n_head, self.head_size,
                                    seq.device, self.emb.weight.dtype)
        segment_count = math.ceil(T / self.ctx_len)

        do_bptt_learning = self.bptt_learning and is_training_run

        
        if do_bptt_learning:

                gradient_accumulation_steps = max(1, self.trainer.accumulate_grad_batches)
                optimizer = self.optimizers()
                cur_device = self.device

                segment_size = min(math.ceil(T / segment_count), self.ctx_len)

                # Dummy 2D tenros of shape [1,1], are used to do "dummy checkpoint/forward/backprop" to keep everything in sync
                dummy_2d_zero = torch.tensor([[0]], dtype=torch.long, device=cur_device)

                
                if self.trainer.num_devices > 1:
                    if self.bptt_learning_range <= 0:
                        # We perform forward/backward on the shared max segment count across all GPUs
                        forward_segment_count  = self.trainer.strategy.reduce(segment_count, reduce_op="max").item()
                        backward_segment_count = forward_segment_count
                    else:
                        # We perform as many forward pass as we need to be equal or more then bptt_learning_range
                        # and perform an equal amount of backward pass
                        forward_segment_count  = max(segment_count, self.bptt_learning_range)
                        backward_segment_count = self.bptt_learning_range
                else:
                    if self.bptt_learning_range <= 0:
                        # Since we do not need to sync GPUs here, we perform as much forward as we exactly need
                        forward_segment_count  = segment_count
                        backward_segment_count = forward_segment_count
                    else:
                        # We clamp the backward segment count to the forward count, and bptt_learning_range
                        forward_segment_count  = segment_count
                        backward_segment_count = min(self.bptt_learning_range, segment_count)

                # We compute when we start the segmented learning process
                if forward_segment_count != backward_segment_count:
                    start_learning_segment = max(segment_count - self.bptt_learning_range, 0);
                else:
                    start_learning_segment = 0;



                # Lets go through and forward all the segments 
                # (including dummy ones)
                for i in range(forward_segment_count):
                    # Apply state truncation, if truncated learning is enabled
                    # this limits the backprop process, reduces loss learning rate, 
                    # but save vram across extreamly large backpropagation steps
                    if self.bptt_truncated_learning:
                        prv_shift_states = states.shift_states.clone().detach().requires_grad_(False)
                        prv_wkv_states = states.wkv_states.clone().detach().requires_grad_(False)
                    else:
                        prv_shift_states = states.shift_states
                        prv_wkv_states = states.wkv_states
                    
                    # We use a dummy masked token 0, to do additional dummy checkpoint/forward/backprop when needed
                    # for each additional call after the current "segment_count" max
                    if i <= segment_count - 1:
                        cur_idx = idx[:, i * segment_size:(i + 1) * segment_size]
                        cur_tar = targets[:, i * segment_size:(i + 1) * segment_size]
                        cur_msk = seq_mask[:, i * segment_size:(i + 1) * segment_size]
                    else:
                        cur_idx = dummy_2d_zero
                        cur_tar = dummy_2d_zero
                        cur_msk = dummy_2d_zero

                    # Segmented learning, applies the forward/pass over each chunk seperately
                    segment_loss, new_shift_states, new_wkv_states, steps = checkpointed_step(
                        cur_idx,
                        cur_tar,
                        cur_msk,
                        torch.tensor(0, dtype=self.emb.weight.dtype, device=cur_device).requires_grad_(True),
                        prv_shift_states,
                        prv_wkv_states,
                        steps,
                    )
                    states = BlockStateList(new_shift_states, new_wkv_states)

                    if i >= start_learning_segment and i < start_learning_segment + backward_segment_count:

                        learning_loss = segment_loss / gradient_accumulation_steps

                        # Perform the backward pass accordingly, for valid segments (besides the last segment)
                        if i == start_learning_segment + backward_segment_count - 1:
                            total_loss = total_loss + segment_loss
                        else:
                        
                            self.manual_backward(learning_loss, optimizer, retain_graph=True)
                
                            # Accumulate without gradient, as we already did the backward pass
                            total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
                    else:
                        # Even if its not the segments we use for backward pass, we still need to accumulate the loss
                        total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)                        
        else:

                # Normal operations without BPTT
                segment_size = self.ctx_len
                for i in range(segment_count):
                    if i < segment_count-1 and is_training_run:
                        total_loss, new_shift_states, new_wkv_states, steps = deepspeed_checkpoint(
                            checkpointed_step,
                            idx[:, i * segment_size:(i + 1) * segment_size],
                            targets[:, i * segment_size:(i + 1) * segment_size],
                            seq_mask[:, i * segment_size:(i + 1) * segment_size],
                            total_loss,
                            states.shift_states,
                            states.wkv_states,
                            steps,
                        )
                    else:
                        total_loss, new_shift_states, new_wkv_states, steps = checkpointed_step(
                            idx[:, i * segment_size:(i + 1) * segment_size],
                            targets[:, i * segment_size:(i + 1) * segment_size],
                            seq_mask[:, i * segment_size:(i + 1) * segment_size],
                            total_loss,
                            states.shift_states,
                            states.wkv_states,
                            steps,
                        )

                    states = BlockStateList(new_shift_states, new_wkv_states)
                    gc.collect()
                    # torch.cuda.empty_cache()

        if wandb.run is not None:
            self._log_metrics(batch_idx, total_loss, T)
        
        return total_loss    


    def _log_metrics(self, batch_idx, total_loss, T):
        global_rank = self.global_rank
        global_device_count = self.trainer.num_devices * self.trainer.num_nodes
        wandb.log({
            'substep': batch_idx * global_device_count + global_rank,
            'batchidx': batch_idx,
            'global_rank': global_rank,
            'real_ctx_len': T,
            'train/loss': total_loss,
            'trainer/global_step': self.global_step,
            'trainer/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        })            


    @TCompileBaseline
    def training_step(self, batch, batch_idx):
        # Compute the training loss using helper function
        total_loss = self.compute_loss(batch, batch_idx, is_training_run=True)

        # Log the training loss
        self.log('train/loss', total_loss, prog_bar=True)

        # Force log line to be on a new line if substep_logging is set
        if self.substep_logging:
            print("")

        # Clear CUDA cache if enabled
        if self.substep_cuda_cache_clear:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss

    @TCompileBaseline
    def validation_step(self, batch, batch_idx):
        # Compute the validation loss using helper function
        total_loss = self.compute_loss(batch, batch_idx, is_training_run=False)

        # Log the validation loss
        self.log('validation/loss', total_loss, prog_bar=True, sync_dist=True)

        return total_loss
# =================== RWKV: SimpleRWKV, a wrapper for RWKV that allows for simple usage of the model ===================
# SimpleRWKV specific imports
from transformers import PreTrainedTokenizerFast

# Current script dir
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../'))

# SimpleRWKV is a wrapper for RWKV that allows for simple usage of the model
#
# it is not meant to be highly performant, but rather a simple minimal way to run the RWKV trainer module
# in inference mode, and can be used to validate the model trainer code / its changes

class SimpleRWKV:
    # Initialization logic
    def __init__(self, model_path: str, ctx_len: int = 1024,
                 device: str = "cuda", dtype: str = "fp32"):


        # Log the mismatch dtype
        if dtype != "fp32":
            print("[SimpleRWKV] Warning: dtype mismatch, only fp32 is supported (for now)")

        # Prepare the model config with the model path, and custom torch load
        model_config = {}
        model_config["load_model"] = model_path
        model_config["ctx_len"] = ctx_len

        # This feature depends on deepspeed
        model_config["grad_cp"] = False
        # model_config["_torch_load_state"] = loaded_state

        # Save the config settings
        self.ctx_len = ctx_len
        self.device = device

        # Lets actually load the model
        self.model = RWKV(**model_config)

        # Lets map it over to the respective device type
        # and set it to run as eval/inference mode
        self.model.to(device)
        self.model.eval()

        # Get the model detected vocab size
        vocab_size = self.model.vocab_size

        # The tokenizer object values
        self.fastTokenizer = None
        self.worldTokenizer = None

        # Setup the tokenizer
        if vocab_size == 50277:
            # Use the neox tokenizer
            tokenizer_file = os.path.join(SCRIPT_DIR,"./dataflow/20B_tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            self.fastTokenizer = tokenizer
        elif vocab_size == 65529 or vocab_size == 65536:
            # Use the world tokenizer
            from .dataflow.trie_tokenizer import MT_TRIE_TOKENIZER
            world_tokenizer = MT_TRIE_TOKENIZER(os.path.join(SCRIPT_DIR, "./dataflow/rwkv_vocab_v20230424.txt"))
            self.worldTokenizer = world_tokenizer
        else:
            raise NotImplementedError(f"Unsupported vocab size ({vocab_size}) - custom tokenizer not supported")

    # Encoding strings
    def encode(self, text: str):
        if self.worldTokenizer != None:
            return self.worldTokenizer.encode(text)
        return self.fastTokenizer.encode(text)

    # Decoding strings
    def decode(self, tokens: list):
        if self.worldTokenizer != None:
            return self.worldTokenizer.decode(tokens)
        return self.fastTokenizer.decode(tokens)

    # Forwarding logic without torch._no_grad() context
    def _forward(self, tokens, stateObj=None, all_logits=False):
        logits_arr = None
        token_len = len(tokens)

        # Get the shift/wkv state
        if stateObj is None:
            shift_states = None
            wkv_states = None
        else:
            shift_states = stateObj["shift_states"]
            wkv_states = stateObj["wkv_states"]
        
        # The all_logits array, if requested
        all_logits_arr = None

        # For each token, process the state, in batches up to ctx_len
        for i in range(0, token_len, self.ctx_len):
            # Token set
            token_set = tokens[i:i+self.ctx_len]

            # Check if tokens are already tensors
            batch_tokens = torch.tensor(
                token_set, 
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            # Compute the logits and state
            logits_arr, shift_states, wkv_states = self.model.forward(
                batch_tokens, shift_states, wkv_states
            )

            # Build the all_logits array
            if all_logits:
                if all_logits_arr is None:
                    all_logits_arr = logits_arr[0]
                else:
                    all_logits_arr = torch.cat([all_logits_arr, logits_arr[0]], dim=0)

        # Return the logits and state
        if all_logits:
            return all_logits_arr, { "shift_states": shift_states, "wkv_states": wkv_states }
        else:
            return logits_arr[0][-1], { "shift_states": shift_states, "wkv_states": wkv_states }
    
    
    # Forwarding logic with torch._no_grad() context
    def forward(self, tokens: list, stateObj=None, all_logits=False):
        with torch.no_grad():
            return self._forward(tokens, stateObj, all_logits)

    # Sampling logits
    def sample_logits(self, logits, prv_tokens=[0],
                      temperature=1.0, top_p=0.9, token_ban: list = []):
        # Copy to CPU first
        logits = logits.cpu()

        # Max negative float
        max_neg = -torch.finfo(torch.float).max

        # Apply token ban
        for x in token_ban:
            logits[x] = max_neg
        
        # Remove NaNs from logits
        for x in range(len(logits)):
            if torch.isnan(logits[x]):
                logits[x] = max_neg

        # Handle sampling with temperature
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out
        else: 
            # Since the tokenizer sample does not support temp==0
            # we handle this case ourself, by fining the top token
            return torch.argmax(logits, dim=-1).item()

    # Completion API
    def completion(self, prompt, max_tokens: int = 32,
                   temperature: float = 1.0, top_p: float = 0.9,
                   token_ban: list = [], start_state=None,
                   stream_to_stdout: bool = False):
        # Encode the context, if its a string
        if isinstance(prompt, str):
            enc = self.encode(prompt)
        # Check if the prompt is a list of tokens
        elif isinstance(prompt, list):
            enc = prompt
        else:
            raise ValueError("Prompt must be a string or a list of tokens")

        # Keep track of the logits and state
        logits = None
        stateObj = start_state

        # For each token, process the state
        logits, stateObj = self.forward(enc, stateObj)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Generate each token
        out_tokens = []
        for i in range(max_tokens):
            ttt = self.sample_logits(
                logits, 
                # prv_tokens=full_tokens,
                temperature=temperature, top_p=top_p,
                token_ban=token_ban
            )
            
            # Append the token
            out_tokens.append(ttt)
            # full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, stateObj = self.forward([ttt], stateObj)

        # Decode the tokens
        out_str = self.decode(out_tokens)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, stateObj