import logging
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
from torchvision import transforms

from copy import deepcopy
from functools import wraps
from models.model import ResNetDomainNet126

import math
from utils.losses import Entropy

logger = logging.getLogger(__name__)


class TTAMethod(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.episodic = cfg.MODEL.EPISODIC
        self.dataset_name = cfg.CORRUPTION.DATASET
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables for resetting the model after a certain amount of performed update steps
        self.performed_updates = 0
        self.reset_after_num_updates = cfg.MODEL.RESET_AFTER_NUM_UPDATES

        # restore the image input size from the model pre-processing if it is defined
        # this is required for methods relying on test-time augmentation
        self.img_size = (32, 32) if "cifar" in self.dataset_name else (224, 224)
        if hasattr(self.model, "model_preprocess") and isinstance(self.model.model_preprocess, transforms.Compose):
            for transf in self.model.model_preprocess.transforms[::-1]:
                if hasattr(transf, "size"):
                    self.img_size = getattr(transf, "size")
                    if self.dataset_name in ["imagenet_c", "ccc"] and max(self.img_size) > 224:
                        raise ValueError(f"The specified model with pre-processing {model.model_preprocess} "
                                         f"is not suited in combination with ImageNet-C and CCC! "
                                         f"These datasets are already resized and center cropped to 224")
                    break

        # configure model and optimizer
        self.configure_model()
        self.params, self.param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.num_trainable_params, self.num_total_params = self.get_number_trainable_params()

        # variables needed for single sample test-time adaptation (sstta) using a sliding window (buffer) approach
        self.input_buffer = None
        self.window_length = cfg.TEST.WINDOW_LENGTH
        self.pointer = torch.tensor([0], dtype=torch.long).to(self.device)
        # sstta: if the model has no batchnorm layers, we do not need to forward the whole buffer when not performing any updates
        self.has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # setup for mixed-precision or single precision
        self.mixed_precision = cfg.MIXED_PRECISION
        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

        self.asr_on = cfg.ASR.ON
        self.asr_m0 = cfg.ASR.M0
        self.asr_p0 = cfg.ASR.P0
        self.asr_a0 = cfg.ASR.A0
        self.asr_r0 = cfg.ASR.R0
        self.asr_r1 = cfg.ASR.R1

        self.asr_step = 0
        self.p_fisher = 0
        self.entropy = math.inf
        self.softmax_entropy = Entropy()
        self.src = deepcopy(self.model)
        self.src_states = deepcopy(self.model_states)
        self.cum_entropy = math.log(self.num_classes * self.asr_a0)

    def forward(self, x):
        if self.episodic:
            self.reset()

        x = x if isinstance(x, list) else [x]

        if x[0].shape[0] == 1:  # single sample test-time adaptation
            # create the sliding window input
            if self.input_buffer is None:
                self.input_buffer = [x_item for x_item in x]
                # set bn1d layers into eval mode, since no statistics can be extracted from 1 sample
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=False)
            elif self.input_buffer[0].shape[0] < self.window_length:
                self.input_buffer = [torch.cat([self.input_buffer[i], x_item], dim=0) for i, x_item in enumerate(x)]
                # set bn1d layers into train mode
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=True)
            else:
                for i, x_item in enumerate(x):
                    self.input_buffer[i][self.pointer] = x_item

            if self.pointer == (self.window_length - 1):
                # update the model, since the complete buffer has changed
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(self.input_buffer)

                    # if specified, reset the model after a certain amount of update steps
                    self.performed_updates += 1
                    if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                        self.reset()

                outputs = outputs[self.pointer.long()]
            else:
                # create the prediction without updating the model
                if self.has_bn:
                    # forward the whole buffer to get good batchnorm statistics
                    outputs = self.forward_sliding_window(self.input_buffer)
                    outputs = outputs[self.pointer.long()]
                else:
                    # only forward the current test sample, since there are no batchnorm layers
                    outputs = self.forward_sliding_window(x)

            # increase the pointer
            self.pointer += 1
            self.pointer %= self.window_length

        else:   # common batch adaptation setting
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)

                # if specified, reset the model after a certain amount of update steps
                self.performed_updates += 1
                if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                    self.reset()
                
                if self.asr_on:
                    self.asr_step += 1
                    self.entropy = self.softmax_entropy(outputs.mean(dim=0, keepdim=True)).item()

                    y_src = self.src(x[0]).max(-1)[1]
                    y_tgt = outputs.max(-1)[1]
                    y_flip = (sum(y_src != y_tgt) / len(y_tgt)).item()

                    self.m_entropy = self.asr_m0 * y_flip + 1 - self.asr_m0
                    self.p_fisher = self.asr_p0 * (y_flip ** 2)

                    if self.entropy < self.cum_entropy:
                        self.register_ewc_params(self.asr_step, momentum=0.9)
                        self.reset()
                    else:
                        self.cum_entropy = self.m_entropy * self.cum_entropy + (1 - self.m_entropy) * self.entropy

                    if 'roid' in self.cfg.MODEL.ADAPTATION:
                        prior = outputs.softmax(1).mean(0)
                        smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                        smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                        outputs *= smoothed_prior

        return outputs

    def register_ewc_params(self, step, momentum):
        for nm in self.param_names:
            buff_param_name = nm.replace('.', '__')
            buffered_fisher = 1 / step * getattr(self.model, '{}_buffered_fisher'.format(buff_param_name))
            if hasattr(self.model, '{}_estimated_fisher'.format(buff_param_name)):
                fisher = getattr(self.model, '{}_estimated_fisher'.format(buff_param_name))
                self.model.register_buffer(buff_param_name+'_estimated_fisher', momentum * fisher + (1 - momentum) * buffered_fisher)
            else:
                self.model.register_buffer(buff_param_name+'_estimated_fisher', buffered_fisher)

            buffered_mean = 1 / step * getattr(self.model, '{}_buffered_mean'.format(buff_param_name))
            if hasattr(self.model, '{}_estimated_mean'.format(buff_param_name)):
                mean = getattr(self.model, '{}_estimated_mean'.format(buff_param_name))
                self.model.register_buffer(buff_param_name+'_estimated_mean', momentum * mean + (1 - momentum) * buffered_mean)
            else:
                self.model.register_buffer(buff_param_name+'_estimated_mean', buffered_mean)

            delattr(self.model, '{}_buffered_fisher'.format(buff_param_name))
            delattr(self.model, '{}_buffered_mean'.format(buff_param_name))

    def memorise_ewc_params(self, loss):
        self.update_fisher_params(loss)
        self.update_mean_params()

    def update_fisher_params(self, loss):
        grad_loss = torch.autograd.grad(loss, self.params, retain_graph=True)
        buff_param_names = [nm.replace('.', '__') for nm in self.param_names]
        for buff_param_name, param in zip(buff_param_names, grad_loss):
            if not hasattr(self.model, '{}_buffered_fisher'.format(buff_param_name)):
                self.model.register_buffer(buff_param_name+'_buffered_fisher', param.data.clone() ** 2)
            else:
                buffered_fisher = getattr(self.model, '{}_buffered_fisher'.format(buff_param_name))
                buffered_fisher += param.data.clone() ** 2
                self.model.register_buffer(buff_param_name+'_buffered_fisher', buffered_fisher)

    def update_mean_params(self):
        for nm, param in zip(self.param_names, self.params):
            buff_param_name = nm.replace('.', '__')
            if not hasattr(self.model, '{}_buffered_mean'.format(buff_param_name)):
                self.model.register_buffer(buff_param_name+'_buffered_mean', param.data.clone())
            else:
                buffered_mean = getattr(self.model, '{}_buffered_mean'.format(buff_param_name))
                buffered_mean += param.data.clone()
                self.model.register_buffer(buff_param_name+'_buffered_mean', buffered_mean)

    def compute_consolidation_loss(self, coef):
        try:
            losses = []
            for nm, param in zip(self.param_names, self.params):
                buff_param_name = nm.replace('.', '__')
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(buff_param_name))
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(buff_param_name))
                losses.append(sum(estimated_fisher * (param - estimated_mean) ** 2))
            return coef * sum(losses)
        except AttributeError:
            return 0

    def asr_reset(self):
        for model, model_state, src_state in zip(self.models, self.model_states, self.src_states):
            scope = max(self.asr_r0 - self.asr_r1 * (self.cum_entropy - self.entropy), 0)
            new_states = []
            for i, (param, nm) in enumerate(zip(self.params, self.param_names)):
                if i < scope * len(self.params):
                    new_states.append([nm, param])
                else:
                    new_states.append([nm, src_state[nm]])
            model_state.update(deepcopy(dict(new_states)))
            model.load_state_dict(model_state, strict=False)
        self.optimizer.load_state_dict(self.optimizer_state)
        logger.info(f"[reset] scope={scope:.2f}, asr_step={self.asr_step}, entropy={self.entropy:.3f}, cum_entropy={self.cum_entropy:.3f}")
        self.cum_entropy = math.log(self.num_classes * self.asr_a0)
        self.asr_step = 0

    def loss_calculation(self, x):
        """
        Loss calculation.
        """
        raise NotImplementedError

    def forward_and_adapt(self, x):
        """
        Forward and adapt the model on a batch of data.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model(imgs_test)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.cfg.OPTIM.METHOD == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.cfg.OPTIM.LR,
                                    betas=(self.cfg.OPTIM.BETA, 0.999),
                                    weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'AdamW':
            return torch.optim.AdamW(self.params,
                                     lr=self.cfg.OPTIM.LR,
                                     betas=(self.cfg.OPTIM.BETA, 0.999),
                                     weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return torch.optim.SGD(self.params,
                                   lr=self.cfg.OPTIM.LR,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def get_number_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"#Trainable/total parameters: {trainable:,}/{total:,} \t Ratio: {trainable / total * 100:.3f}% ")
        return trainable, total

    def reset(self):
        """Reset the model and optimizer state to the initial source state"""
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def copy_model(model):
        if isinstance(model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            coppied_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            coppied_model = deepcopy(model)
        return coppied_model

    @staticmethod
    def change_mode_of_batchnorm1d(model_list, to_train_mode=True):
        # batchnorm1d layers do not work with single sample inputs
        for model in model_list:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if to_train_mode:
                        m.train()
                    else:
                        m.eval()


def forward_decorator(fn):
    @wraps(fn)
    def decorator(self, *args, **kwargs): 
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = fn(self, *args, **kwargs)
        else:
            outputs = fn(self, *args, **kwargs)
        return outputs
    return decorator

