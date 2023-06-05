import torch
import math
import random
from torch import nn


def low_rank_decomposition(weight, rank_ratio=0.1, parameter_ratio=0.15,
                           remove_criteria='max_eigenvalue',
                           log_level='INFO',
                           return_dict=False):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param      rank_ratio: rank_of_decomposed_matrix / rank_of_input_weight
    :param parameter_ratio: parameter_num_of_decomposed_matrix / (H * W). If specify, override rank_ratio
    :param remove_criteria: choose from ['max_eigenvalue', 'random', 'min_eigenvalue']
    :param       log_level: choose from ['IGNORE', 'INFO', 'DEBUG']
    :param     return_dict: Return a dict if True, else return a tuple (L, R)
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)

    if parameter_ratio is not None:
        reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
    else:
        reduced_rank = math.ceil(rank * rank_ratio)

    if remove_criteria == 'max_eigenvalue':
        L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
        R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh
    elif remove_criteria == 'random':
        selected_index = random.choices(range(len(S)), k=reduced_rank)
        L = U @ (torch.sqrt(torch.diag(S)[:, selected_index]))
        R = torch.sqrt(torch.diag(S)[selected_index, :]) @ Vh
    elif remove_criteria == 'min_eigenvalue':
        len_s = len(S)
        L = U @ (torch.sqrt(torch.diag(S)[:, len_s - reduced_rank:]))
        R = torch.sqrt(torch.diag(S)[len_s - reduced_rank:, :]) @ Vh
    else:
        raise NameError("remove criteria not support")

    #########
    #  LOG  #
    #########
    if log_level == 'INFO':
        if not is_full_rank:
            print(f"It is not a full rank matrix. Rank: {rank} | H x W: {H}, {W}")
        print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
    if log_level == 'DEBUG':
        print(f"W: ({H},{W}) | Rank: {rank} | U:{U.shape} | S:{S.shape} | Vh:{Vh.shape}")
        print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
        print(f"L: {L.shape} | R: {R.shape}")

    if return_dict:
        return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'reduced_rank': reduced_rank}
    else:
        return L, R


class LinearLoSparse(nn.Module):
    def __init__(self, in_feature, out_feature, reduced_rank, has_bias=True, has_sparse=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.reduced_rank = reduced_rank
        self.has_bias = has_bias
        self.has_sparse = has_sparse

        self.right = nn.Linear(in_feature, reduced_rank, bias=False)
        self.left = nn.Linear(reduced_rank, out_feature, bias=False)
        if self.has_sparse:
            self.sparse = nn.Linear(in_feature, out_feature, bias=False)

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(out_feature, requires_grad=True))

        self.nonzero_idx = None
        self.sparse_weight_pruned = None
        self.SX = None
        self.SX_deberta = None    # Deberta will use Q and K again

    def forward(self, x):
        """Y = XW.T+B = X(LR+S).T+B = X(LR).T+XS.T+B"""
        LRX = self.left(self.right(x))
        if self.has_sparse:
            if self.sparse_weight_pruned is not None:
                SX_ = torch.matmul(x, self.sparse_weight_pruned.T)
                B, L, D = x.shape

                # restore y
                # keep record for the first forward
                if self.SX is None or self.SX_deberta is None:  # For QKV at the first time
                    out_feature, in_feature = self.sparse.weight.shape
                    device = x.device
                    if B != 1:
                        self.SX = torch.zeros(B, L, out_feature, device=device)
                        self.SX[..., self.nonzero_idx] = SX_
                        Y = LRX + self.SX + self.bias if self.has_bias else LRX + self.SX
                    else:   # For QK at the second time
                        self.SX_deberta = torch.zeros(B, L, out_feature, device=device)
                        self.SX_deberta[..., self.nonzero_idx] = SX_
                        Y = LRX + self.SX_deberta + self.bias if self.has_bias else LRX + self.SX_deberta

                # do not need to create new cuda memory
                else:
                    if B != 1:
                        self.SX[..., self.nonzero_idx] = SX_
                        Y = LRX + self.SX + self.bias if self.has_bias else LRX + self.SX
                    else:
                        self.SX_deberta[..., self.nonzero_idx] = SX_
                        Y = LRX + self.SX_deberta + self.bias if self.has_bias else LRX + self.SX_deberta
            else:
                SX = self.sparse(x)
                Y = LRX + SX + self.bias if self.has_bias else LRX + SX
        else:
            Y = LRX + self.bias if self.has_bias else LRX
        return Y

    def initialize_weight(self, left_weight, right_weight, sparse_weight=None, bias=None):
        self.left.weight = nn.Parameter(left_weight, requires_grad=True)
        self.right.weight = nn.Parameter(right_weight, requires_grad=True)
        if self.has_sparse:
            self.sparse.weight = nn.Parameter(sparse_weight, requires_grad=True)
        if self.has_bias:
            self.bias = nn.Parameter(bias, requires_grad=True)

    def prune_sparse(self):
        self.nonzero_idx = torch.nonzero(self.sparse.weight.sum(dim=1)).flatten()
        self.sparse_weight_pruned = self.sparse.weight[self.nonzero_idx, :]


def prune(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == LinearLoSparse:
            print("====================================================")
            print(attr_str, target_attr)
            target_attr.prune_sparse()
    for name, immediate_child_module in module.named_children():
        prune(immediate_child_module)


def substitute_layer_weights(module,
                             allow_name=None,
                             block_name=None,
                             parameter_ratio=0.15,
                             has_sparse=True,
                             do_svd=True,
                             **kwargs):
    """
    :param          do_svd: operate SVD
    :param          module: an nn.Module class
    :param      block_name: do not continue to iterate when the module's name is in the block_name
    :param      allow_name: replace the module if its name is in the allow_name
    :param parameter_ratio: low rank matrix parameter / original matrix parameter
    :param      has_sparse: True if use LoRaS, false if use Low Rank only

    :return: None
    """
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query', 'key', 'value', 'dense', 'attention']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == nn.Linear and any(attr_str in an for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)

            if do_svd:
                # Decompose a matrix by SVD
                output = low_rank_decomposition(target_attr.weight, parameter_ratio=parameter_ratio,
                                                return_dict=True, **kwargs)
                L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
                S = target_attr.weight - torch.mm(L, R)
                print(f"Reduced rank: {reduced_rank}")

                # Create a nn.Module and assign decomposed weights to the parameters
                linear_loras = LinearLoSparse(target_attr.in_features, target_attr.out_features, reduced_rank,
                                           has_bias=True, has_sparse=has_sparse)
                linear_loras.initialize_weight(L, R, S, target_attr.bias)

            else:
                H, W = target_attr.weight.shape
                reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
                L = torch.zeros(H, reduced_rank, requires_grad=True)
                R = torch.zeros(reduced_rank, W, requires_grad=True)
                S = torch.zeros(H, W, requires_grad=True)

                # Create a nn.Module and assign decomposed weights to the parameters
                linear_loras = LinearLoSparse(target_attr.in_features, target_attr.out_features, reduced_rank,
                                           has_bias=True, has_sparse=has_sparse)

                linear_loras.initialize_weight(L, R, S, target_attr.bias)

            setattr(module, attr_str, linear_loras)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            substitute_layer_weights(immediate_child_module, allow_name, block_name, parameter_ratio,
                                     has_sparse, do_svd, **kwargs)


class Pruner(object):
    def __init__(self, model, args, total_step, tb_writer=None,
                 mask_param_name=None,
                 non_mask_name=None,
                 use_no_mask=False,
                 pruner_name='PLATON',
                 structured_method='mean',
                 structured_direction='row'):

        if non_mask_name is None:
            non_mask_name = ["embedding", "norm"]
        if mask_param_name is None:
            mask_param_name = ['sparse']
        self.model = model
        self.config = vars(args)
        self.args = args
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.mask_param_name = mask_param_name
        self.non_mask_name = non_mask_name
        self.use_no_mask = use_no_mask
        self.total_step = total_step
        self.tb_writer = tb_writer
        self.pruner_name = pruner_name
        self.beta1 = self.config["beta1"]
        self.beta2 = self.config["beta2"]
        self.deltaT = self.config["deltaT"]
        self.structured_method = structured_method
        self.structured_direction = structured_direction

    def whether_mask_para(self, n):
        if not self.use_no_mask:
            return any(nd in n for nd in self.mask_param_name)
        else:
            return not any([nd in n for nd in self.non_mask_name])

    def structured_prune(self, is_dict_mat, name):
        num_row, num_col = is_dict_mat.shape
        if self.structured_direction == 'row_col':
            if self.structured_method == "mean":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.mean(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.sum(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.max(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.min(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Sturctured Method: %s" % self.structured_method)
        elif self.structured_direction == 'row':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            else:
                raise ValueError("Unimplemented Sturctured Method: %s" % self.structured_method)
        elif self.structured_direction == 'col':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Sturctured Method: %s" % self.structured_method)
        else:
            raise ValueError("Unimplemented Sturctured Direction: %s" % self.structured_direction)

    def schedule_threshold_comb(self, step: int):
        # Schedule the remaining ratio
        args = self.args
        total_step = self.total_step
        initial_threshold = self.config['initial_threshold']
        final_threshold = self.config['final_threshold']
        initial_warmup = self.config['initial_warmup']
        final_warmup = self.config['final_warmup']
        warmup_steps = self.config['warmup_steps']
        mask_ind = False
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        else:
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
            mask_ind = True if step % self.deltaT == 0 else False
        return threshold, mask_ind

    def update_ipt_with_local_window(self, model, global_step):
        # Calculate the sensitivity and uncertainty
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.beta2 > 0 and self.beta2 != 1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                if self.pruner_name == 'Magnitude':
                    # Calculate the score of magnitude pruning
                    self.ipt[n] = p.abs().detach()
                elif self.pruner_name == 'PLATON':
                    local_step = global_step % self.deltaT
                    update_step = global_step // self.deltaT
                    if local_step == 0:
                        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                        if 0 < self.beta2 < 1:
                            self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                                  (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                        elif self.beta2 == 2.:
                            self.exp_avg_unc[n] = (update_step * self.exp_avg_unc[n] +
                                                   (self.ipt[n] - self.exp_avg_ipt[n]) ** 2) / (update_step + 1)
                        self.ipt[n] = (p * p.grad).abs().detach()
                    else:
                        self.ipt[n] = (self.ipt[n] * local_step + (p * p.grad).abs().detach()) / (local_step + 1)
                else:
                    raise ValueError("Incorrect Pruner Name.")

    def mask_with_threshold(self, model, threshold):
        # Calculate the final importance score
        is_dict = {}
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                if self.pruner_name == 'Magnitude':
                    is_dict[n] = self.ipt[n]
                elif self.pruner_name == 'PLATON':
                    if 0 < self.beta2 < 1:
                        is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
                    elif self.beta2 == 1.:
                        is_dict[n] = self.exp_avg_ipt[n]
                    elif self.beta2 == 2.:
                        is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt()
                    else:
                        # Handling the unaccepted beta2 to default setting
                        is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                else:
                    raise ValueError("Incorrect Pruner Name.")
                if self.structured_method is not None and len(is_dict[n].shape) == 2:
                    is_dict[n] = self.structured_prune(is_dict[n], n)
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - threshold)))[0].item()
        # Mask weights whose importance lower than threshold
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                p.data.masked_fill_(is_dict[n] < mask_threshold, 0.0)
        return mask_threshold

    def update_and_pruning(self, model, global_step):
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, global_step)
        # Get the remaining ratio
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(model, threshold)
        else:
            mask_threshold = None
        return threshold, mask_threshold

