import math 
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
import loralib as lora


class ClipModel(nn.Module):
    def __init__(self, name, opt, num_classes=1):
        super(ClipModel, self).__init__()
        self.use_svd = opt.use_svd
        
        if self.use_svd:
            self.model = CLIPModel.from_pretrained(name)
            svd_rank = self.model.config.vision_config.hidden_size - 1
            self.model.vision_model = apply_svd_residual_to_self_attn(
                self.model.vision_model,
                r=svd_rank,
                low_rank_forward=getattr(opt, "svd_low_rank_forward", False),
            )
            
            for name, param in self.model.vision_model.named_parameters():
                print('{}: {}'.format(name, param.requires_grad))
            num_param = sum(p.numel() for p in self.model.vision_model.parameters() if p.requires_grad)
            num_total_param = sum(p.numel() for p in self.model.vision_model.parameters())
            print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

            hidden_size = self.model.config.vision_config.hidden_size
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.model = CLIPModel.from_pretrained(name)
            
            for name, param in self.model.vision_model.named_parameters():
                print('{}: {}'.format(name, param.requires_grad))
            num_param = sum(p.numel() for p in self.model.vision_model.parameters() if p.requires_grad)
            num_total_param = sum(p.numel() for p in self.model.vision_model.parameters())
            print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

            hidden_size = self.model.config.vision_config.hidden_size
            self.fc = nn.Linear(hidden_size, num_classes)

        self.feature_dim = self.fc.in_features

    def forward(self, x, return_feature=False, return_aux_map=False):
        features, logits, spatial_feat = self.extract_features(x, return_aux_map=return_aux_map)
        if return_feature:
            return features, logits, spatial_feat
        return logits

    def extract_features(self, x, return_aux_map=False):
        outputs = self.model.vision_model(pixel_values=x)
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
            features = outputs[1] if len(outputs) > 1 else last_hidden_state[:, 0, :]
        else:
            last_hidden_state = outputs.last_hidden_state
            features = outputs.pooler_output
        logits = self.fc(features)
        spatial_feat = None
        if return_aux_map:
            tokens = last_hidden_state[:, 1:, :]
            n = tokens.shape[1]
            h = int(round(math.sqrt(n)))
            if h * h == n:
                spatial_feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, h)
            else:
                # Fallback for non-square token grids.
                spatial_feat = tokens.transpose(1, 2).unsqueeze(-2)
        return features, logits, spatial_feat

    def svd_regularization_losses(self):
        zero = self.fc.weight.sum() * 0.0
        orth_loss = zero
        ksv_loss = zero
        num_orth = 0
        num_ksv = 0

        for module in self.modules():
            if not isinstance(module, SVDResidualLinear):
                continue

            if module.U_residual is not None and module.V_residual is not None:
                eye_u = torch.eye(module.U_residual.shape[1], device=module.U_residual.device, dtype=module.U_residual.dtype)
                eye_v = torch.eye(module.V_residual.shape[0], device=module.V_residual.device, dtype=module.V_residual.dtype)
                orth_u = torch.norm(module.U_residual.t() @ module.U_residual - eye_u, p='fro') ** 2
                orth_v = torch.norm(module.V_residual @ module.V_residual.t() - eye_v, p='fro') ** 2
                orth_loss = orth_loss + orth_u + orth_v
                num_orth += 1

            if module.S_residual is not None and hasattr(module, "S_residual_init"):
                denom = torch.sum(module.S_residual_init ** 2) + 1e-8
                numer = torch.sum(module.S_residual ** 2)
                ksv_loss = ksv_loss + torch.abs(numer / denom - 1.0)
                num_ksv += 1

        if num_orth > 0:
            orth_loss = orth_loss / num_orth
        if num_ksv > 0:
            ksv_loss = ksv_loss / num_ksv
        return orth_loss, ksv_loss


# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None, low_rank_forward=False):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude
        self.low_rank_forward = low_rank_forward

        # Main weight (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        # SVD components
        self.S_r = None
        self.U_r = None
        self.V_r = None
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None

    def forward(self, x):
        if self.S_residual is not None:
            if self.low_rank_forward:
                main_out = F.linear(x, self.weight_main, self.bias)
                low_rank = x.matmul(self.V_residual.t())
                low_rank = low_rank * self.S_residual
                delta_out = low_rank.matmul(self.U_residual.t())
                return main_out + delta_out
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)
                   

# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r, low_rank_forward=False):
    for name, module in model.named_children():
        # if ('self_attn' in name) or ('mlp' in name):
        if ('self_attn' in name):
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(
                        parent_module,
                        sub_module_names[-1],
                        replace_with_svd_residual(sub_module, r, low_rank_forward=low_rank_forward),
                    )
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r, low_rank_forward=low_rank_forward)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r, low_rank_forward=False):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(
            in_features,
            out_features,
            r,
            bias=bias,
            init_weight=module.weight.data.clone(),
            low_rank_forward=low_rank_forward,
        )

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
            
        # Calculate the frobenius norm of original weight
        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        
        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
            new_module.register_buffer('S_residual_init', S_residual.clone(), persistent=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

        return new_module
    else:
        return module
