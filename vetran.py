import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parameters import params
from functions import shuffle_down, shuffle_up, get_gaussian_kernel

# A Dynamic Filter from DFN
class HSA(nn.Module):
    def __init__(self, filter_size, nf, factor):
        super(HSA, self).__init__()
        num_filters = np.prod(filter_size)
        expand_filter_np = np.reshape(np.eye(num_filters, num_filters),
                                     (num_filters, filter_size[0], filter_size[1], filter_size[2]))
        expand_filter = torch.from_numpy(expand_filter_np).float()
        self.expand_filter = expand_filter.repeat(nf, 1, 1, 1)

        self.nf = nf
        self.k = filter_size[2]
        self.factor = factor
        self.conv_filter = nn.Conv2d(3 * self.factor ** 2, num_filters, self.k, padding=1)

    def forward(self, h, i_t):
        filters = F.relu(self.conv_filter(shuffle_down(i_t, factor=self.factor)))
        filters = filters.unsqueeze(dim=2)
        B, nF, R, H, W = filters.size()

        # using group convolution
        input_expand = F.conv2d(h, self.expand_filter.type_as(h), padding=1, groups=self.nf)
        input_expand = input_expand.view(B, self.nf, nF, H, W).permute(0, 3, 4, 1, 2)
        filters = filters.permute(0, 3, 4, 1, 2)
        out = torch.matmul(input_expand, filters)
        return out.permute(0, 3, 4, 1, 2).squeeze(dim=2)


class ForgetAttentionModule(nn.Module):
    def __init__(self, k, num_filters, factor, activation=torch.sigmoid):
        super(ForgetAttentionModule, self).__init__()

        self.activation = activation
        self.num_filters = num_filters
        self.k = k
        self.factor = factor

        self.key_conv_1 = nn.Conv2d(3, num_filters // self.factor**2, self.k, padding=(self.k // 2))

        self.value_conv = nn.Conv2d(num_filters, num_filters, self.k, padding=(self.k // 2))
        self.query_conv = nn.Conv2d(3 * self.factor ** 2, num_filters, self.k, padding=(self.k // 2))

    def forward(self, hs_t, i_t):
        k_it1 = shuffle_down(self.activation(self.key_conv_1(i_t[:, 1])), factor=self.factor)
        v_it = F.relu(self.value_conv(hs_t)) 

        energy = torch.mul(v_it, k_it1)
        q_it = torch.sigmoid(self.query_conv(shuffle_down(torch.abs(i_t[:, 1] - i_t[:, 0]), factor=self.factor)))

        out = hs_t + q_it * energy
        return out 


class SelfAttention(nn.Module):
    def __init__(self, k, num_filters, factor, activation=torch.sigmoid):
        super(SelfAttention, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.activation = activation

        self.num_filters = num_filters
        self.k = k
        self.factor = factor

        self.input_conv = nn.Conv2d(3 * self.factor ** 2, num_filters, self.k, padding=(self.k // 2))
        self.state_conv = nn.Conv2d(num_filters, num_filters, self.k, padding=(self.k // 2))
        self.value_conv = nn.Conv2d(num_filters, num_filters, self.k, padding=(self.k // 2))

    def forward(self, hs_t, i_t):
        if params['riam'] == 'diff':
            ki_t = self.activation(self.input_conv(shuffle_down(torch.abs(i_t[:, 1] - i_t[:, 0]), factor=self.factor)))
        elif params['riam'] == 'max':
            ki_t = self.activation(self.input_conv(shuffle_down(torch.maximum(i_t[:, 1], i_t[:, 0]), factor=self.factor)))
        else:
            ki_t = self.activation(self.input_conv(shuffle_down(i_t[:, 1], factor=self.factor)))
    
        qh_t = self.activation(self.state_conv(hs_t)) 
       
        energy = torch.mul(ki_t, qh_t)
        energy = torch.tanh(energy)

        if params['lerp'] == True:
            lamb_ = torch.sigmoid(self.gamma)
            out = lamb_ * hs_t + (1 - lamb_) * torch.mul(energy, self.value_conv(hs_t))
        elif params['forget'] == True:
            out = hs_t + self.gamma * torch.mul(energy, self.value_conv(hs_t))
        else:
            out = hs_t + torch.mul(energy, self.value_conv(hs_t))

        return out


class UnbalancedSDBlock(nn.Module):
    def __init__(self, num_ch_s, num_ch_d):
        super(UnbalancedSDBlock, self).__init__()
        self.conv1_s = nn.Conv2d(num_ch_s, num_ch_s, 3, padding=1)
        self.conv2_s = nn.Conv2d(num_ch_s + num_ch_d, num_ch_s, 3, padding=1)

        self.conv1_d = nn.Conv2d(num_ch_d, num_ch_d, 3, padding=1)
        self.conv2_d = nn.Conv2d(num_ch_d + num_ch_s, num_ch_d, 3, padding=1)

    def forward(self, s, d):
        sh = F.relu(self.conv1_s(s))
        dh = F.relu(self.conv1_d(d))

        s_out = F.relu(s + self.conv2_s(torch.cat((sh, dh), dim=1))) # relu for clamping
        d_out = F.relu(d + self.conv2_d(torch.cat((dh, sh), dim=1)))

        return s_out, d_out


class UnbalancedAdditiveSDBlock(nn.Module):
    def __init__(self, num_ch_s, num_ch_d):
        super(UnbalancedAdditiveSDBlock, self).__init__()
        self.conv1_s = nn.Conv2d(num_ch_s, num_ch_s, 3, padding=1)
        self.conv_ds = nn.Conv2d(num_ch_d, num_ch_s, 3, padding=1)
        self.conv2_s = nn.Conv2d(num_ch_s, num_ch_s, 3, padding=1)

        self.conv1_d = nn.Conv2d(num_ch_d, num_ch_d, 3, padding=1)
        self.conv_sd = nn.Conv2d(num_ch_s, num_ch_d, 3, padding=1)
        self.conv2_d = nn.Conv2d(num_ch_d, num_ch_d, 3, padding=1)

        if params['block activation'] == 'prelu':
            self.ws1 = nn.Parameter(torch.randn(1))
            self.ws2 = nn.Parameter(torch.randn(1))
            self.wd1 = nn.Parameter(torch.randn(1))
            self.wd2 = nn.Parameter(torch.randn(1))

    def forward(self, s, d):
        if params['block activation'] == 'gelu':
            sh = F.gelu(self.conv1_s(s))
            dh = F.gelu(self.conv1_d(d))
        
            so = F.gelu(sh + F.relu(self.conv_ds(dh)))
            do = F.gelu(dh + F.relu(self.conv_sd(sh)))
        else:
            sh = F.prelu(self.conv1_s(s), self.ws1)
            dh = F.prelu(self.conv1_d(d), self.wd1)

            so = F.prelu(sh + F.relu(self.conv_ds(dh)), self.ws2)
            do = F.prelu(dh + F.relu(self.conv_sd(sh)), self.wd2)

        s_out = s + so
        d_out = d + do

        return s_out, d_out


class UnbalancedTransformerCell(nn.Module):
    def __init__(self, num_blocks, kernel_size, filters_s, filters_d, factor):
        super(UnbalancedTransformerCell, self).__init__()
        
        self.num_blocks = num_blocks
        self.k = kernel_size
        self.s_channels = filters_s
        self.d_channels = filters_d
        self.factor = factor
        
        if params['att type'] == 'similarity':
            self.att_s = ForgetAttentionModule(kernel_size, filters_s, factor)
            self.att_d = ForgetAttentionModule(kernel_size, filters_d, factor)
        else:
            self.att_s = SelfAttention(kernel_size, filters_s, factor)
            self.att_d = SelfAttention(kernel_size, filters_d, factor)
        
        if params['sd mode'] == 'cat':
            self.blocks = nn.ModuleList([UnbalancedSDBlock(filters_s, filters_d) for _ in range(num_blocks)])
        else:
            self.blocks = nn.ModuleList([UnbalancedAdditiveSDBlock(filters_s, filters_d) for _ in range(num_blocks)])

    def forward(self, sk, dk, s_in, d_in):
        s = self.att_s(sk, s_in)
        d = self.att_d(dk, d_in)

        for i in range(self.num_blocks):
            s, d = self.blocks[i](s, d)

        return s, d


class Transformer(nn.Module):
    def __init__(self, num_cells, attn_channels, state_dim, kernel_size, filters_s, filters_d, factor):
        super(Transformer, self).__init__()
        
        self.hsa_module = HSA(attn_channels, state_dim, factor)
        
        self.num_cells = num_cells
        self.cells = nn.ModuleList([UnbalancedTransformerCell(params['num_blocks'], kernel_size, filters_s, filters_d, factor)
                                            for _ in range(num_cells)])
        
        self.gauss_kernel = get_gaussian_kernel(5, 4, 3)
        
        # downscaling
        self.conv1_s = nn.Conv2d(2*3*factor**2 + filters_s + state_dim, filters_s, kernel_size, padding=int(kernel_size/2))
        self.conv1_d = nn.Conv2d(2*3*factor**2 + filters_d + state_dim, filters_d, kernel_size, padding=int(kernel_size/2))

        # output stage
        self.conv_state = nn.Conv2d(filters_d, state_dim, kernel_size, padding=int(kernel_size / 2))
        self.conv_out = nn.Conv2d(filters_d // factor**2, 3, kernel_size, padding=int(kernel_size / 2))
        
        self.conv_s_out = nn.Conv2d(filters_s // factor**2, 3, kernel_size, padding=int(kernel_size / 2))
        self.conv_d_out = nn.Conv2d(filters_d // factor**2, 3, kernel_size, padding=int(kernel_size / 2))

        # the goal is to prune the structure branch, so an additional conv should be added
        self.conv_norm_state = nn.Conv2d(filters_s, filters_d, kernel_size, padding=int(kernel_size / 2))
        self.conv_norm_out = nn.Conv2d(filters_s // factor**2, filters_d // factor**2, kernel_size, padding=int(kernel_size / 2))

    def get_structure_detail(self, y):
        y_s = self.gauss_kernel(y)
        y_d = y - y_s
        return y_s, y_d
    
    def forward(self, x_in, state_in, fb_s, fb_d, mode='direct'):
        if mode == 'of':
            state_out = self.hsa_module(state_in, torch.abs(x_in[:, 1] - x_in[:, 0]))
        else:
            state_out = self.hsa_module(state_in, x_in[:, 1])

        x_s = self.gauss_kernel(x_in.squeeze(0)).unsqueeze(0)
        x_d = x_in - x_s

        factor = params["shuffle_factor"]

        input_s = torch.cat([
            shuffle_down(x_s[:, 0], factor),
            shuffle_down(x_s[:, 1], factor),
            fb_s,
            state_out], -3)

        input_d = torch.cat([
            shuffle_down(x_d[:, 0], factor),
            shuffle_down(x_d[:, 1], factor),
            fb_d,
            state_out], -3)

        structure_x = F.relu(self.conv1_s(input_s))
        detail_x = F.relu(self.conv1_d(input_d))

        for i in range(self.num_cells):
            structure_x, detail_x = self.cells[i](structure_x, detail_x, x_s, x_d)
        
        # output the current state
        current_state = self.conv_state(self.conv_norm_state(structure_x) + detail_x)

        # prepare structure and detail
        
        s = F.pixel_shuffle(structure_x, factor)
        d = F.pixel_shuffle(detail_x, factor)

        s_out = self.conv_s_out(s) + x_s[:, 1]
        d_out = self.conv_d_out(d) + x_d[:, 1]

        out = self.conv_out(F.relu(self.conv_norm_out(s)) + d) + x_in[:, 1]
        return out, current_state, s_out, d_out, structure_x, detail_x


class Generator(nn.Module):
    def __init__(self, device=torch.device("cuda:0")):
        super(Generator, self).__init__()
        
        self.factor = params["shuffle_factor"]
        
        self.filters_s = params["filters_s"]
        self.filters_d = params["filters_d"]

        self.kernel_size = params["kernel size"]
        self.cells = params["generator layers"]
        
        self.state_dim = params["state dimension"]
        self.mode = params["block mode"]
        
        self.transformer = Transformer(self.cells, (1, 3, 3),  self.state_dim, self.kernel_size, self.filters_s, self.filters_d, self.factor)
        self.state = None

    def forward(self, x, y):
        y_s, y_d = self.transformer.get_structure_detail(y.squeeze(0))
        y_s = y_s.unsqueeze(0)
        y_d = y_d.unsqueeze(0)
        
        seq_hq = []
        seq_s  = []
        seq_d = []

        for i in range(x.shape[1]):
            if i == 0:
                out = torch.zeros_like(x[:, 0])
                b, c, h, w = out.size()

                s = torch.zeros((b, self.filters_s, h // self.factor, w // self.factor)).cuda()
                d = torch.zeros((b, self.filters_d, h // self.factor, w // self.factor)).cuda()

                if self.state is None:
                    state = torch.zeros_like(out[:, 0:1, ::self.factor, ::self.factor]).repeat(1, params["state dimension"], 1, 1)
                else:
                    state = self.state

                x_input = torch.cat((x[:, i].unsqueeze(0), x[:, i].unsqueeze(0)), dim=1)
                out, state, s_out, d_out, s, d = self.transformer(x_input, state, s, d, mode=self.mode)
            else:
                x_input = x[:, i-1: i+1]
                out, state, s_out, d_out, s, d = self.transformer(x_input, state, s, d, mode=self.mode)
        
            if params['type'] == 'deployment':
                self.state = state

            seq_hq.append(out)
            seq_s.append(s_out)
            seq_d.append(d_out)

        seq_hq = torch.stack(seq_hq, 1)
        seq_s = torch.stack(seq_s, 1)
        seq_d = torch.stack(seq_d, 1)
        return seq_hq, seq_s ,seq_d, y_s, y_d
