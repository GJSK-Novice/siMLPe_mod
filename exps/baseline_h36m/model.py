import copy

import torch
from torch import nn
from mlp import build_mlps
import mlp
from einops.layers.torch import Rearrange

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        # MLP block input should be [B,C,T]
        motion_feats = self.motion_mlp(motion_feats)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats

class build_rnns(nn.Module):
    def __init__(self, config, rnn_state_size, rnn_layers, num_blocks):
        super().__init__()
        self.rnns = nn.Sequential(*[
            RNN_block(config, rnn_state_size, rnn_layers)
            for i in range(num_blocks)])
    
        if config.motion_rnn.with_normalization:
            print("Using LayerNorm")
        else:
            print("Not using LayerNorm")

    def forward(self, x):
        x = self.rnns(x)
        return x


class RNN_block(nn.Module):

    def __init__(self, config, rnn_state_size, rnn_layers):
        super().__init__()

        self.rnn = Seq2SeqGRU_simple(config, rnn_state_size, rnn_layers)

        if config.motion_rnn.with_normalization:
            self.spatial_norm = mlp.LN_v2(dim=config.motion.dim)
        else:
            self.spatial_norm = nn.Identity()
        
        self.rnn_fc = nn.Linear(rnn_state_size, config.motion.dim)

    def forward(self, x):
        x_ = self.rnn(x)
        x_ = self.rnn_fc(x_)
        x_ = self.spatial_norm(x_)
        # x = x + x_
        return x_

class Seq2SeqGRU_simple(nn.Module):
    def __init__(self, config, state_size, num_layers):
        self.config = copy.deepcopy(config)
        super(Seq2SeqGRU_simple, self).__init__()

        self.state_size = state_size
        input_size = config.motion.dim
        # let encoder and decoder share weights
        # self.endecoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        # untie encoder and decoder
        self.encoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        self.decoder = nn.GRU(state_size, state_size, num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Encoder: start with zero hidden states
        encoder_out, rnn_states = self.encoder(x)  # hidden: [num_layers, B, state_size]

        # Decoder initialization
        # last_input_frame = x[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        last_input_frame = encoder_out[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        decoder_input = last_input_frame.clone()
        
        output_delta_frames = torch.zeros(B, T, self.state_size).cuda()
        for frame_id in range(T):
            # Decoder input is always the last input frame
            frame_delta, rnn_states = self.decoder(decoder_input, rnn_states)
            output_delta_frames[:, frame_id:frame_id+1, :] = frame_delta.clone()
            decoder_input = frame_delta.clone()  # Next input is current output

        return output_delta_frames

class Seq2SeqGRU(nn.Module):
    def __init__(self, config, state_size, num_layers):
        self.config = copy.deepcopy(config)
        super(Seq2SeqGRU, self).__init__()

        input_size = config.motion.dim

        # let encoder and decoder share weights
        self.endecoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)

        # let encoder and decoder not share weights
        # self.encoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        # self.decoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)

        self.fc0 = nn.Linear(state_size, config.motion.dim)
        # self.fc1 = nn.Linear(config.motion.dim, config.motion.dim)

        if config.motion_rnn.with_normalization:
            self.spatial_norm = mlp.LN_v2(dim=config.motion.dim)
        else:
            self.spatial_norm = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.weight, gain=1e-8)
        nn.init.constant_(self.fc0.bias, 0)
        # nn.init.xavier_uniform_(self.fc1.weight, gain=1e-8)
        # nn.init.constant_(self.fc1.bias, 0)
        
    def forward(self, x):
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Encoder: start with zero hidden states
        _, rnn_states = self.endecoder(x)  # hidden: [num_layers, B, state_size]
        
        # Decoder initialization
        last_input_frame = x[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        decoder_input = last_input_frame.clone()
        
        output_frames = torch.zeros(B, T, C).cuda()
        for frame_id in range(T):
            # Decoder
            frame_delta, rnn_states = self.endecoder(decoder_input, rnn_states)
            frame_delta = self.fc0(frame_delta)  # [B, 1, C]
            # frame_delta = self.fc1(frame_delta) + frame_delta  # [B, 1, C]

            if self.config.motion_rnn.recursive_residual:
                # Residual method 1 (recursive residual; same as in 2017 Martinez paper):
                new_frame = self.spatial_norm(frame_delta) + decoder_input
            else:
                # Residual method 2 (residual from the last input frame):
                new_frame = self.spatial_norm(frame_delta) + last_input_frame

            output_frames[:, frame_id:frame_id+1, :] = new_frame.clone()
            decoder_input = new_frame.clone()  # Next input is current output

        return output_frames

class Seq2SeqLSTM(nn.Module):
    def __init__(self, config, state_size, num_layers):
        self.config = copy.deepcopy(config)
        super(Seq2SeqLSTM, self).__init__()

        input_size = config.motion.dim

        # let encoder and decoder share weights
        self.endecoder = nn.LSTM(input_size, state_size, num_layers, batch_first=True)

        # let encoder and decoder not share weights
        # self.encoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        # self.decoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)

        self.fc0 = nn.Linear(state_size, config.motion.dim)
        # self.fc1 = nn.Linear(config.motion.dim, config.motion.dim)
        # self.fc2 = nn.Linear(config.motion.dim, config.motion.dim)
        # self.fc3 = nn.Linear(config.motion.dim, config.motion.dim)
        # self.fc4 = nn.Linear(config.motion.dim, config.motion.dim)

        if config.motion_rnn.with_normalization:
            self.spatial_norm = mlp.LN_v2(dim=config.motion.dim)
        else:
            self.spatial_norm = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.weight, gain=1e-8)
        nn.init.constant_(self.fc0.bias, 0)
        # nn.init.xavier_uniform_(self.fc1.weight, gain=1e-8)
        # nn.init.constant_(self.fc1.bias, 0)
        # nn.init.xavier_uniform_(self.fc2.weight, gain=1e-8)
        # nn.init.constant_(self.fc2.bias, 0)
        # nn.init.xavier_uniform_(self.fc3.weight, gain=1e-8)
        # nn.init.constant_(self.fc3.bias, 0)
        # nn.init.xavier_uniform_(self.fc4.weight, gain=1e-8)
        # nn.init.constant_(self.fc4.bias, 0)
        
    def forward(self, x):
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Encoder: start with zero hidden states
        _, (hidden_states, cell_states) = self.endecoder(x)  # hidden: [num_layers, B, state_size]
        
        # Decoder initialization
        last_input_frame = x[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        decoder_input = last_input_frame.clone()
        
        output_frames = torch.zeros(B, T, C).cuda()
        for frame_id in range(T):
            # Decoder
            frame_delta, (hidden_states, cell_states) = self.endecoder(decoder_input, (hidden_states, cell_states))
            frame_delta = self.fc0(frame_delta)  # [B, 1, C]
            # frame_delta = self.fc1(frame_delta)  # [B, 1, C]
            # frame_delta = self.fc2(frame_delta)  # [B, 1, C]
            # frame_delta = self.fc3(frame_delta)  # [B, 1, C]
            # frame_delta = self.fc4(frame_delta)  # [B, 1, C]

            if self.config.motion_rnn.recursive_residual:
                # Residual method 1 (recursive residual; same as in 2017 Martinez paper):
                new_frame = self.spatial_norm(frame_delta) + decoder_input
            else:
                # Residual method 2 (residual from the last input frame):
                new_frame = self.spatial_norm(frame_delta) + last_input_frame

            output_frames[:, frame_id:frame_id+1, :] = new_frame.clone()
            decoder_input = new_frame.clone()  # Next input is current output

        return output_frames


class SlidingGRU(nn.Module):
    def __init__(self, config, state_size, num_layers, window_size):
        self.config = copy.deepcopy(config)
        super(SlidingGRU, self).__init__()

        self.window_size = window_size
        input_size = config.motion.dim

        # let encoder and decoder share weights
        self.endecoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)

        # let encoder and decoder not share weights
        # self.encoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        # self.decoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)

        self.temporal_fc1 = nn.Linear(window_size, window_size)
        self.temporal_fc_last = nn.Linear(window_size, 1)
        self.spatial_fc = nn.Linear(state_size, config.motion.dim)

        self.arr0 = Rearrange('b n d -> b d n')

        if config.motion_rnn.with_normalization:
            self.spatial_norm = mlp.LN_v2(dim=config.motion.dim)
        else:
            self.spatial_norm = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.temporal_fc1.weight, gain=1e-8)
        nn.init.constant_(self.temporal_fc1.bias, 0)
        nn.init.xavier_uniform_(self.temporal_fc_last.weight, gain=1e-8)
        nn.init.constant_(self.temporal_fc_last.bias, 0)
        nn.init.xavier_uniform_(self.spatial_fc.weight, gain=1e-8)
        nn.init.constant_(self.spatial_fc.bias, 0)
        
    def forward(self, x):
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Encoder: start with zero hidden states
        encoder_out, rnn_states = self.endecoder(x)  # hidden: [num_layers, B, state_size]
        
        # Decoder initialization
        last_input_frame = x[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        decoder_input = last_input_frame.clone()

        # size = [B, window_size, state_size]
        encoder_window = encoder_out[:, -self.window_size:, :]
        
        output_frames = torch.zeros(B, T, C).cuda()
        for frame_id in range(T):
            # Decoder: # [B, 1, C]
            decoder_out, rnn_states = self.endecoder(decoder_input, rnn_states)

            # Sliding window
            encoder_window = torch.cat([encoder_window[:, 1:, :], decoder_out], dim=1)
            _decoder_out = self.temporal_fc1(self.arr0(encoder_window))
            _decoder_out = self.arr0(self.temporal_fc_last(_decoder_out))
            _decoder_out = self.spatial_fc(_decoder_out)

            # decoder_out = self.temporal_fc(decoder_out)  # [B, 1, C]
            # decoder_out = self.fc1(decoder_out) + decoder_out  # [B, 1, C]

            if self.config.motion_rnn.recursive_residual:
                # Residual method 1 (recursive residual; same as in 2017 Martinez paper):
                new_frame = self.spatial_norm(_decoder_out) + decoder_input
            else:
                # Residual method 2 (residual from the last input frame):
                new_frame = self.spatial_norm(_decoder_out) + last_input_frame

            output_frames[:, frame_id:frame_id+1, :] = new_frame
            decoder_input = new_frame  # Next input is current output

        return output_frames

class SlidingRNN(nn.Module):
    def __init__(self, config, state_size, num_layers, window_size):
        self.config = copy.deepcopy(config)
        super(SlidingRNN, self).__init__()

        self.window_size = window_size
        input_size = config.motion.dim

        if config.motion_rnn.use_gru:
            self.endecoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        else:
            # let encoder and decoder share weights
            self.endecoder = nn.LSTM(input_size, state_size, num_layers, batch_first=True)

        # let encoder and decoder not share weights
        # self.encoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        # self.decoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)

        # self.spatial_fc1 = nn.Linear(state_size, state_size)
        self.temporal_fc = nn.Linear(window_size, 1)
        self.spatial_fc = nn.Linear(state_size, config.motion.dim)

        self.arr0 = Rearrange('b n d -> b d n')
        # self.arr_flat = Rearrange('b n d -> b (n d)')
        # self.arr_deflat = Rearrange('b (n d) -> b n d', n=1)

        if config.motion_rnn.with_normalization:
            self.spatial_norm = mlp.LN_v2(dim=config.motion.dim)
        else:
            self.spatial_norm = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.spatial_fc1.weight, gain=1e-8)
        # nn.init.constant_(self.spatial_fc1.bias, 0)
        nn.init.xavier_uniform_(self.temporal_fc.weight, gain=1e-8)
        nn.init.constant_(self.temporal_fc.bias, 0)
        nn.init.xavier_uniform_(self.spatial_fc.weight, gain=1e-8)
        nn.init.constant_(self.spatial_fc.bias, 0)
        
    def forward(self, x):
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Encoder: start with zero hidden states
        if self.config.motion_rnn.use_gru:
            encoder_out, rnn_states = self.endecoder(x)
        else:
            encoder_out, (rnn_states, cell_states) = self.endecoder(x)  # hidden: [num_layers, B, state_size]
        
        # Decoder initialization
        last_input_frame = x[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        decoder_input = last_input_frame.clone()

        # size = [B, window_size, state_size]
        encoder_window = encoder_out[:, -self.window_size:, :]
        
        output_frames = torch.zeros(B, T, C).cuda()
        for frame_id in range(T):
            # Decoder: # [B, 1, C]
            if self.config.motion_rnn.use_gru:
                decoder_out, rnn_states = self.endecoder(decoder_input, rnn_states)
            else:
                decoder_out, (rnn_states, cell_states) = self.endecoder(decoder_input, (rnn_states, cell_states))

            # Sliding window
            encoder_window = torch.cat([encoder_window[:, 1:, :], decoder_out], dim=1)
            # _decoder_out = self.spatial_fc1(encoder_window)
            _decoder_out = self.arr0(self.temporal_fc(self.arr0(encoder_window)))
            _decoder_out = self.spatial_fc(_decoder_out)

            # _decoder_out = self.arr_deflat(self.spatial_fc(self.arr_flat(encoder_window)))

            # decoder_out = self.temporal_fc(decoder_out)  # [B, 1, C]
            # decoder_out = self.fc1(decoder_out) + decoder_out  # [B, 1, C]

            if self.config.motion_rnn.recursive_residual:
                # Residual method 1 (recursive residual; same as in 2017 Martinez paper):
                new_frame = self.spatial_norm(_decoder_out) + decoder_input
            else:
                # Residual method 2 (residual from the last input frame):
                new_frame = self.spatial_norm(_decoder_out) + last_input_frame

            output_frames[:, frame_id:frame_id+1, :] = new_frame
            decoder_input = new_frame  # Next input is current output

        return output_frames

class siMLPe_RNN(nn.Module):
    def __init__(self, config, rnn_state_size, rnn_layers, num_blocks, window_size):
        super().__init__()
        # self.rnn = nn.Sequential(*[
        #     Seq2SeqLSTM(config, rnn_state_size, rnn_layers)
        #     for i in range(num_blocks)])
        self.rnn = SlidingRNN(config, rnn_state_size, rnn_layers, window_size)

    def forward(self, x):
        _x = self.rnn(x)
        return _x