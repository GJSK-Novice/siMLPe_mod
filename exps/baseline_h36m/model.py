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
        nn.init.xavier_uniform_(self.motion_fc_in.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_in.bias, 0)
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
        encoder_out, rnn_states = self.encoder(x)

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
        _, rnn_states = self.endecoder(x)
        
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
        _, (hidden_states, cell_states) = self.endecoder(x)
        
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

        # self.temporal_fc1 = nn.Linear(window_size, window_size)
        if (config.motion_rnn.num_temp_blocks > 1):
            self.temporal_fc1 = nn.Sequential(*[
                nn.Linear(window_size, window_size)
                for i in range(config.motion_rnn.num_temp_blocks-1)])
        else:
            self.temporal_fc1 = nn.Identity()

        self.temporal_fc_last = nn.Linear(window_size, 1)

        if config.motion_rnn.local_spatial_fc:
            self.spatial_fc1 = nn.Linear(window_size, window_size)
        else:
            self.spatial_fc1 = nn.Identity()
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
        nn.init.xavier_uniform_(self.spatial_fc1.weight, gain=1e-8)
        nn.init.constant_(self.spatial_fc1.bias, 0)
        nn.init.xavier_uniform_(self.spatial_fc.weight, gain=1e-8)
        nn.init.constant_(self.spatial_fc.bias, 0)
        
    def forward(self, x):
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Encoder: start with zero hidden states
        if self.config.motion_rnn.use_gru:
            encoder_out, rnn_states = self.endecoder(x)
        else:
            encoder_out, (rnn_states, cell_states) = self.endecoder(x)
        
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
            _decoder_out = encoder_window
            _decoder_out = self.spatial_fc1(_decoder_out)
            _decoder_out = self.temporal_fc1(self.arr0(_decoder_out))
            _decoder_out = self.arr0(self.temporal_fc_last(_decoder_out))
            _decoder_out = self.spatial_fc(_decoder_out)

            if self.config.motion_rnn.recursive_residual:
                # Residual method 1 (recursive residual; same as in 2017 Martinez paper):
                new_frame = self.spatial_norm(_decoder_out) + decoder_input
            else:
                # Residual method 2 (residual from the last input frame):
                new_frame = self.spatial_norm(_decoder_out) + last_input_frame

            output_frames[:, frame_id:frame_id+1, :] = new_frame
            decoder_input = new_frame  # Next input is current output

        return output_frames

class siMLPe_mini(nn.Module):
    def __init__(self, config, window_size, concatenated_dim):
        self.config = copy.deepcopy(config)
        super(siMLPe_mini, self).__init__()
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = mlp.TransMLP(
                dim=self.config.motion_mlp.hidden_dim,
                seq=concatenated_dim,
                use_norm=True,
                use_spatial_fc=False,
                num_layers=1,
                layernorm_axis='spatial',
            )

        self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        self.temporal_merge_fc = nn.Linear(concatenated_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_in.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_in.bias, 0)
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)
        nn.init.xavier_uniform_(self.temporal_merge_fc.weight, gain=1e-8)
        nn.init.constant_(self.temporal_merge_fc.bias, 0)

    def forward(self, motion_input):

        motion_feats = self.motion_fc_in(motion_input)
        motion_feats = self.arr0(motion_feats)

        # MLP block input should be [B,C,T]
        motion_feats = self.motion_mlp(motion_feats)

        motion_feats = self.arr1(motion_feats)
        motion_feats = self.motion_fc_out(motion_feats)

        motion_feats = self.arr1(self.temporal_merge_fc(self.arr1(motion_feats)))

        return motion_feats

class SlidingRNN_v2(nn.Module):
    def __init__(self, config, state_size, num_layers, window_size):
        self.config = copy.deepcopy(config)
        super(SlidingRNN_v2, self).__init__()

        self.window_size = window_size
        input_size = config.motion.dim
        self.mlp_mini = siMLPe_mini(config, window_size, concatenated_dim=window_size+1)

        if config.motion_rnn.use_gru:
            self.endecoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        else:
            # let encoder and decoder share weights
            self.endecoder = nn.LSTM(input_size, state_size, num_layers, batch_first=True)

        self.arr0 = Rearrange('b n d -> b d n')
        self.fc_decoder = nn.Linear(state_size, config.motion.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_decoder.weight, gain=1e-8)
        nn.init.constant_(self.fc_decoder.bias, 0)
        
    def forward(self, x):
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Encoder: start with zero hidden states
        if self.config.motion_rnn.use_gru:
            encoder_out, rnn_states = self.endecoder(x)
        else:
            encoder_out, (rnn_states, cell_states) = self.endecoder(x)
        
        # Decoder initialization
        last_input_frame = x[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        decoder_input = last_input_frame.clone()

        # size = [B, window_size, state_size]
        out_frame_window = x[:, -self.window_size:, :]
        
        output_frames = torch.zeros(B, T, C).cuda()
        for frame_id in range(T):
            # Decoder: # [B, 1, C]
            if self.config.motion_rnn.use_gru:
                decoder_out, rnn_states = self.endecoder(decoder_input, rnn_states)
            else:
                decoder_out, (rnn_states, cell_states) = self.endecoder(decoder_input, (rnn_states, cell_states))

            # decode [B,1,H] to [B,1,C]
            _decoder_out = decoder_out
            _decoder_out = self.fc_decoder(_decoder_out)
            mlp_input = torch.cat([out_frame_window, _decoder_out], dim=1)
            
            if self.config.motion_rnn.recursive_residual:
                # Residual method 1 (recursive residual; same as in 2017 Martinez paper):
                new_frame = self.mlp_mini(mlp_input) + decoder_input
            else:
                # Residual method 2 (residual from the last input frame):
                new_frame = self.mlp_mini(mlp_input) + last_input_frame

            output_frames[:, frame_id:frame_id+1, :] = new_frame
            # Next input is current output
            decoder_input = new_frame
            # Sliding frame window
            out_frame_window = torch.cat([out_frame_window[:, 1:, :], new_frame], dim=1)

        return output_frames

class SlidingRNN_v3(nn.Module):
    def __init__(self, config, state_size, num_layers, window_size):
        self.config = copy.deepcopy(config)
        super(SlidingRNN_v3, self).__init__()

        self.window_size = window_size
        input_size = config.motion.dim
        concatenated_dim=config.motion.h36m_input_length_dct+1
        self.mlp_mini = siMLPe_mini(config, window_size, concatenated_dim)

        if config.motion_rnn.use_gru:
            self.endecoder = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        else:
            # let encoder and decoder share weights
            self.endecoder = nn.LSTM(input_size, state_size, num_layers, batch_first=True)

        self.arr0 = Rearrange('b n d -> b d n')
        self.fc_decoder = nn.Linear(state_size, config.motion.dim)

        self.spatial_fc = nn.Linear(config.motion.dim, config.motion.dim)
        self.temporal_merge_fc = nn.Linear(concatenated_dim, 1)
        self.arr1 = Rearrange('b d n -> b n d')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_decoder.weight, gain=1e-8)
        nn.init.constant_(self.fc_decoder.bias, 0)
        
    def forward(self, x):
        B, T, C = x.size()
        assert(C == self.config.motion.dim)
        
        # Decoder initialization
        last_input_frame = x[:, -1:, :]  # Last time step of input as initial input [B, 1, C]
        last_rnn_input = last_input_frame.clone()

        # size = [B, window_size, state_size]
        out_window_short_term = x[:, -self.window_size:, :].clone()
        out_window_long_term = x.clone()

        output_frames = torch.zeros(B, T, C).cuda()
        for frame_id in range(T):
            # Encoder: start with zero hidden states
            if self.config.motion_rnn.use_gru:
                encoder_out, rnn_states = self.endecoder(out_window_short_term)
            else:
                encoder_out, (rnn_states, cell_states) = self.endecoder(out_window_short_term)

            # decode [B,1,H] to [B,1,C]
            _decoder_out = self.fc_decoder(encoder_out[:,-1:,:])
            mlp_input = torch.cat([out_window_long_term, _decoder_out], dim=1)
            
            # mlp_mini
            mlp_output = self.mlp_mini(mlp_input)
            # mlp_output = self.spatial_fc(mlp_input)
            # mlp_output = self.arr1(self.temporal_merge_fc(self.arr1(mlp_output)))

            if self.config.motion_rnn.recursive_residual:
                # Residual method 1 (recursive residual; same as in 2017 Martinez paper):
                new_frame = mlp_output + last_rnn_input
            else:
                # Residual method 2 (residual from the last input frame):
                new_frame = mlp_output + last_input_frame

            output_frames[:, frame_id:frame_id+1, :] = new_frame
            # Next input is current output
            last_rnn_input = new_frame
            # Sliding frame window
            out_window_short_term = torch.cat([out_window_short_term[:, 1:, :], new_frame], dim=1)
            out_window_long_term = torch.cat([out_window_long_term[:, 1:, :], new_frame], dim=1)

        return output_frames

class GRU_classic(nn.Module):
    def __init__(self, config, state_size, num_layers):
        self.config = copy.deepcopy(config)
        super(GRU_classic, self).__init__()

        self.state_size = state_size
        input_size = config.motion.dim
        self.rnn = nn.GRU(input_size, state_size, num_layers, batch_first=True)

        self.fc0_temp = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        self.fc1 = nn.Linear(state_size, config.motion.dim)

        self.arr0 = Rearrange('b n d -> b d n')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0_temp.weight, gain=1e-8)
        nn.init.constant_(self.fc0_temp.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1e-8)
        nn.init.constant_(self.fc1.bias, 0)
        
    def forward(self, x):
        # rnn: start with zero hidden states
        encoder_out = self.arr0(self.fc0_temp(self.arr0(x)))
        rnn_out, rnn_states = self.rnn(encoder_out)
        decoder_out = self.fc1(rnn_out)
        return decoder_out

class siMLPe_RNN(nn.Module):
    def __init__(self, config, rnn_state_size, rnn_layers, num_blocks, window_size):
        super().__init__()
        self.rnn = SlidingRNN_v3(config, rnn_state_size, rnn_layers, window_size)

    def forward(self, x):
        _x = self.rnn(x)
        return _x