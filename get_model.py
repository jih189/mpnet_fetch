''' Code to train the model.
'''
import torch
import torch.nn as nn
import os.path as osp

import argparse 

from mpnet_models import MLP, Encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./paths/')
    parser.add_argument('--pt_dir', type=str, default='./pt_dir/')
    parser.add_argument('--enc_input_size', type=int, default=6000)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_output_size', type=int, default=7) # number of joint

    args = parser.parse_args()

    # Define the models.
    mlp = MLP(args.enc_output_size+args.mlp_output_size*2, args.mlp_output_size)
    encoder = Encoder(args.enc_input_size, args.enc_output_size)

    checkpoint = torch.load(osp.join(args.log_dir, 'model_499.pkl'))
    mlp.load_state_dict(checkpoint['mlp_state'])
    encoder.load_state_dict(checkpoint['encoder_state'])

    # if eval mode is set, then the model will be deterministic
    #mlp.eval()
    encoder.eval()

    encoder_example = torch.rand(1, args.enc_input_size)
    mlp_example = torch.rand(1, args.mlp_output_size + args.mlp_output_size + args.enc_output_size)

    encoder_model = torch.jit.trace(encoder, encoder_example)
    mlp_model = torch.jit.trace(mlp, mlp_example)

    encoder_model.save(osp.join(args.pt_dir, 'encoder_model.pt'))
    mlp_model.save(osp.join(args.pt_dir, 'mlp_model.pt'))
