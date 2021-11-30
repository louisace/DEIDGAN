import argparse


def get_parameters():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=128)

    # Training setting
    parser.add_argument('--max_epoch', type=int, default=200, help='how many times to update the model')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_update_step', type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--n_layers_D', type=int, default=3)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--gan_mode", type=str, default='hinge')
    parser.add_argument("--image_path", type=str, default='')
    parser.add_argument("--mask_path", type=str, default='')

    # training Path
    parser.add_argument('--log_path', type=str, default='')
    parser.add_argument('--tensorboard_path', type=str, default='')
    parser.add_argument('--model_save_path', type=str, default='')
    parser.add_argument('--samples_path', type=str, default='')

