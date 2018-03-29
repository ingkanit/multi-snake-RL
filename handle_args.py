# Argument handler

import argparse

def handle_args(test_model, save_gif):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train', action='store_true', default=not test_model, help="Train network weights")
    parser.add_argument('--savegif', action='store_true', default=save_gif, help="Save GIF video of first 3 episodes")
    
    args = parser.parse_args()
    
    test_model = not args.train
    save_gif = args.savegif
    
    return test_model, save_gif