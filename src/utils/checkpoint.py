import torch

def save_checkpoint(state, filename):
    print("=> Saving checkpoint '{}'".format(filename))

    torch.save(state, filename)

def load_checkpoint(model, checkpoint_path):
    print("=> Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

