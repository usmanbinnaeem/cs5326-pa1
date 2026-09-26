from __future__ import annotations

import torch


def save_checkpoint(model, optimizer, next_step, train_generator, val_generator, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_step": int(next_step),
            "train_generator": train_generator.get_state(),
            "val_generator": val_generator.get_state(),
        },
        out,
    )


def load_checkpoint(src, model, optimizer, train_generator, val_generator):
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_generator.set_state(checkpoint["train_generator"])
    val_generator.set_state(checkpoint["val_generator"])
    return int(checkpoint["next_step"])


def export_model_state(model, out):
    state = {
        name: tensor.detach().cpu().to(torch.float16)
        if tensor.is_floating_point()
        else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, out)


def load_model_state(src):
    state = torch.load(src, map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    return state
