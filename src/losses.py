import torch

def masked_mse_loss(pred_noise, actual_noise, mask):
    """
    Masked Mean Squared Error loss.
    Used to calculate the error between the predicted noise
    and the actual noise.
    Masked to only account for valid postions: those not
    corresponding to null coordinates or padded positions.
    """
    mse_loss = torch.nn.MSELoss(reduction='none')
    loss = mse_loss(pred_noise, actual_noise)
    loss = (loss * mask.float()).sum() / mask.sum()
    return loss
