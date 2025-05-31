"""
"""
import torch
import numpy as np
from tqdm.auto import tqdm as tqdma

def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    neg_sampler,
    num_negatives: int,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    device: torch.device,
    steps: int,
    batch_size: int = 5000,
):
    """
    """
    model.train()
    step = 0
    loss_history = []
    pos_labels = torch.ones(batch_size).to(device)
    neg_labels = torch.zeros(batch_size, num_negatives).to(device)
    while step <= steps:
        for target, context in dataloader:
            if step > steps:
                break
            center_words = target.long().to(device)
            pos_context_words = context.long().to(device)
            neg_context = torch.LongTensor(np.array([np.array(list(neg_sampler)) for t in center_words]))
            neg_context_words = neg_context.long().to(device)
    
            optimizer.zero_grad()
            pos_scores, neg_scores = model(
                center_words, pos_context_words, neg_context_words
            )
            loss_pos = loss_fn(pos_scores, pos_labels)
            loss_neg = loss_fn(neg_scores, neg_labels)
    
            loss = loss_pos + loss_neg
            loss.backward()
            optimizer.step()
    
            loss_history.append(loss.item())
            lr_scheduler.step(loss_history[-1])
    
            if step % 10 == 0:
                print(f"Step {step}, Loss: {np.mean(loss_history[-10:])}, learning rate: {lr_scheduler._last_lr}")
            step += 1

    return np.mean(loss_history)
