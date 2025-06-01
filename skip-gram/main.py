"""
"""
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.optim as optim
import torch.nn as nn
import data_setup, engine, model_builder, utils

torch.manual_seed(42)
torch.cuda.manual_seed(42)

dataloader, neg_sampler, word_to_index = data_setup.get_dataloader('data/quora.txt')
NUM_STEPS = 750
device = "cuda" if torch.cuda.is_available() else "cpu"


vocab_size = len(word_to_index)
embedding_dim = 32
num_negatives = 15
model_0 = model_builder.SkipGramModelWithNegSampling(vocab_size, embedding_dim).to(device)

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # тот самый лосс, похож на логлосс
optimizer = optim.Adam(model_0.parameters(), lr=0.05)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=30) # штука, которая будет уполовинивать (* factor = 1/2) lr при отсутствии улучшений в течение 150 эпох

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_0 
model_0_results = engine.train(model=model_0, 
                        dataloader=dataloader,
                        neg_sampler=neg_sampler,
                        loss_fn=loss_fn,
                        num_negatives=num_negatives,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        steps=NUM_STEPS,
                        device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model
utils.save_model(model=model_0,
           target_dir="models",
           model_name="skip-gram with negative sampling.pth")
