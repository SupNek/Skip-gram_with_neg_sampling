"""
Contains functionality for creating PyTorch DataLoader for 
text data.
"""
import data_preprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
import os

NUM_WORKERS = os.cpu_count()

# def get_one_random_sample_with_negatives(context_pairs,
#                                          keep_prob_array,
#                                          negative_sampling_prob_array,
#                                          num_negatives):
#     """Returns one sample of center, context and negative samples

#     Parameters:
#     - context_pairs: array of tuples (central_word_idx, context_word_idx)
#     - keep_prob_array: array of probabilities for every allowed word to keep
#     - negative_sampling_prob_array: array of probabilities for every allowed word
#         to use as negative sample

#     Returns:
#     A tuple of center and context words with negative samples.
#     In the form (center, context, neg_sample).
#     """
#     while True:
#         center, context = random.choice(context_pairs)
#         if random.random() < keep_prob_array[center]:
#             neg_sample = np.random.choice(
#                 range(len(negative_sampling_prob_array)),
#                 size=num_negatives,
#                 p=negative_sampling_prob_array,
#             )
#             return (center, context, neg_sample)

class Word2VecDataset(Dataset):
    def __init__(self, context_pairs):
        self.context_pairs = context_pairs

    def __len__(self):
        return len(self.context_pairs)

    def __getitem__(self, idx):
        return self.context_pairs[idx]

def get_dataloader(
    data_path: str,
    batch_size: int = 5000,
    # num_workers: int=NUM_WORKERS,
    num_workers: int=1,
    min_count: int = 5,
    window_radius: int = 5,
    num_negatives: int = 15
):
    """
    
    """
    word_to_index, context_pairs, keep_prob_array, negative_sampling_prob_array = data_preprocessing.preprocessing(data_path,
                                                                                                min_count,
                                                                                                window_radius)
    dataset = Word2VecDataset(context_pairs)
    neg_sampler = WeightedRandomSampler(negative_sampling_prob_array, num_negatives, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader, neg_sampler, word_to_index
