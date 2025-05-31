
def subsample_frequent_words(word_count_dict, threshold=1e-5):
    """
    Calculates the subsampling probabilities for words based on their frequencies.

    This function is used to determine the probability of keeping a word in the dataset
    when subsampling frequent words. The method used is inspired by the subsampling approach
    in Word2Vec, where each word's frequency affects its probability of being kept.

    Parameters:
    - word_count_dict (dict): A dictionary where keys are words and values are the counts of those words.
    - threshold (float, optional): A threshold parameter used to adjust the frequency of word subsampling.
                                   Defaults to 1e-5.

    Returns:
    - dict: A dictionary where keys are words and values are the probabilities of keeping each word.
    """
    all_w_count = sum(word_count_dict.values())
    freq = {word: word_count_dict[word] / all_w_count for word in word_count_dict}
    prob = {word: (threshold / freq[word]) ** 0.5 for word in freq}
    return prob

def get_negative_sampling_prob(word_count_dict):
    """
    Calculates the negative sampling probabilities for words based on their frequencies.

    This function adjusts the frequency of each word raised to the power of 0.75, which is
    commonly used in algorithms like Word2Vec to moderate the influence of very frequent words.
    It then normalizes these adjusted frequencies to ensure they sum to 1, forming a probability
    distribution used for negative sampling.

    Parameters:
    - word_count_dict (dict): A dictionary where keys are words and values are the counts of those words.

    Returns:
    - dict: A dictionary where keys are words and values are the probabilities of selecting each word
            for negative sampling.
    """
    all_w_count = sum(word_count_dict.values())
    freq = {word: (word_count_dict[word] / all_w_count) ** 0.75 for word in word_count_dict}
    Z = sum(freq.values())
    return {word: freq[word] / Z for word in freq}

def preprocessing(
    data_path: str,
    min_count: int = 5,
    window_radius: int = 5
):
    """Preprocess data and return different word sampling arrays and dictionaries

    Takes in a data directory path and returns context pairs array,
    array probabilities of negative sampling and array of probabilities of keeping words

    Parameters:
    - data_path: Path to data.
    - min_count: min number of word occurance in data to add to vocabulary.
    - window_radius: number of words to add to the context before and after central word.

    Returns:
    - word_to_index: mapping of allowed words in data to indexes
    - context_pairs: array of tuples (central_word_idx, context_word_idx)
    - keep_prob_array: array of probabilities for every allowed word to keep
    - negative_sampling_prob_array: array of probabilities for every allowed word
        to use as negative sample
    """
    data = list(open(data_path, encoding="utf-8"))

    tokenizer = WordPunctTokenizer()
    data_tok = [
        tokenizer.tokenize(
            line.translate(str.maketrans("", "", string.punctuation)).lower()
        )
        for line in data
    ] # генератор в котором токенизируем каждое предложение
    data_tok = [x for x in data_tok if len(x) >= 3] # оставляем только те, чьи длина больше 2 (т.е. минимум два слова в предложении)

    vocabulary_with_counter = Counter(chain.from_iterable(data_tok))

    word_count_dict = dict()
    for word, counter in vocabulary_with_counter.items():
        if counter >= min_count: # отбрасываем слова встречаемые реже 5 раз
            word_count_dict[word] = counter
    
    vocabulary = set(word_count_dict.keys())
    del vocabulary_with_counter

    word_to_index = {word: index for index, word in enumerate(vocabulary)} # (слово, индекс)
    index_to_word = {index: word for word, index in word_to_index.items()} # (индекс, слово)

    context_pairs = []
    for text in data_tok:
        for i, central_word in enumerate(text): # выбираем центральное слово
            context_indices = range(
                max(0, i - window_radius), min(i + window_radius, len(text))
            ) # сбор контекста к центральному слову
            for j in context_indices:
                if j == i:
                    continue
                context_word = text[j]
                if central_word in vocabulary and context_word in vocabulary:
                    context_pairs.append(
                        (word_to_index[central_word], word_to_index[context_word]) # нашли пары разрешенных слов и добавили в массив
                    )

    keep_prob_dict = subsample_frequent_words(word_count_dict)
    negative_sampling_prob_dict = get_negative_sampling_prob(word_count_dict)

    # полученные массивы
    keep_prob_array = np.array(
        [keep_prob_dict[index_to_word[idx]] for idx in range(len(word_to_index))]
    )
    negative_sampling_prob_array = np.array(
        [
            negative_sampling_prob_dict[index_to_word[idx]]
            for idx in range(len(word_to_index))
        ]
    )

    return word_to_index, context_pairs, keep_prob_array, negetive_sampling_prob_array
