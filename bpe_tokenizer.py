from joblib import Parallel, delayed
from typing import BinaryIO, List, Tuple
import regex as re
from collections import Counter, defaultdict
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='bpe_tokenizer.log', filemode='w')

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

#the single process for pretokenization
def sj_bpe(input_path, start, end, special_token):
    """
    Reads a chunk of the input file and pretokenizes it.
    """
    with open(input_path, "rb") as file:
        #exclude special token from chunk
        file.seek(start+len(special_token))
        chunk = file.read(end - start - len(special_token)).decode("utf-8", errors="ignore")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = re.findall(PAT, chunk)
    tokens = [token.encode("utf-8") for token in tokens if token.strip()]
    #freq of tokens after pretokenization
    token_counts = Counter(tokens)
    return token_counts

#multiprocessing version of pretokenization
def get_token_counts_mp(input_path, num_process, special_token) -> Counter:
    num_file_splits = 12
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_file_splits, special_token.encode("utf-8"))
        
    logging.info(f"Chunk boundaries: {boundaries}")
    logging.info(f"start multiprocessing with {num_process} processes")
    res = Parallel(n_jobs=num_process, verbose=10)(
    delayed(sj_bpe)(input_path, boundaries[i], boundaries[i+1],
                    special_token)for i in range(len(boundaries)-1))
    #the aggregated counter for token
    total_res = Counter()
    for token_counter in res:
        total_res.update(token_counter)
    logging.info(f"Total tokens found: {len(total_res)}")
    return total_res

def merge(token_dict: dict, best_pair, new_id) -> dict:
    res = {}
    for token in token_dict.keys():
        i = 0
        new_token = []
        while i < len(token):
            if token[i] == best_pair[0][0] and i < len(token) - 1 and token[i+1] == best_pair[1][0]:
                new_token.append(new_id)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        res[tuple(new_token)] = token_dict[token]
    return res




def mp_merge(token_dict: dict, best_pair, new_id, num_process) -> dict:
    """
    Merges the best pair of tokens in the token_dict and returns the updated dictionary.
    """
    res = {}
    token_item = list(token_dict.items())
    k, m = divmod(len(token_item), num_process)
    res_all = Parallel(n_jobs=num_process, verbose=10)(
        delayed(merge)(dict(token_item[i*k + min(i, m):(i+1)*k + min(i+1, m)]),
                        best_pair, new_id)
        for i in range(num_process)
    )
    for item in res_all:
        res.update(item)
    return res


def bpe_train(token_counts, vocab_size, special_tokens, initial_vocab):
    vocab = initial_vocab.copy()
    merges = []
    token_dict = {}
    for key, val in token_counts.items():
        token_dict[tuple(key)] = val
    # Merge tokens until the vocabulary size is reached
    print (len(vocab), vocab_size)
    while len(vocab) < vocab_size:
        pairs = defaultdict(int)
        # Count pairs of adjacent tokens
        for token, count in token_dict.items():
            if len(token) == 1: 
                continue  # Skip single-character tokens
            for i in range(len(token) - 1):
                pair = (token[i:i+1], token[i+1:i+2])
                pairs[tuple(pair)] += count
        if not pairs:
            break
        # Find the most common pair, best_pair: [bytes,bytes]
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] < 2:
            break  # Stop if no pairs are frequent enough
        # Merge the best pair



        new_token = vocab[best_pair[0][0]] + vocab[best_pair[1][0]]
        
        # Add the new token to the vocabulary
        if new_token not in vocab.values():
            new_token_id = len(vocab)
        vocab[new_token_id] = new_token
        #print (">>>", new_token, new_token_id,len(vocab))
        logging.info(f"Merging {vocab[best_pair[0][0]]} and {vocab[best_pair[1][0]]} into {new_token.decode('utf-8')} with ID {new_token_id}")
        #print ("-------------------", len(vocab), vocab_size)
        
        merges.append((vocab[best_pair[0][0]], vocab[best_pair[1][0]]))
        
        
        # Update token counts
        #token_dict = merge(token_dict, best_pair, new_token_id)
        # Use multiprocessing to merge tokens
        token_dict = mp_merge(token_dict, best_pair, new_token_id, num_process=4)
    return vocab, merges



def train_bpe_tokenizer(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        num_processes: int = 1
):
    """
    Tokenizes the input file using Byte Pair Encoding (BPE) and returns the vocabulary and tokenized data.
    input:
    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.
    num_processes: int The number of processes to use for parallel processing. Defaults to 1 (no parallelism).
    output:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation
    """
    assert isinstance(special_tokens, list), "Special tokens must be a list of strings"
    assert all(isinstance(token, str) for token in special_tokens), "All special tokens must be strings"    
    assert num_processes > 0, "Number of processes must be greater than 0"

    #step1. pretokenize each chunk(with multiprocessing )
    #step2. merge pretokenized chunks
    #step3. train BPE tokenizer on merged pretokenized data
    #step4. return vocabulary and merges

    #init vocab with initial byte vocabulary
    vocab = {idx: bytes([idx]) for idx in range(256)}
    # Add special tokens to the vocabulary
    for token in special_tokens:
        if token not in vocab.values():
            new_id = len(vocab)
            vocab[new_id] = token.encode("utf-8")
            logging.info(f"Adding special token '{token}' with ID {new_id}")

    #step1. pretokenize each chunk(with multiprocessing)
    logging.info("Starting pretokenization...")
    total_res = get_token_counts_mp(input_path, num_processes, special_tokens[0])

    logging.info("Pretokenization complete.")

    vocab, merge = bpe_train(total_res, vocab_size, special_tokens, vocab)
    return vocab, merge
