import numpy as np
import torch
import multiprocessing
from nltk.tokenize import TweetTokenizer
import logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


class SHAPexplainer:

    def __init__(self, model, tokenizer, words_dict, words_dict_reverse):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.tweet_tokenizer = TweetTokenizer()
        self.words_dict = words_dict
        self.words_dict_reverse = words_dict_reverse
        
    def parallel_pred(self, tokens):
                predictions = []
                for i, token in enumerate(tokens):
                    outputs = self.model(input_ids=token.unsqueeze(0))
                    predictions.append(outputs.detach().cpu().numpy())
                return predictions
            
    def predict(self, indexed_words):
        # self.model.to(self.device)
        # print("entering predict")
        # print(f"{indexed_words=}")
        sentences = []
        for i, x in enumerate(indexed_words):
            if i % 1000 == 0:
                print(f"\rProcessing sentences iteration: {i + 1}/{len(indexed_words)}", end="")
            sentences.append([self.words_dict[xx] if xx != 0 else "" for xx in x])
        print()  # To move to the next line after the loop
        
        # print(f"{sentences=}")
        indexed_tokens, tokenized_text, _ = self.tknz_to_idx(sentences)
        # print("indexed_tokens", indexed_tokens)
        # print(tokenized_text)

        tokens_tensor = torch.tensor(indexed_tokens)
        # print("tokens_tensor", tokens_tensor)
        with torch.no_grad():
            predictions = []
            
            def parallel_execute(complete_data):
                num_cores = 4
                chunk_size = len(complete_data) // num_cores
                chunks = [complete_data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_cores)]
                if len(complete_data) % num_cores != 0:
                    chunks[-1].extend(complete_data[num_cores * chunk_size:])
                
                with multiprocessing.Pool(num_cores) as pool:
                    results = pool.map(self.parallel_pred, chunks)
                return [item for sublist in results for item in sublist]
            
            predictions = parallel_execute(tokens_tensor)
        final = [self.softmax(x) for x in predictions]
        # print(final)
        return np.array(final)

    def softmax(self, it):
        exps = np.exp(np.array(it))
        return exps / np.sum(exps)

    def split_string(self, string):
        data_raw = self.tweet_tokenizer.tokenize(string)
        data_raw = [x for x in data_raw if x not in ".,:;'"]
        return data_raw

    def tknz_to_idx(self, train_data, MAX_SEQ_LEN=None):
        # print("entering tknz_to_idx")
        # print(f"{train_data=}")
        
        tokenized_nopad = []
        for i, text in enumerate(train_data):
            print(f"\rProcessing data to token iteration: {i + 1}/{len(train_data)}", end="")
            tokenized_nopad.append(self.tokenizer.tokenize(" ".join(text)))
        print()  # To move to the next line after the loop
        
        if not MAX_SEQ_LEN:
            MAX_SEQ_LEN = min(max(len(x) for x in train_data), 512)
        tokenized_text = [['[PAD]', ] * MAX_SEQ_LEN for _ in range(len(tokenized_nopad))]
        
        for i in range(len(tokenized_nopad)):
            if i%1000 == 0:
                print(f"\rProcessing adding padding iteration: {i + 1}/{len(tokenized_nopad)}", end="")
            tokenized_text[i][0:len(tokenized_nopad[i])] = tokenized_nopad[i][0:MAX_SEQ_LEN]
        print()
        
        indexed_tokens = []
        for i, tt in enumerate(tokenized_text):
            print(f"\rProcessing tokens to ids iteration: {i + 1}/{len(tokenized_text)}", end="")
            indexed_tokens.append(self.tokenizer.convert_tokens_to_ids(tt))
        print()  # To move to the next line after the loop
        indexed_tokens = np.array([np.array(tt) for tt in indexed_tokens])
        
        # print("done tokenizing")
        return indexed_tokens, tokenized_text, MAX_SEQ_LEN

    def dt_to_idx(self, data, max_seq_len=None):
        # print("entering dt_to_idx")
        idx_dt = [[self.words_dict_reverse[xx] for xx in x] for x in data]
        if not max_seq_len:
            max_seq_len = min(max(len(x) for x in idx_dt), 512)
            # print("max_seq_len", max_seq_len)
        for i, x in enumerate(idx_dt):
            if len(x) < max_seq_len:
                idx_dt[i] = x + [0] * (max_seq_len - len(x))
            elif len(x) > max_seq_len:
                idx_dt[i] = x[0:max_seq_len]   
        for i in idx_dt:
            if len(i) != max_seq_len:
                raise ValueError("Error in padding: Length mismatch during padding process")
        return np.array(idx_dt), max_seq_len
