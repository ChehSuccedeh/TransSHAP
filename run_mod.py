import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from nltk.tokenize import TweetTokenizer
# Bag of words
train_data = pd.read_json("../code_classification/data/test_dataset.jsonl", lines=True)
train_data = list(train_data["code"])
# print(np.array(train_data).shape)
tknzr = TweetTokenizer()
complete_data = train_data + ["def ciao():\n\tend = idk() + random + \"return_\"\nreturn", "import mod_name from './mod_name';\nvar value=mod_name+1;\nexport default value;"]
bag_of_words = set([xx for x in complete_data for xx in tknzr.tokenize(x)])
# print(bag_of_words)

print("Finished loading tokenizer and bag of words")


class SCForShap(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        output = super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
        return output[0]

pretrained_model = "huggingface/CodeBERTa-language-id"
tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
model = SCForShap.from_pretrained(pretrained_model)

# Test texts
t1 = "def ciao():\n\tend = idk() + random + \"return_\"\nreturn" #neutral
t2 = "import mod_name from './mod_name';\nvar value=mod_name+1;\nexport default value;" #negative

texts = [t1, t2]

import shap
import random
import logging
import matplotlib.pyplot as plt
from explainers.SHAP_for_text import SHAPexplainer
from explainers import visualize_explanations
logging.getLogger("shap").setLevel(logging.WARNING)
shap.initjs()

words_dict = {0: None}
words_dict_reverse = {None: 0}
for h, hh in enumerate(bag_of_words):
    words_dict[h + 1] = hh
    words_dict_reverse[hh] = h + 1

predictor = SHAPexplainer(model, tokenizer, words_dict, words_dict_reverse)

train_dt = np.array([predictor.split_string(x) for x in np.array(train_data)], dtype=object)
idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

# print(idx_train_data)
print("Finished preparing shap training data")

explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=100))

print("Finished preparing shap explainer")

texts_ = [predictor.split_string(x) for x in texts][1:4]
print(texts_)
idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

for ii in range(len(idx_texts)):
    t = idx_texts[ii]
    to_use = t.reshape(1, -1)
    f = predictor.predict(to_use)
    pred_f = np.argmax(f[0])

    shap_values = explainer.shap_values(X=to_use, l1_reg="aic", nsamples="auto") #nsamples=64

    # shap.force_plot(explainer.expected_value[m], shap_values[m][0, :len_], texts_[ii])

    visualize_explanations.joint_visualization(texts_[ii], shap_values[pred_f][0, :len(texts_[ii])],
                                               ["go", "java", "javascript", "php", "python", "ruby"][int(pred_f)], f[0][pred_f], ii)
