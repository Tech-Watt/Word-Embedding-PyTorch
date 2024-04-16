from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
import torch

text = 'this is my first word embedding project'

tokenizer = get_tokenizer('basic_english')
tokens = tokenizer(text)
print(tokens)

glove = GloVe(name='6B', dim=100)
indices = [glove.stoi[token] for token in tokens]
print(indices)

embeddings = glove.vectors[torch.tensor(indices)]
print(embeddings)
print(embeddings.shape)