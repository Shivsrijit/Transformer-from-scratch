import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int , vocab_size:int):
        super().__init__()
        self.d_model = d_model # d_model is the size of embedding vector of each word
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  #torchs inbuilt embedder  

    def forward(self, x): 
        #in embedding layer the weights of embedding layer are multiplied by sqrt of dmodel 
        return self.embedding(x) * math.sqrt(self.d_model)
    

# Now we would calculate positional encoding using the formula given in paper for odd and and even position 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_length : int, dropout : float) -> None :   # here dropout is to make our model less overfit 
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        #now we would create a matrix of size (seq_length, d_model) to store embeddings 
        #it means that we would have a vector of size d_model for each sentence stored in that matrix 
        pe = torch.zeroes(seq_length, d_model)

        #creating a vector of size seq_length which would represent the position of the word inside the sentence
        position = torch.arange(0 , seq_length, dtype=torch.float).unsqueeze(1)
 
