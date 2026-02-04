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
        # Each row is a positional embedding vector (size d_model) for one token position
        pe = torch.zeros(seq_length, d_model)

        #creating a vector of size seq_length which would represent the position of the word inside the sentence
        position = torch.arange(0 , seq_length, dtype=torch.float).unsqueeze(1) #(seq_len, 1)
        # division term 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) /d_model))  #gives the denominator term 

        #apply sin and cos 
        pe[:, 0::2] = torch.sin(position * div_term) # for all the words in the sentence , but only even positions 
        pe[:, 1::2] = torch.cos(position * div_term) # for all odd positions

        #now we would add batch dimension to this tensor 
        # so that we can apply this to many sentences ..currently it is for just one sentence 
        pe = pe.unsqueeze(0) #(1 , seq_len , d_model)

        #we will register this encoding in the buffer of this model 
        self.register_buffer('pe', pe)  # we want this tensor to be saved when we save the module 

    def forward(self ,x ): 
        #x.shape = (batch_size, seq_length, d_model)
        #self.pe.shape = (1, max_supported_seq_length, d_model)
        x = x + (self.pe[: , :x.shape[1], :]).requires_grad_(False)
        # : → keep batch dimension (1)
        # :x.shape[1] → take only the first seq_length positions
        # : → keep all embedding dimensions
        # requires_grad(False) means we dont wanna learn this , it is already there ...do NOT compute gradients for this tensor
        #requires grad is redundant cuz it is already applied 
        return self.dropout(x)
    
class LayerNormalisation(nn.Module): 
        def __init__(self, eps : float = 10**-6) -> None : #eps is epsilon
             super().__init__()
             self.eps = eps
             self.alpha = nn.Parameter(torch.ones(1)) #muliplied
             self.bias = nn.Parameter(torch.zeros(1)) # Added
            
        def forward(self , x): 
             mean = x.mean(dim = -1, keepdim=True)
             std = x.std(dim = -1, keepdim=True)
             return self.alpha * (x - mean) / (std + self.eps) + self.bias
        
            #dim = -1 is used because the last dimension represents the feature/embedding dimension, 
            #and normalization is performed independently for each token across its features.

       
class FeedForwardBlock(nn.Module):
     
     def __init__(self, d_model : int, d_ff: int, dropout: float) -> None: 
          super().__init__()
          self.linear1 = nn.Linear(d_model, d_ff) #W1 and B1
          self.dropout = nn.Dropout(dropout)
          self.linear2 = nn.Linear(d_ff, d_model) #W2 and B2

     def forward(self, x):
          #(batch , seq_lengthm d_model) --> (Batch, Seq_length, d_ff)..using linear 1 --> (batch , seq_lengthm d_model)..using d_model
          return self.linear2(self.dropout(torch.relu(self.linear1(x))))
          #check notebook for formula 

class MultiHeadAttentionBlock(nn.Module) : 
     
     def __init__(self, d_model: int , h : int, dropout : float):
          super().__init__()
          self.d_model = d_model
          self.h = h # no of heads 
          self.d = nn.Dropout(dropout)
          assert d_model % h == 0, "d_model is not divisible by h "
          self.d_k = d_model // h 
          self.w_q =  nn.Linear(d_model, d_model)  
          self.w_k =  nn.Linear(d_mod el, d_model)  
          self.w_v =  nn.Linear(d_model, d_model)       

          self.w_o = nn.Linear(d_model , d_model)   

     def forward(self, q, k , v, mask) : 
          query = self.w_q(q) # (batch , seq_length , d_model) --> (batch, seq_len, d_model)
          key = self.w_k(k)  # (batch , seq_length , d_model) --> (batch, seq_len, d_model)
          value = self.w_v(v) # (batch , seq_length , d_model) --> (batch, seq_len, d_model)
     
          # (batch , seq_lngth, d_model) --> (batch , seq_length, h , d_k) --> (batch , h , seq_length, d_k)
          query = query.view(query.shape[0], query.shape[1] , self.h, self.d_k).transpose(1 ,2 )
         
        




