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
          self.dropout = nn.Dropout(dropout)
          assert d_model % h == 0, "d_model is not divisible by h "
          self.d_k = d_model // h 
          self.w_q =  nn.Linear(d_model, d_model)  
          self.w_k =  nn.Linear(d_model, d_model)  
          self.w_v =  nn.Linear(d_model, d_model)       

          self.w_o = nn.Linear(d_model , d_model)   
     

     @staticmethod
     def attention(query, key , value , mask , dropout : nn.Dropout):
          d_k = query[-1]  #d_k is the last dim in query/key/value

          #(batch , h , seq_len , d_k ) * (batch , h , d_k , seq_len ) --> (batch, heads, seq_len, seq_len)
          attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
          
          if mask is not None: 
               attention_scores.masked_fill(mask ==0 , -1e9 ) #replace all the values where (mask==0) is true with the value -1e9
          
          # tensor shape is : (batch, heads, seq_len, seq_len)
          # So each slice scores[b, h] is a seq_len × seq_len matrix: Rows → queries  and Columns → keys
          #For each query token, we want a probability distribution over all keys it can attend to.
          #Each row should sum to 1.....That means we apply softmax across the columns (keys).
          # thats why we choose dim = - 1
          attention_scores = attention_scores.softmax(dim = -1)

          if dropout is not None : 
               attention_scores = dropout(attention_scores)

          return (attention_scores @ value) , attention_scores  # we will use attention scores for visualization 
          #gives (batch , h , seq_len , d_k)

          

     def forward(self, q, k , v, mask) : 
          query = self.w_q(q) # (batch , seq_length , d_model) --> (batch, seq_len, d_model)
          key = self.w_k(k)  # (batch , seq_length , d_model) --> (batch, seq_len, d_model)
          value = self.w_v(v) # (batch , seq_length , d_model) --> (batch, seq_len, d_model)
     
          # (batch , seq_lngth, d_model) --> (batch , seq_length, h , d_k) --> (batch , h , seq_length, d_k)
          # we are swapping seq_length and h cuz PyTorch does batched matrix multiplication on the last two dimensions.
          # Because the next operation is: Q @ Kᵀ
          #We want each head to compute: (seq_len, d_k) @ (d_k, seq_len)   ........So we rearrange: (batch, heads, seq_len, d_k)
          query = query.view(query.shape[0], query.shape[1] , self.h, self.d_k).transpose(1 ,2 )
          key = key.view(key.shape[0], key.shape[1], self.h , self.d_k).transpose(1,2 )
          value = value.view(value.shape[0], value.shape[1], self.h , self.d_k).transpose(1,2 )

          x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value , mask, self.dropout  )

          # (batch , h , seq_len , d_k) --> (batch , seq_len , h , d_k) --> (batch , seq_len , s_model)
          #  basically undoing 
          #-1 → let PyTorch automatically compute that dimension
          # transpose() → breaks memory contiguity and ...view() → requires contiguous memory
          # transpose() changes the dimension order without rearranging memory, making the tensor non-contiguous. Since view() can only reshape tensors with contiguous memory, we call .contiguous() to rearrange the data first. Earlier we didn’t need it because tensors from nn.Linear were already contiguous.
          x = x.transpose(1,2).contiguos().view(x.shape[0] , -1 , self.h * self.d_k)

          # (batch , seq_len , d_model) --> (batch , seq_len , d_model )
          return self.w_o(x)

class ResidualConnection(nn.Module):
     
     def __init__(self, dropout: float) -> None : 
          super().__init__()
          self.dropout = nn.Dropout(dropout)
          self.norm = LayerNormalisation()
     
     #taking x and combinning it with the output of next layer ...residual connections 
     def forward(self , x, sublayer) : 
          return x + self.dropout(sublayer(self.norm(x)))
          #here sublayer is the previous layer


class EncoderBlock(nn.Module):

     def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float) -> None :
          super().__init__()
          self.self_attention_block = self_attention_block
          self.feed_forward_block = feed_forward_block
          # x → Residual(Attention)....and x → Residual(FeedForward)
          self.residual_connections =  nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

     #we need mask bcz we need to hide the interaction of paddin words with other words
     def forward(self, x, src_mask):
          x = self.residual_connections[0](x , lambda x : self.self_attention_block(x,x,x, src_mask))
          x = self.residual_connections[1](x, self.feed_forward_block)

          return x 

class Encoder(nn.Module):

     def __init__(self, layers : nn.ModuleList) -> None:
          super().__init__()
          self.layers = layers
          self.norm = LayerNormalisation()

     def forward(self, x, mask):
          for layer in self.layers:
               x = layer(x , mask )
          return self.norm(x)

