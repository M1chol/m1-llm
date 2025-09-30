import torch

class CasualAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_lenght, dropout, qkv_bias=False) -> None:
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key= torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value= torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1)
        )
    def forward(self, x):
        """
        Calculates the context vector for a given embedding with mask 
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf # type: ignore
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1], dim=-1)
        context_vector = attn_weights @ values
        return context_vector


if __name__ == "__main__":
    torch.manual_seed(789)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],    # Your (x^1)
        [0.55, 0.87, 0.66],     # journey (x^2)
        [0.57, 0.85, 0.64],     # starts (x^3)
        [0.22, 0.58, 0.33],     # with (x^4)
        [0.77, 0.25, 0.10],     # one (x^5)
        [0.05, 0.80, 0.55]]     # step (x^6)
    )
    d_in = inputs.shape[1]
    d_out = 2
    batch = torch.stack((inputs, inputs), dim=0)
    context_len = batch.shape[1]
    sa = CasualAttention(d_in, d_out, context_len, 0.0)
    context_vectors = sa(batch)
    print(context_vectors)
    print("context_vec.shape:", context_vectors.shape)
    