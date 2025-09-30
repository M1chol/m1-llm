import torch

inputs = torch.tensor(
[[0.43, 0.15, 0.89],    # Your (x^1)
[0.55, 0.87, 0.66],     # journey (x^2)
[0.57, 0.85, 0.64],     # starts (x^3)
[0.22, 0.58, 0.33],     # with (x^4)
[0.77, 0.25, 0.10],     # one (x^5)
[0.05, 0.80, 0.55]]     # step (x^6)
)

## Obliczenie pojedynczego wektora
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print("Tensor tokena:", query)
print("Obliczony tensor uwagi dla tokeana:", attn_scores_2)

attn_waights_2 = torch.softmax(attn_scores_2, dim=0)
print("Wagi uwagi po normalizacji:", attn_waights_2)
print("Suma wag:", attn_waights_2.sum())

context_vector2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector2 += attn_waights_2[i] * x_i
print("Wektor kontekstu:", context_vector2)

## Obliczanie wszystkich wektorów kontekstu na raz

# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

attn_scores = inputs @ inputs.T
print("Wszystkie wektory uwagi:\n", attn_scores)

attn_waights = torch.softmax(attn_scores, dim=-1)
print("Wszystkie wagi po normalizacji:\n", attn_waights)
print("Sumy wierszy:", attn_waights.sum(dim=-1))

context_vectors = attn_waights @ inputs
print("Wektory kontekstu:\n", context_vectors)