def calculate_params(vocab_size, d_model, n_layers, n_heads, d_ff, tie_embeddings=True):
    embedding_params = vocab_size * d_model
    
    # Attention: Wq, Wk, Wv, Wo
    attn_params_per_layer = 4 * (d_model * d_model)
    
    # MLP (SwiGLU): W1, W2, W3
    mlp_params_per_layer = 3 * (d_model * d_ff)
    
    # Norms: attention norm + mlp norm
    norm_params_per_layer = 2 * d_model
    
    total_per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    transformer_params = n_layers * total_per_layer
    
    # Final norm
    final_norm = d_model
    
    lm_head = 0 if tie_embeddings else vocab_size * d_model
    
    total = embedding_params + transformer_params + final_norm + lm_head
    
    print(f"Vocab: {vocab_size}, d_model: {d_model}, layers: {n_layers}, heads: {n_heads}, d_ff: {d_ff}")
    print(f"Embedding: {embedding_params:,} ({embedding_params/total*100:.1f}%)")
    print(f"Transformer body: {transformer_params:,}")
    print(f"Total: {total:,} ({total/1e6:.2f}M)")
    print("-" * 40)

# V1 Baseline
calculate_params(2048, 256, 6, 8, 1024)

# Candidate A (Target ~15M)
calculate_params(8192, 384, 6, 6, 1024)
calculate_params(8192, 384, 6, 12, 1024)

# Candidate B (Target ~30M)
calculate_params(8192, 512, 6, 8, 2048)
calculate_params(8192, 512, 8, 8, 2048)
