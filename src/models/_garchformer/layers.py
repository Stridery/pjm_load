import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
	"""
	Split (B, L, d_input) into non-overlapping patches of length patch_len,
	then project each flattened patch into a d_model embedding.

	Output shape: (B, N_patches, d_model)
	"""

	def __init__(self, patch_len, d_input, d_model):
		super().__init__()
		self.patch_len = patch_len
		self.d_input = d_input
		self.d_model = d_model
		self.proj = nn.Linear(patch_len * d_input, d_model)

	def forward(self, x):
		"""
		Args:
			x: input sequence, shape (B, L, d_input)

		Returns:
			x_tokenized: shape (B, N_patches, d_model)
		"""
		B, L, d_in = x.shape
		assert d_in == self.d_input
		assert L % self.patch_len == 0, "L must be divisible by patch_len"

		N = L // self.patch_len
		x = x.view(B, N, self.patch_len, d_in)  # (B, N, p, d)
		x = x.reshape(B, N, self.patch_len * d_in)  # (B, N, p*d)
		x = self.proj(x)  # (B, N, d_model)
		return x


class PositionalEncoding(nn.Module):
	"""Learnable positional encoding."""
	def __init__(self, seq_len, d_emb):
		super().__init__()
		self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_emb))

	def forward(self, x):
		# x: (B, L, d)
		# return x + self.pos_emb[:, :x.size(1), :]  # (B, L, d)
		return x + self.pos_emb


class FeedForwardBlock(nn.Module):
	"""Feedforward network with residual connection used in both encoder and decoder."""
	def __init__(self, d_emb, expansion=4, dropout=0.1):
		super().__init__()
		self.fc1 = nn.Linear(d_emb, expansion * d_emb)
		self.activation = nn.GELU()
		self.dropout1 = nn.Dropout(dropout)
		self.fc2 = nn.Linear(expansion * d_emb, d_emb)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x):
		# x: (B, L, d)
		residual = x
		x = self.fc1(x)               # (B, L, d * expansion)
		x = self.activation(x)
		x = self.dropout1(x)
		x = self.fc2(x)               # (B, L, d)
		x = self.dropout2(x)
		return x + residual           # (B, L, d)

class SigmaAttentionBlock(nn.Module):
	"""
	GARCH-style variance attention block.
	Computes time-varying σ_t^2 ∈ ℝ^d from past ε_j^2 ∈ ℝ^d and previous σ_{t-1}^2
	using causal attention, with learnable ω (base), α (ARCH weight), and β (GARCH weight).

	Input:
		x: hidden representation sequence, shape (B, L, d)
		epsilon_sq: residual squared errors, shape (B, L, d)

	Output:
		sigma_sq: estimated conditional variance σ_t^2, shape (B, L, d)
	"""

	def __init__(self, d_model, dropout=0.1):
		super().__init__()
		self.query_proj = nn.Linear(d_model, d_model)
		self.key_proj = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.scale = d_model ** -0.5

		# Learnable GARCH parameters
		self.omega_raw = nn.Parameter(torch.zeros(1, 1, d_model))
		self.alpha_raw = nn.Parameter(torch.zeros(1, 1, d_model))
		self.beta_raw  = nn.Parameter(torch.zeros(1, 1, d_model))

	def forward(self, x, epsilon_sq):
		"""
		Args:
			x: input sequence (B, L, d)
			epsilon_sq: residual squared (B, L, d)

		Returns:
			sigma_sq: (B, L, d)
		"""
		B, L, d = x.shape

		# ---- Debug section ----
		# print("\n[DEBUG SigmaAttentionBlock]")
		# print(f"Input x shape: {x.shape}")  # Expect (B, L, d)
		# print(f"Input epsilon_sq shape: {epsilon_sq.shape}")  # Expect (B, L, d)

		# Project queries and keys
		q = self.query_proj(x)  # (B, L, d)
		k = self.key_proj(x)  # (B, L, d)
		# print(f"q shape: {q.shape}, k shape: {k.shape}")

		# Attention weights
		attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, L, L)
		# print(f"attn_scores shape: {attn_scores.shape}")

		causal_mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0)
		attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
		attn_weights = F.softmax(attn_scores, dim=-1)
		attn_weights = self.dropout(attn_weights)
		# print(f"attn_weights shape: {attn_weights.shape}")

		# Weighted ε² (ARCH term)
		arch_term = torch.einsum('bij,bjd->bid', attn_weights, epsilon_sq)
		# print(f"arch_term shape: {arch_term.shape}")

		# GARCH coefficients
		omega = F.softplus(self.omega_raw)
		alpha = torch.sigmoid(self.alpha_raw)
		beta = torch.sigmoid(self.beta_raw)
		total = alpha + beta
		beta = beta / torch.clamp(total, min=1.0)

		# Recursive computation
		sigma_sq = []
		prev_sigma = torch.zeros(B, d, device=x.device)
		for t in range(L):
			sigma_t = omega.squeeze(0) + alpha.squeeze(0) * arch_term[:, t, :] + beta.squeeze(0) * prev_sigma
			sigma_sq.append(sigma_t)
			prev_sigma = sigma_t

		sigma_sq = torch.stack(sigma_sq, dim=1)  # (B, L, d)
		# print(f"Final sigma_sq shape: {sigma_sq.shape}")
		# print("[DEBUG SigmaAttentionBlock] End.\n")

		return sigma_sq


class ArchQAttentionBlock(nn.Module):
	"""
	One encoder block that computes μ (mean path) with explicit causal self-attention,
	then computes σ² (variance path) using the GARCH-style SigmaAttentionBlock.
	"""

	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		self.head_dim = d_model // num_heads

		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

		self.W_q = nn.Linear(d_model, d_model)
		self.W_k = nn.Linear(d_model, d_model)
		self.W_v = nn.Linear(d_model, d_model)

		self.dropout = nn.Dropout(dropout)
		self.norm1 = nn.LayerNorm(d_model)
		self.ffn = FeedForwardBlock(d_model, expansion=4, dropout=dropout)
		self.norm2 = nn.LayerNorm(d_model)

		self.scale = (self.head_dim) ** -0.5
		self.sigma_attention = SigmaAttentionBlock(d_model, dropout=dropout)

	def forward(self, x):
		"""
		Args:
			x: Input tensor, shape (B, L, d)
		Returns:
			mu: (B, L, d)
			sigma_sq: (B, L, d)
		"""
		B, L, d = x.shape

		# === Step 1: Compute μ using explicit causal multi-head attention ===
		x_norm = self.norm1(x)

		q = self.W_q(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)
		k = self.W_k(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)
		v = self.W_v(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)

		attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, L, L)

		# causal mask: allow attending to past and self only
		causal_mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
		attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

		attn_weights = F.softmax(attn_scores, dim=-1)  # (B, h, L, L)
		attn_weights = self.dropout(attn_weights)

		context = torch.matmul(attn_weights, v)  # (B, h, L, d_h)
		context = context.transpose(1, 2).contiguous().view(B, L, d)  # (B, L, d)

		mu = self.norm2(self.ffn(x_norm + context))  # (B, L, d)

		# === Step 2: Compute synchronous residuals ===
		epsilon = x - mu            # (B, L, d)
		epsilon_sq = epsilon ** 2   # (B, L, d)

		# === Step 3: Compute σ² ===
		sigma_sq = self.sigma_attention(x, epsilon_sq)  # (B, L, d)

		return mu, sigma_sq


class VarianceFiLMFusion(nn.Module):
	"""
	Variance-aware FiLM fusion block.
	Generates channel-wise scaling (γ) and bias (β) from log(σ²),
	then applies them to modulate the mean path μ.

	Computes:
		h = γ ⊙ μ + β

	Input:
		mu: (B, L, d)
		log_sigma: (B, L, d)

	Output:
		h: (B, L, d)
	"""

	def __init__(self, d_model, hidden_dim=None):
		super().__init__()
		if hidden_dim is None:
			hidden_dim = d_model * 2

		# Small MLP to map log(σ²) → (γ, β)
		self.mlp = nn.Sequential(
			nn.Linear(d_model, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, 2 * d_model)  # output concatenates [γ, β]
		)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, mu, log_sigma):
		"""
		Args:
			mu: (B, L, d)
			log_sigma: (B, L, d)

		Returns:
			h: (B, L, d)
		"""
		B, L, d = mu.shape

		# Step 1: Map log_sigma → [γ, β]
		film_params = self.mlp(log_sigma)          # (B, L, 2d)
		gamma, beta = film_params.chunk(2, dim=-1) # (B, L, d), (B, L, d)

		# Step 2: Apply FiLM modulation to μ
		h = gamma * mu + beta                      # (B, L, d)

		# Step 3: Normalize (optional, improves stability)
		h = self.norm(h)

		return h


class ARCHEncoderLayer(nn.Module):
	"""
	One encoder layer of ArchFormer2:
	Computes (μ, σ²) → applies FiLM-based variance modulation → produces next-layer representation.
	"""

	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		self.arch_block = ArchQAttentionBlock(d_model, num_heads, dropout)
		self.film_fusion = VarianceFiLMFusion(d_model)
		self.rescale = nn.Linear(d_model, d_model)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x):
		"""
		Args:
			x: (B, L, d)

		Returns:
			x_next: (B, L, d)
			mu: (B, L, d)
			sigma_sq: (B, L, d)
		"""
		mu, sigma_sq = self.arch_block(x)             # (B, L, d), (B, L, d)
		log_sigma = F.softplus(self.rescale(sigma_sq))# ensure positive + stabilize
		x_mod = self.film_fusion(mu, log_sigma)       # (B, L, d)
		x_next = self.norm(x + x_mod)                 # residual connection
		return x_next, mu, sigma_sq


class ARCHEncoder(nn.Module):
	"""
	Stacked encoder consisting of multiple ARCHEncoderLayers.
	Each layer computes (μ, σ²) and produces a variance-aware representation x_next
	that serves as input for the next layer.
	"""

	def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
		super().__init__()
		self.layers = nn.ModuleList([
			ARCHEncoderLayer(d_model, num_heads, dropout)
			for _ in range(num_layers)
		])
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x):
		"""
		Args:
			x: Input sequence, shape (B, L, d)

		Returns:
			x_final: Final encoded representation, shape (B, L, d)
			mu_list: List of μ from each layer, each of shape (B, L, d)
			sigma_list: List of σ² from each layer, each of shape (B, L, d)
		"""
		mu_list = []
		sigma_list = []

		# Sequentially pass through encoder layers
		for layer in self.layers:
			x, mu, sigma_sq = layer(x)      # (B, L, d)
			mu_list.append(mu)
			sigma_list.append(sigma_sq)

		x_final = self.norm(x)             # Final normalization
		return x_final, mu_list, sigma_list



class MLPConnectionBlock(nn.Module):
	"""
	4-layer MLP block that transforms the encoder output
	to match the decoder input shape.
	"""

	def __init__(self, seq_length_enc, seq_length_dec, d_emb, intermediate_dim=32):
		"""
		Args:
			seq_length_enc (int): Encoder sequence length.
			seq_length_dec (int): Decoder sequence length.
			d_emb (int): Embedding dimension.
			intermediate_dim (int): Hidden intermediate temporal size (M).
		"""
		super().__init__()

		self.fc1 = nn.Linear(d_emb, 1)  # Step 1: Reduce embedding dim (batch, seq_length_enc, 1)
		self.fc2 = nn.Linear(seq_length_enc, intermediate_dim)  # Step 2: Expand temporal axis (batch, M, 1)
		self.activation = nn.GELU()  # Activation function
		self.fc3 = nn.Linear(intermediate_dim, seq_length_dec)  # Step 3: Map to output length (batch, seq_length_dec, 1)
		self.fc4 = nn.Linear(1, d_emb)  # Step 4: Restore embedding dimension (batch, seq_length_dec, d_emb)

	def forward(self, x):
		"""
		Forward pass through the MLP block.
		"""
		# print(f"MLP Block Input Shape: {x.shape}")  # Expecting (batch, 168, 64)
		x = self.fc1(x)  # (batch, 168, 1)
		# print(f"MLP Block Shape after fc1: {x.shape}")
		x = x.permute(0, 2, 1)  # (batch, 1, 168) for correct transformation
		x = self.fc2(x)  # (batch, 1, M)
		# print(f"MLP Block Shape after fc2: {x.shape}")
		x = self.activation(x)  # Activation
		# print(f"MLP Block Shape after activation: {x.shape}")
		x = self.fc3(x)  # (batch, 1, 24)
		# print(f"MLP Block Shape after fc3: {x.shape}")
		x = x.permute(0, 2, 1)  # (batch, 24, 1) back to normal
		x = self.fc4(x)  # (batch, 24, d_emb)
		# print(f"MLP Block Output Shape: {x.shape}")  # Expecting (batch, 24, 64)
		return x




class MuCrossAttentionBlock(nn.Module):
	"""
	Cross-attention block for the mean path in the decoder.
	Computes μ_dec = CrossAttn(query = x_dec, key = μ_enc, value = μ_enc).
	"""
	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		self.head_dim = d_model // num_heads
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

		self.W_q = nn.Linear(d_model, d_model)
		self.W_k = nn.Linear(d_model, d_model)
		self.W_v = nn.Linear(d_model, d_model)

		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(d_model)
		self.scale = self.head_dim ** -0.5

	def forward(self, x_dec, mu_enc):
		"""
		Args:
			x_dec: decoder input tokens, shape (B, L, d)
			mu_enc: encoder μ, shape (B, L, d)

		Returns:
			mu_dec: (B, L, d)
		"""
		B, L, d = x_dec.shape

		x_norm = self.norm(x_dec)

		# Project query, key, value
		Q = self.W_q(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)
		K = self.W_k(mu_enc).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)
		V = self.W_v(mu_enc).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)

		# Compute scaled attention
		attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, h, L, L)
		attn_weights = F.softmax(attn_scores, dim=-1)
		attn_weights = self.dropout(attn_weights)

		context = torch.matmul(attn_weights, V)  # (B, h, L, d_h)
		context = context.transpose(1, 2).contiguous().view(B, L, d)  # (B, L, d)

		return context  # (B, L, d)


class SigmaCrossAttentionBlock(nn.Module):
	"""
	Variance path attention block (decoder side).
	Computes σ_dec² using causal attention over decoder ε²
	and a recursive GARCH-like update with learnable ω, α, β.
	"""
	def __init__(self, d_model, dropout=0.1):
		super().__init__()
		self.query_proj = nn.Linear(d_model, d_model)
		self.key_proj = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.scale = d_model ** -0.5

		# Learnable GARCH coefficients
		self.omega_raw = nn.Parameter(torch.zeros(1, 1, d_model))
		self.alpha_raw = nn.Parameter(torch.zeros(1, 1, d_model))
		self.beta_raw = nn.Parameter(torch.zeros(1, 1, d_model))

	def forward(self, x_dec, epsilon_sq):
		"""
		Args:
			x_dec: decoder hidden tokens (variance-aware representation), shape (B, L, d)
			epsilon_sq: squared residuals, shape (B, L, d)

		Returns:
			sigma_sq: estimated conditional variance, shape (B, L, d)
		"""
		B, L, d = x_dec.shape

		# Project Q, K for attention
		q = self.query_proj(x_dec)  # (B, L, d)
		k = self.key_proj(x_dec)    # (B, L, d)

		# Compute causal attention scores
		attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, L, L)
		causal_mask = torch.tril(torch.ones(L, L, device=x_dec.device)).unsqueeze(0)  # (1, L, L)
		attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

		attn_weights = F.softmax(attn_scores, dim=-1)  # (B, L, L)
		attn_weights = self.dropout(attn_weights)

		# Attention-weighted ARCH term
		arch_term = torch.einsum('bij,bjd->bid', attn_weights, epsilon_sq)  # (B, L, d)

		# GARCH coefficients (bounded)
		omega = F.softplus(self.omega_raw)       # base term, ensure positive
		alpha = torch.sigmoid(self.alpha_raw)    # (0,1)
		beta = torch.sigmoid(self.beta_raw)      # (0,1)
		total = alpha + beta
		beta = beta / torch.clamp(total, min=1.0)

		# Recursive GARCH update
		sigma_sq = []
		prev_sigma = torch.zeros(B, d, device=x_dec.device)
		for t in range(L):
			sigma_t = omega.squeeze(0) + alpha.squeeze(0) * arch_term[:, t, :] + beta.squeeze(0) * prev_sigma  # (B, d)
			sigma_sq.append(sigma_t)
			prev_sigma = sigma_t

		sigma_sq = torch.stack(sigma_sq, dim=1)  # (B, L, d)
		return sigma_sq


class ARCHDecoderLayer(nn.Module):
	"""
	One decoder layer of ArchFormer2:
	- Mean path: cross-attention over encoder μ
	- Variance path: causal attention over decoder ε^2 with GARCH-like recursion
	- Fusion: FiLM modulation (log σ^2 -> (γ, β), then h = γ ⊙ μ + β), residual to x_dec
	"""

	def __init__(self, d_model, num_heads, dropout=0.1):
		super().__init__()
		self.mu_attn = MuCrossAttentionBlock(d_model, num_heads, dropout)
		self.sigma_attention = SigmaCrossAttentionBlock(d_model, dropout)  # decoder-side σ^2
		self.film_fusion = VarianceFiLMFusion(d_model)
		self.rescale = nn.Linear(d_model, d_model)  # map σ^2 to log-σ for FiLM stability
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x_t, mu_enc, sigma_enc=None):
		"""
		Args:
			x_t: decoder input tokens, shape (B, L, d)
			mu_enc: encoder mean path, shape (B, L, d)
			sigma_enc: encoder variance path (kept for interface compatibility), shape (B, L, d) or None

		Returns:
			x_next: (B, L, d) — variance-modulated hidden tokens for the next decoder layer
			mu:     (B, L, d) — decoder mean path
			sigma_sq: (B, L, d) — decoder variance path
		"""
		B, L, d = x_t.shape

		# === Step 1: mean path via cross-attention over encoder μ ===
		mu = self.mu_attn(x_t, mu_enc)                    # (B, L, d)

		# === Step 2: synchronous residuals (decoder hidden space) ===
		epsilon = x_t - mu                                # (B, L, d)
		epsilon_sq = epsilon ** 2                         # (B, L, d)

		# === Step 3: variance path via causal attention over ε^2 (decoder-side) ===
		sigma_sq = self.sigma_attention(x_t, epsilon_sq)  # (B, L, d)

		# === Step 4: FiLM fusion from log-σ ===
		log_sigma = F.softplus(self.rescale(sigma_sq))    # stabilize and ensure positivity
		x_mod = self.film_fusion(mu, log_sigma)           # (B, L, d)

		# Residual to x_t keeps the decoder dynamics stable
		x_next = self.norm(x_t + x_mod)                   # (B, L, d)

		return x_next, mu, sigma_sq

class ARCHDecoder(nn.Module):
	"""
	Stacked decoder of ArchFormer2.
	Consumes the bridge output x (B, L, d) and encoder paths (μ_enc, σ_enc),
	and produces a variance-aware decoded representation plus per-layer (μ, σ^2).
	Note: σ_enc is reserved for potential prior injection; not used in Variant-A layer.
	"""

	def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
		super().__init__()
		self.layers = nn.ModuleList([
			ARCHDecoderLayer(d_model, num_heads, dropout)
			for _ in range(num_layers)
		])
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x, mu_enc, sigma_enc=None):
		"""
		Args:
			x: decoder input after bridge, shape (B, L, d)
			mu_enc: encoder μ (B, L, d)
			sigma_enc: encoder σ^2 (B, L, d) or None (kept for API compatibility)

		Returns:
			x_final: (B, L, d)
			mu_list: list of length num_layers, each (B, L, d)
			sigma_list: list of length num_layers, each (B, L, d)
		"""
		mu_list = []
		sigma_list = []

		for layer in self.layers:
			x, mu, sigma_sq = layer(x, mu_enc, sigma_enc)
			mu_list.append(mu)
			sigma_list.append(sigma_sq)

		x_final = self.norm(x)
		return x_final, mu_list, sigma_list

class InversePatchTokenizer(nn.Module):
	"""
	Reconstruct original sequence from patch tokens.
	Input shape: (B, N_patches, d_model)
	Output shape: (B, L, d_input), where L = N_patches * patch_len
	"""

	def __init__(self, patch_len, d_input, d_model):
		super().__init__()
		self.patch_len = patch_len
		self.d_input = d_input
		self.d_model = d_model
		self.reproj = nn.Linear(d_model, patch_len * d_input)

	def forward(self, x):
		"""
		Args:
			x: (B, N, d_model)

		Returns:
			x_reconstructed: (B, N * patch_len, d_input)
		"""
		B, N, d_model = x.shape
		assert d_model == self.d_model, "Input embedding dim mismatch"

		x = self.reproj(x)  # (B, N, patch_len * d_input)
		x = x.view(B, N * self.patch_len, self.d_input)  # (B, L, d_input)
		return x


class FinalProjectionHead(nn.Module):
	"""
	Projects (B, L, d_input) → (B, out_len) for final point prediction.
	"""
	def __init__(self, d_input, out_len):
		super().__init__()
		self.project = nn.Linear(d_input, 1)
		self.out_len = out_len

	def forward(self, x):
		# x: (B, L, d_input)
		x = self.project(x)             # (B, L, 1)
		x = x[:, -self.out_len:, :]     # (B, out_len, 1)
		return x.squeeze(-1)            # (B, out_len)



if __name__ == "__main__":
	B, L, d_input = 32, 168, 10
	patch_len = 1  # no tokenization for now
	d_model = 64
	seq_len_enc = L // patch_len
	seq_len_dec = seq_len_enc      # assume equal lengths
	intermediate_dim = 32

	# === Step 1: Raw input ===
	x_raw = torch.randn(B, L, d_input)  # (32, 168, 10)

	# === Step 2: Patch Tokenization ===
	tokenizer = PatchTokenizer(patch_len=patch_len, d_input=d_input, d_model=d_model)
	x_tokenized = tokenizer(x_raw)
	print("Tokenized shape:", x_tokenized.shape)

	# === Step 3: Encoder ===
	encoder = ARCHEncoder(d_model=d_model, num_heads=4, num_layers=1, dropout=0.1)
	x_encoded, mu_list, sigma_list = encoder(x_tokenized)
	print("Encoder x output shape:", x_encoded.shape)
	print("Encoder mu output shape:", mu_list[-1].shape)
	print("Encoder sigma output shape:", sigma_list[-1].shape)

	# === Step 4: Bridge to Decoder ===
	bridge = MLPConnectionBlock(
		seq_length_enc=seq_len_enc,
		seq_length_dec=seq_len_dec,
		d_emb=d_model,
		intermediate_dim=intermediate_dim
	)
	x_dec_init = bridge(x_encoded)
	print("Decoder init shape:", x_dec_init.shape)

	# === Step 5: Decoder ===
	decoder = ARCHDecoder(d_model=d_model, num_heads=4, num_layers=1, dropout=0.1)
	x_dec_final, mu_dec_list, sigma_dec_list = decoder(x_dec_init, mu_list[-1], sigma_list[-1])
	print("\n--- Decoder Test ---")
	print("Decoder final output shape:", x_dec_final.shape)
	print("Decoder mu list length:", len(mu_dec_list))
	print("Decoder sigma list length:", len(sigma_dec_list))
	print("Decoder mu[0] shape:", mu_dec_list[-1].shape)
	print("Decoder sigma[0] shape:", sigma_dec_list[-1].shape)

	# === Step 6: Inverse Patch Tokenization ===
	reconstructor = InversePatchTokenizer(patch_len=patch_len, d_input=d_input, d_model=d_model)
	x_rec = reconstructor(x_dec_final)
	print("After InversePatchTokenizer x_rec shape:", x_rec.shape)

	# === Step 7: Final Projection Head ===
	out_len = 24
	head = FinalProjectionHead(d_input=d_input, out_len=out_len)
	y_hat = head(x_rec)
	print("Final output shape (y_hat):", y_hat.shape)

	# === Step 8: Parameter Counting ===
	def count_parameters(model):
		return sum(p.numel() for p in model.parameters() if p.requires_grad)

	total_params = (
		count_parameters(tokenizer)
		+ count_parameters(encoder)
		+ count_parameters(bridge)
		+ count_parameters(decoder)
		+ count_parameters(reconstructor)
		+ count_parameters(head)
	)

	print(f"\nTotal trainable parameters in this configuration: {total_params:,}")
	print(f" - Tokenizer: {count_parameters(tokenizer):,}")
	print(f" - Encoder: {count_parameters(encoder):,}")
	print(f" - Bridge: {count_parameters(bridge):,}")
	print(f" - Decoder: {count_parameters(decoder):,}")
	print(f" - Reconstructor: {count_parameters(reconstructor):,}")
	print(f" - Head: {count_parameters(head):,}")
