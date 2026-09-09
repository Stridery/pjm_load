import torch
import torch.nn as nn
import math
import sys
import os

from src.models._garchformer.layers import (
	PatchTokenizer, PositionalEncoding, ARCHEncoder, MLPConnectionBlock,
	ARCHDecoder, InversePatchTokenizer, FinalProjectionHead
)


class ARCHFormer3(nn.Module):
	"""
	ArchFormer3: Variance-aware Transformer with FiLM fusion and synchronized residuals.
	Integrates PatchTokenizer, PositionalEncoding, ARCHEncoder (causal mean + variance path),
	MLP Connection Bridge, ARCHDecoder, InversePatchTokenizer, and FinalProjectionHead.
	"""

	def __init__(self, seq_length_enc, seq_length_dec, num_features, patch_len, d_model,
				 num_heads, num_encoder_layers, num_decoder_layers, intermediate_dim,
				 out_len, dropout=0.1):
		"""
		Args:
			seq_length_enc (int): Encoder input sequence length (e.g., 168)
			seq_length_dec (int): Decoder sequence length (e.g., 24)
			num_features (int): Number of input features (d_input)
			patch_len (int): Patch length for tokenization
			d_model (int): Embedding dimension
			num_heads (int): Number of attention heads
			num_encoder_layers (int): Number of encoder layers
			num_decoder_layers (int): Number of decoder layers
			intermediate_dim (int): Hidden dimension of the MLP bridge
			out_len (int): Output prediction horizon
			dropout (float): Dropout probability
		"""
		super().__init__()
		self.seq_len_enc = seq_length_enc
		self.seq_len_dec = seq_length_dec
		self.patch_len = patch_len

		# === Core components ===
		self.tokenizer = PatchTokenizer(patch_len, num_features, d_model)
		self.pos_encoder = PositionalEncoding(seq_length_enc // patch_len, d_model)
		self.encoder = ARCHEncoder(d_model, num_heads, num_encoder_layers, dropout)
		self.connection_block = MLPConnectionBlock(
			seq_length_enc // patch_len,
			seq_length_dec // patch_len,
			d_model,
			intermediate_dim
		)
		self.decoder = ARCHDecoder(d_model, num_heads, num_decoder_layers, dropout)
		self.inverse_tokenizer = InversePatchTokenizer(patch_len, num_features, d_model)
		self.head = FinalProjectionHead(num_features, out_len)

	def forward(self, x):
		"""
		Forward pass of ArchFormer3.

		Args:
			x: Input tensor of shape (B, seq_length_enc, num_features)

		Returns:
			y_hat: Final point prediction of shape (B, out_len)
			sigma_sq_out: Variance prediction aligned with y_hat, shape (B, out_len)
		"""
		# === Step 1: Tokenization ===
		x_patch = self.tokenizer(x)  # (B, N_patch, d_model)

		# === Step 2: Positional encoding ===
		x_patch = self.pos_encoder(x_patch)

		# === Step 3: Variance-aware encoder ===
		x_encoded, mu_list, sigma_list = self.encoder(x_patch)

		# === Step 4: Connection block (bridge) ===
		x_dec_input = self.connection_block(x_encoded)

		# === Step 5: Variance-aware decoder ===
		x_dec_output, mu_dec_list, sigma_dec_list = self.decoder(
			x_dec_input, mu_list[-1], sigma_list[-1]
		)

		# === Step 6: Inverse patch tokenization ===
		x_reconstructed = self.inverse_tokenizer(x_dec_output)  # (B, L, d_input)
		y_hat = self.head(x_reconstructed)  # (B, out_len)

		# === Step 7: Variance alignment head ===
		# Take the last decoder σ² (B, N_dec, d_model)
		sigma_last = sigma_dec_list[-1]  # (B, N_dec, d_model)
		# Map it through the same inverse tokenizer and projection head to align shapes
		sigma_rec = self.inverse_tokenizer(sigma_last)  # (B, L, d_input)
		sigma_sq_out = self.head(sigma_rec)  # (B, out_len)
		sigma_sq_out = torch.abs(sigma_sq_out) + 1e-6  # ensure positive variance

		return y_hat, sigma_sq_out


# ---------------------------- Testing Script ----------------------------
if __name__ == "__main__":
	batch_size = 32
	seq_length_enc = 168
	seq_length_dec = 168  # can match or differ, depending on your dataset
	num_features = 10
	patch_len = 1  # tokenization granularity
	d_model = 64
	num_heads = 4
	num_encoder_layers = 2
	num_decoder_layers = 2
	intermediate_dim = 32
	out_len = 24
	dropout = 0.1

	# === Synthetic input ===
	x_input = torch.randn(batch_size, seq_length_enc, num_features)

	# === Model initialization ===
	model = ARCHFormer3(
		seq_length_enc=seq_length_enc,
		seq_length_dec=seq_length_dec,
		num_features=num_features,
		patch_len=patch_len,
		d_model=d_model,
		num_heads=num_heads,
		num_encoder_layers=num_encoder_layers,
		num_decoder_layers=num_decoder_layers,
		intermediate_dim=intermediate_dim,
		out_len=out_len,
		dropout=dropout
	)

	# === Forward ===
	y_output, sigma_sq = model(x_input)

	# === Print diagnostic info ===
	print("\n--- ArchFormer3 Test Run ---")
	print(f"Input shape:  {x_input.shape}")    # (B, 168, 10)
	print(f"Output shape: {y_output.shape}")   # (B, 24)
	print(f"sigma_sq shape: {sigma_sq.shape}")
	print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
