"""
model.py — Seq2Seq with Bahdanau Attention (PyTorch).

Architecture
  Encoder : Bidirectional LSTM
  Decoder : LSTM with additive (Bahdanau-style) attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD_ID, SOS_ID, EOS_ID, UNK_ID


# ==============================================================
# Helpers
# ==============================================================

def length_mask(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(batch, max_len) boolean mask — True for valid positions."""
    max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    return arange < lengths.unsqueeze(1)  # (batch, max_len)


# ==============================================================
# Attention Decoder
# ==============================================================

class AttnDecoder(nn.Module):
    """LSTM decoder with Bahdanau-style additive attention."""

    def __init__(self, embedding: nn.Embedding, hidden_size: int,
                 output_size: int, enc_out_dim: int,
                 n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = embedding
        emb_dim = embedding.embedding_dim

        self.decoder_cell = nn.LSTMCell(emb_dim, hidden_size)
        self.attn_proj = nn.Linear(enc_out_dim, hidden_size, bias=False)
        self.concat = nn.Linear(enc_out_dim + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    # ── Full sequence forward (teacher-forcing) ───────────────
    def forward(self, target, init_states, enc_outs, attn_mask):
        """
        Args
            target      (batch, tgt_len)           decoder input IDs
            init_states (h0, c0) each (batch, H)
            enc_outs    (src_len, batch, enc_dim)
            attn_mask   (batch, src_len) bool
        Returns
            logits      (batch, tgt_len, vocab)
        """
        h, c = init_states
        logits = []
        for t in range(target.size(1)):
            inp_t = target[:, t : t + 1]  # (batch, 1)
            logit, (h, c) = self.step(inp_t, (h, c), enc_outs, attn_mask)
            logits.append(logit)
        return torch.stack(logits, dim=1)

    # ── Single decoding step ──────────────────────────────────
    def step(self, inp, hidden, enc_outs, attn_mask):
        h_prev, c_prev = hidden
        emb = self.dropout(self.embedding(inp).squeeze(1))  # (batch, emb)
        h_t, c_t = self.decoder_cell(emb, (h_prev, c_prev))

        # Attention
        attn_weights = self._attention(h_t, enc_outs, attn_mask)  # (B,1,S)
        context = attn_weights.bmm(enc_outs.transpose(0, 1)).squeeze(1)  # (B, enc_dim)

        concat_out = torch.tanh(self.concat(torch.cat([context, h_t], dim=1)))
        logit = self.out(concat_out)  # raw logits — no log_softmax here
        return logit, (h_t, c_t)

    def _attention(self, dec_h, enc_outs, mask):
        """Additive attention scores → softmax weights."""
        query = dec_h.unsqueeze(0)                          # (1, B, H)
        keys = self.attn_proj(enc_outs)                     # (S, B, H)
        scores = (query * keys).sum(dim=2).t()              # (B, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=1).unsqueeze(1)        # (B, 1, S)


# ==============================================================
# Seq2Seq wrapper
# ==============================================================

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 emb_dim: int = 256, hidden_size: int = 512,
                 n_layers: int = 1, dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1
        self.enc_out_dim = hidden_size * num_dirs

        # Shared embedding — SentencePiece gives us a single vocab for
        # both source and target, so we share the embedding matrix.
        self.shared_embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=PAD_ID)

        # Encoder
        self.encoder = nn.LSTM(
            emb_dim, hidden_size, n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=False,
        )

        # Project encoder final state → decoder initial state
        self._dec_h = nn.Linear(self.enc_out_dim, hidden_size, bias=False)
        self._dec_c = nn.Linear(self.enc_out_dim, hidden_size, bias=False)

        # Decoder (uses same shared embedding)
        self.decoder = AttnDecoder(
            self.shared_embedding, hidden_size, tgt_vocab_size,
            self.enc_out_dim, n_layers, dropout,
        )

    # ── Encode ────────────────────────────────────────────────
    def encode(self, src, src_lengths):
        """
        Args
            src          (batch, src_len)
            src_lengths  (batch,)
        Returns
            enc_outs     (src_len, batch, enc_dim)
            (dec_h, dec_c) each (batch, hidden)
        """
        emb = self.shared_embedding(src).transpose(0, 1)  # (S, B, E)

        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lengths.cpu(), enforce_sorted=False
        )
        enc_out, (h, c) = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out)  # (S, B, enc_dim)

        if self.bidirectional:
            # (2*layers, B, H) → (layers, B, 2H)
            h = torch.cat(h.chunk(2, dim=0), dim=2)
            c = torch.cat(c.chunk(2, dim=0), dim=2)

        dec_h = torch.tanh(self._dec_h(h.squeeze(0)))  # (B, H)
        dec_c = torch.tanh(self._dec_c(c.squeeze(0)))
        return enc_out, (dec_h, dec_c)

    # ── Forward (teacher forcing) ─────────────────────────────
    def forward(self, src, src_lengths, tgt_in):
        enc_outs, init_states = self.encode(src, src_lengths)
        attn_mask = length_mask(src_lengths, src.device)
        logits = self.decoder(tgt_in, init_states, enc_outs, attn_mask)
        return logits  # (B, T, V)

    # ── Greedy decode (inference) ─────────────────────────────
    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, max_len: int = 100):
        """Greedy auto-regressive decoding for a batch."""
        self.eval()
        batch_size = src.size(0)
        enc_outs, (h, c) = self.encode(src, src_lengths)
        attn_mask = length_mask(src_lengths, src.device)

        inp = torch.full((batch_size, 1), SOS_ID, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        outputs = []

        for _ in range(max_len):
            logit, (h, c) = self.decoder.step(inp, (h, c), enc_outs, attn_mask)
            next_tok = logit.argmax(dim=-1)  # (B,)
            outputs.append(next_tok)
            finished |= (next_tok == EOS_ID)
            if finished.all():
                break
            inp = next_tok.unsqueeze(1)

        return torch.stack(outputs, dim=1)  # (B, decoded_len)

    # ── Beam search (single sentence) ─────────────────────────
    @torch.no_grad()
    def beam_decode(self, src_ids: list[int], device: torch.device,
                    beam_size: int = 5, max_len: int = 100):
        """
        Args
            src_ids:   list of integer token IDs for ONE source sentence
            beam_size: number of beams
        Returns
            List of (token_id_list, score) sorted best-first.
        """
        self.eval()
        src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long, device=device)

        enc_outs, (h, c) = self.encode(src_t, src_len)
        attn_mask = length_mask(src_len, device)

        # Expand for beam
        h = h.expand(beam_size, -1).contiguous()
        c = c.expand(beam_size, -1).contiguous()
        enc_outs_exp = enc_outs.expand(-1, beam_size, -1).contiguous()
        attn_mask_exp = attn_mask.expand(beam_size, -1).contiguous()

        vocab_size = self.decoder.out.out_features

        top_k_scores = torch.zeros(beam_size, device=device)
        top_k_seqs = torch.full((beam_size, 1), SOS_ID, dtype=torch.long, device=device)
        prev_words = top_k_seqs

        completed_seqs: list[tuple[list[int], float]] = []

        for step in range(1, max_len + 1):
            logit, (h, c) = self.decoder.step(
                prev_words, (h, c), enc_outs_exp, attn_mask_exp
            )
            log_probs = F.log_softmax(logit, dim=1)
            log_probs = top_k_scores.unsqueeze(1) + log_probs  # (beam, V)

            if step == 1:
                k = min(beam_size, vocab_size)
                top_k_scores, top_k_ids = log_probs[0].topk(k)
            else:
                k = min(beam_size, log_probs.numel())
                top_k_scores, top_k_ids = log_probs.view(-1).topk(k)

            beam_idx = top_k_ids // vocab_size
            word_idx = top_k_ids % vocab_size
            top_k_seqs = torch.cat([top_k_seqs[beam_idx], word_idx.unsqueeze(1)], dim=1)

            # Separate complete / incomplete
            complete = (word_idx == EOS_ID).nonzero(as_tuple=True)[0].tolist()
            incomplete = [i for i in range(len(word_idx)) if word_idx[i].item() != EOS_ID]

            for ci in complete:
                completed_seqs.append(
                    (top_k_seqs[ci].tolist(), top_k_scores[ci].item())
                )

            if len(completed_seqs) >= beam_size * 2 or not incomplete:
                break

            top_k_seqs = top_k_seqs[incomplete]
            top_k_scores = top_k_scores[incomplete]
            h = h[beam_idx[incomplete]]
            c = c[beam_idx[incomplete]]
            prev_words = top_k_seqs[:, -1:]

            # Re-expand enc_outs / mask to new beam width
            cur_beam = len(incomplete)
            enc_outs_exp = enc_outs.expand(-1, cur_beam, -1).contiguous()
            attn_mask_exp = attn_mask.expand(cur_beam, -1).contiguous()

        # If nothing completed, take current partial beams
        if not completed_seqs:
            for i in range(top_k_seqs.size(0)):
                completed_seqs.append(
                    (top_k_seqs[i].tolist(), top_k_scores[i].item())
                )

        completed_seqs.sort(key=lambda x: x[1], reverse=True)
        return completed_seqs
