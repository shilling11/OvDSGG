"""
CoOp-style text prompt encoder for open-vocabulary predicate classification.

Learns n_ctx context vectors prepended to predicate class token embeddings,
producing dynamic [N_classes, 512] embeddings via the frozen CLIP text encoder.
The only trainable parameter is self.ctx.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class TextPromptEncoder(nn.Module):
    def __init__(self, clip_model, classnames: list, n_ctx: int, ctx_init: str = ""):
        """
        Args:
            clip_model: loaded OpenAI CLIP model (from clip.load)
            classnames:  list of class name strings; index 0 must be '__background__'
            n_ctx:       number of learnable context token vectors
            ctx_init:    optional string to initialise context from (e.g. 'a video of')
        """
        super().__init__()

        ctx_dim = clip_model.token_embedding.weight.shape[1]  # 512 for ViT-B/16

        # ------------------------------------------------------------------ #
        # Store frozen CLIP text components as submodules so they move with
        # .to(device), but keep all their parameters non-trainable.
        # NOTE: in OpenAI CLIP, clip_model.transformer is the TEXT transformer
        # (confusingly named); the visual transformer is clip_model.visual.transformer.
        # ------------------------------------------------------------------ #
        self.token_embedding = clip_model.token_embedding
        self.text_transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        for p in (list(self.token_embedding.parameters()) +
                  list(self.text_transformer.parameters()) +
                  list(self.ln_final.parameters())):
            p.requires_grad_(False)

        # text_projection is an nn.Parameter on the CLIP root module.
        # Register as a buffer so it travels with the module but is not
        # included in self.parameters() (and thus not in the optimizer).
        self.register_buffer('positional_embedding', clip_model.positional_embedding.clone())
        self.register_buffer('text_proj', clip_model.text_projection.clone())

        # ------------------------------------------------------------------ #
        # Learnable context vectors — the ONLY trainable parameter.
        # ------------------------------------------------------------------ #
        if ctx_init:
            init_tokens = clip.tokenize(ctx_init)[0]
            with torch.no_grad():
                init_embeds = clip_model.token_embedding(init_tokens)  # [77, D]
            ctx_vectors = init_embeds[1:1 + n_ctx].clone()             # [n_ctx, D]
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim).normal_(std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # [n_ctx, ctx_dim]

        self.n_ctx = n_ctx
        self.context_length = clip_model.context_length  # 77

        # Pre-tokenise all class names; stored as a buffer (not a parameter).
        with torch.no_grad():
            tokenized = clip.tokenize(classnames)  # [N, 77]
        self.register_buffer('tokenized_classes', tokenized)
        self.n_classes = len(classnames)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            embeddings: [n_classes, 512] L2-normalised text embeddings,
                        with gradients flowing through self.ctx.
        """
        N = self.n_classes
        tokenized = self.tokenized_classes  # [N, 77]

        # Embed all class tokens using the frozen embedding table.
        # Use no_grad here; gradients flow through ctx, not token_embedding.
        with torch.no_grad():
            full_embeds = self.token_embedding(tokenized)  # [N, 77, D]

        sos_embed = full_embeds[:, :1, :]       # [N, 1, D]  — SOS token
        class_and_suffix = full_embeds[:, 1:, :]  # [N, 76, D] — class tokens + EOS + PAD

        # Expand ctx to batch: [N, n_ctx, D]
        ctx_expanded = self.ctx.unsqueeze(0).expand(N, -1, -1)

        # Build prompted sequence: [SOS | ctx | class_tokens | EOS | PAD ...]
        # then truncate to context_length (77).
        prompted = torch.cat([sos_embed, ctx_expanded, class_and_suffix], dim=1)
        prompted = prompted[:, :self.context_length, :]  # [N, 77, D]

        # EOS token is at argmax of the tokenised IDs (highest vocab index = EOS).
        # After inserting n_ctx prompt tokens, its position shifts right by n_ctx.
        eos_positions = tokenized.argmax(dim=-1)                                  # [N]
        new_eos_positions = (eos_positions + self.n_ctx).clamp(max=self.context_length - 1)

        # Add positional embeddings and run through the frozen text transformer.
        x = prompted + self.positional_embedding   # [N, 77, D]
        x = x.permute(1, 0, 2)                    # [77, N, D]  (LND convention)
        x = self.text_transformer(x)
        x = x.permute(1, 0, 2)                    # [N, 77, D]
        x = self.ln_final(x)

        # Extract the EOS feature for each class.
        x = x[torch.arange(N, device=x.device), new_eos_positions]  # [N, D]

        # Project to joint embedding space and L2-normalise.
        x = x @ self.text_proj                    # [N, 512]
        return F.normalize(x, p=2, dim=-1)
