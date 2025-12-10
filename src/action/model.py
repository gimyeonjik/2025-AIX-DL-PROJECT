import torch
import torch.nn as nn
from typing import Optional

from config import Config, get_config


class BaseballActionLSTM(nn.Module):
    def __init__(self, input_dim: int = 72, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 13, dropout: float = 0.3, bidirectional: bool = True, use_attention: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * self.num_directions
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.use_attention and mask is not None:
            context = self._attention_pooling(lstm_out, mask)
        elif mask is not None:
            context = self._masked_mean_pooling(lstm_out, mask)
        else:
            context = lstm_out.mean(dim=1)
        logits = self.classifier(context)
        return logits

    def _attention_pooling(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

    def _masked_mean_pooling(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1)
        lstm_out_masked = lstm_out * mask_expanded
        sum_out = lstm_out_masked.sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        context = sum_out / count
        return context

    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_attention:
            return None
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out).squeeze(-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        return attn_weights


class BaseballActionGRU(nn.Module):
    def __init__(self, input_dim: int = 72, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 13, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        gru_output_dim = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        gru_out, h_n = self.gru(x)
        if self.num_directions == 2:
            h_n = h_n.view(self.gru.num_layers, 2, batch_size, -1)
            h_forward = h_n[-1, 0]
            h_backward = h_n[-1, 1]
            context = torch.cat([h_forward, h_backward], dim=1)
        else:
            context = h_n[-1]
        logits = self.classifier(context)
        return logits


def create_model(config: Optional[Config] = None) -> nn.Module:
    config = config or get_config()
    model = BaseballActionLSTM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        use_attention=config.use_attention
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    config = get_config()
    print("모델 테스트...")
    model = create_model(config)
    print(f"\n모델 구조:")
    print(model)
    print(f"\n총 파라미터 수: {count_parameters(model):,}")
    batch_size = 4
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.input_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 50:] = 0
    print(f"\n입력 shape: {x.shape}")
    print(f"마스크 shape: {mask.shape}")
    output = model(x, mask)
    print(f"출력 shape: {output.shape}")
    attn = model.get_attention_weights(x, mask)
    print(f"Attention weights shape: {attn.shape}")
    print(f"Attention weights sum: {attn.sum(dim=1)}")
