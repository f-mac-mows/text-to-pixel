import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, Embed_Dim]
        return x + self.pe[:, :x.size(1)]
    
class PixelTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, pad_idx, max_seq_len=256):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        # 💡 디코더를 제거하고 PyTorch 내장 TransformerEncoderLayer 및 Encoder 구성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.2,
            norm_first=True,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 💡 시퀀스 임베딩 벡터를 하나로 합친 뒤 최종 클래스(Vocab) 개수만큼 매핑할 Linear 레이어
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src):
        # src: [Batch, Src_Len]
        # 패딩 토큰 위치를 True로 표시하는 마스크 생성
        src_padding_mask = (src == self.pad_idx)

        # 1. 인풋 자연어 임베딩 및 포지셔널 인코딩
        src_emb = self.pos_encoder(self.embedding(src))

        # 2. 트랜스포머 인코더 통과
        # out: [Batch, Src_Len, Embed_Dim]
        out = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # 3. 💡 풀링 (Pooling) 단계
        # 문장 전체의 의미를 담고 있는 첫 번째 토큰(<SOS>) 위치의 벡터만 추출하거나, 평균(Mean)을 낼 수 있습니다.
        # 자연어 분류에서는 패딩을 제외한 평균 풀링(Mean Pooling)이 성능 안정성에 유리합니다.
        
        # 패딩이 아닌 토큰 마스크 생성 ([Batch, Src_Len, 1])
        mask = (~src_padding_mask).unsqueeze(-1).float()
        
        # 패딩 성분을 0으로 만들고 합산 후, 실제 토큰 개수로 나누어 평균 벡터 계산
        # pooled: [Batch, Embed_Dim]
        sum_embeddings = torch.sum(out * mask, dim=1)
        token_counts = torch.clamp(mask.sum(dim=1), min=1) # 0으로 나누기 방지
        pooled = sum_embeddings / token_counts

        # 4. 최종 로짓 출력 ([Batch, Vocab_Size])
        return self.fc_out(pooled)