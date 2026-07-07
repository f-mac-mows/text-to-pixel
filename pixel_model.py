import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PixelSeq2SeqTransformer(nn.Module):
    def __init__(self, text_vocab_size=5000, pixel_vocab_size=3000, embed_dim=256, hidden_dim=1024, nhead=8, 
                 num_encoder_layers=4, num_decoder_layers=6, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = embed_dim
        
        # 💡 [보존] 2원화된 독립 임베딩 층 구성
        self.text_embedding = nn.Embedding(text_vocab_size, embed_dim, padding_idx=pad_idx)
        self.pixel_embedding = nn.Embedding(pixel_vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 1. 인코더 층 구성
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        
        # 2. 디코더 층 구성
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        
        # 💡 [보존] 최종 출력 헤드 분리
        self.fc_text_out = nn.Linear(embed_dim, text_vocab_size)
        self.fc_pixel_out = nn.Linear(embed_dim, pixel_vocab_size)
        
    def generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        device = src.device
        
        # 1. 태스크 라우팅 확인 (배치 내 첫 번째 토큰 검사)
        is_text_input = (src[:, 0] < self.text_embedding.num_embeddings)
        is_gen_task = is_text_input[0].item() 

        # 패딩 마스크 생성
        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        
        # 디코더 인과적 마스크 가동
        tgt_mask = self.generate_causal_mask(tgt.size(1), device)
        
        # 2. 태스크 유형에 따른 가변 임베딩 라우팅
        if is_gen_task:
            # TASK_GEN: 입력(Text) -> 출력(Pixel)
            src_emb = self.text_embedding(src)
            tgt_emb = self.pixel_embedding(tgt)
        else:
            # TASK_DESC: 입력(Pixel) -> 출력(Text)
            src_emb = self.pixel_embedding(src)
            tgt_emb = self.text_embedding(tgt)
            
        src_emb = self.pos_encoder(src_emb * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(tgt_emb * math.sqrt(self.d_model))
        
        # 3. 인코더 및 디코더 통과
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        output = self.decoder(
            tgt=tgt_emb, memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # 4. 태스크 유형에 따른 최종 출력 헤드 분기 완전히 복원
        if is_gen_task:
            return self.fc_pixel_out(output)
        else:
            return self.fc_text_out(output)