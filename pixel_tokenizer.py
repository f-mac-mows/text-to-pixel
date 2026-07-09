import os
import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit

# ==========================================
# 1. 영어 프롬프트 전용 BPE 토크나이저
# ==========================================
class TextBpeTokenizerWrapper:
    def __init__(self, vocab_size=5000, tokenizer_path="pixel_text_tokenizer.json"):
        self.target_vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<TASK_GEN>", "<TASK_DESC>"]
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        
        self.base_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.base_tokenizer.pre_tokenizer = Whitespace()

        if os.path.exists(self.tokenizer_path):
            self.load()

    def train_from_dataset(self, dataset_path):
        def corpus_iterator():
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    if data["input"].startswith("<TASK_GEN>"):
                        yield data["input"].replace("<TASK_GEN>", "").strip()
                    if "input" in data and data["input"].startswith("<TASK_DESC>"):
                        yield data["output"].strip()

        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            min_frequency=1,
            special_tokens=self.special_tokens
        )
        self.base_tokenizer.train_from_iterator(corpus_iterator(), trainer)
        self.base_tokenizer.save(self.tokenizer_path)

    def encode(self, text, add_sos_eos=True):
        # 💡 [방어 코드] 미공개 테스트셋에서 완전 무작위 Out-of-Vocabulary 발생 시 크래시 원천 차단
        try:
            token_ids = self.base_tokenizer.encode(text).ids
        except Exception:
            token_ids = []
            for char in text:
                try:
                    token_ids.extend(self.base_tokenizer.encode(char).ids)
                except Exception:
                    continue  # 사전에 도저히 없는 이모지나 글자는 스킵
                    
        if add_sos_eos:
            return [self.sos_id] + token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # 💡 [방어 코드] token_to_id가 None을 반환할 것에 대비하여 필터 집합 구성 변경
        ignore_ids = {self.pad_id, self.sos_id, self.eos_id}
        for token_name in ["<TASK_GEN>", "<TASK_DESC>", "<UNK>"]:
            tid = self.base_tokenizer.token_to_id(token_name)
            if tid is not None:
                ignore_ids.add(tid)
                
        filtered_ids = [tid for tid in token_ids if tid not in ignore_ids]
        return self.base_tokenizer.decode(filtered_ids)

    def load(self):
        self.base_tokenizer = Tokenizer.from_file(self.tokenizer_path)


# ==========================================
# 2. RLE 픽셀 프로토콜 전용 WordLevel 토크나이저
# ==========================================
class PixelWordTokenizerWrapper:
    def __init__(self, tokenizer_path="pixel_pixel_tokenizer.json"):
        self.tokenizer_path = tokenizer_path
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<TASK_GEN>", "<TASK_DESC>"]
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        
        self.base_tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        self.base_tokenizer.pre_tokenizer = WhitespaceSplit()

        if os.path.exists(self.tokenizer_path):
            self.load()

    def train_from_dataset(self, dataset_path):
        def corpus_iterator():
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    if data["input"].startswith("<TASK_DESC>"):
                        yield data["input"].replace("<TASK_DESC>", "").strip()
                    if data["input"].startswith("<TASK_GEN>"):
                        yield data["output"].strip()

        trainer = WordLevelTrainer(vocab_size=5000, min_frequency=1, special_tokens=self.special_tokens)
        self.base_tokenizer.train_from_iterator(corpus_iterator(), trainer)
        self.base_tokenizer.save(self.tokenizer_path)

    def encode(self, text, add_sos_eos=True):
        try:
            token_ids = self.base_tokenizer.encode(text).ids
        except Exception:
            token_ids = []
            for word in text.split():
                try:
                    token_ids.extend(self.base_tokenizer.encode(word).ids)
                except Exception:
                    continue
                    
        if add_sos_eos:
            return [self.sos_id] + token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # 💡 [방어 코드] token_to_id가 None을 반환하더라도 안전하게 정수 매핑 셋 확보
        ignore_ids = {self.pad_id, self.sos_id, self.eos_id}
        for token_name in ["<TASK_DESC>", "<TASK_GEN>", "<UNK>"]:
            tid = self.base_tokenizer.token_to_id(token_name)
            if tid is not None:
                ignore_ids.add(tid)
                
        # id_to_token 시 발생할 수 있는 결측 에러 방지 가드
        filtered_tokens = []
        for tid in token_ids:
            if tid in ignore_ids:
                continue
            try:
                t = self.base_tokenizer.id_to_token(tid)
                if t is not None:
                    filtered_tokens.append(t)
            except Exception:
                continue
                
        return " ".join(filtered_tokens)

    def load(self):
        self.base_tokenizer = Tokenizer.from_file(self.tokenizer_path)


# ==========================================
# 3. 2원화 토크나이저 대응 대칭형 Dataset
# ==========================================
class HybridPixelArtDataset(Dataset):
    def __init__(self, dataset_path, text_tokenizer, pixel_tokenizer):
        self.text_tokenizer = text_tokenizer
        self.pixel_tokenizer = pixel_tokenizer
        self.samples = []
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        raw_input = sample["input"]
        raw_output = sample["output"]

        if raw_input.startswith("<TASK_GEN>"):
            pure_text = raw_input.replace("<TASK_GEN>", "").strip()
            gen_task_id = self.text_tokenizer.base_tokenizer.token_to_id("<TASK_GEN>")
            # 💡 혹시라도 토크나이저 내부에서 매핑을 못 찾으면 임시 고정 아이디 배정
            if gen_task_id is None: gen_task_id = 4 
            
            input_ids = [self.text_tokenizer.sos_id, gen_task_id] + self.text_tokenizer.encode(pure_text, add_sos_eos=False) + [self.text_tokenizer.eos_id]
            target_ids = self.pixel_tokenizer.encode(raw_output, add_sos_eos=True)
            task_type = "GEN"

        elif raw_input.startswith("<TASK_DESC>"):
            pure_pixel = raw_input.replace("<TASK_DESC>", "").strip()
            desc_task_id = self.pixel_tokenizer.base_tokenizer.token_to_id("<TASK_DESC>")
            if desc_task_id is None: desc_task_id = 5
            
            input_ids = [self.pixel_tokenizer.sos_id, desc_task_id] + self.pixel_tokenizer.encode(pure_pixel, add_sos_eos=False) + [self.pixel_tokenizer.eos_id]
            target_ids = self.text_tokenizer.encode(raw_output, add_sos_eos=True)
            task_type = "DESC"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "task_type": task_type
        }