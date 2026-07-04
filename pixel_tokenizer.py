import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class PixelArtTokenizerWrapper:
    def __init__(self, vocab_size=2000, tokenizer_path="tokenizer.json"):
        self.input_bpe_space = vocab_size
        self.tokenizer_path = tokenizer_path
        
        self.base_special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        
        self.base_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.base_tokenizer.pre_tokenizer = Whitespace()

        if os.path.exists(self.tokenizer_path):
            self.load()

    @property
    def vocab_size(self):
        return self.base_tokenizer.get_vocab_size()

    @property
    def input_vocab_size(self):
        return self.base_tokenizer.get_vocab_size()

    def train_from_dataset(self, dataset_path):
        """데이터셋에서 input BPE 학습 및 output 문장들을 스페셜 토큰으로 등록"""
        inputs_corpus = []
        outputs_set = set()

        # 💡 분류 성능 방어를 위한 핵심 키워드 보호 풀 빌드
        # 이 단어들은 BPE 알고리즘 안에서 서브워드로 찢어지지 않고 온전한 원형을 보존합니다.
        protected_keywords = [
            "square", "triangle", "cross", "checkerboard", "diamond", "stripes", "frame", "hollow",
            "red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "gray",
            "brown", "gold", "silver", "book", "potion", "bottle", "liquid", "filled", "asset", "pixel",
            "slender", "sleek", "ornate", "reinforced", "basic", "small", "medium", "large", "grand"
        ]

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                inputs_corpus.append(data["input"].lower())
                
                clean_output = data["output"].strip()
                if clean_output:
                    outputs_set.add(clean_output)

        unique_outputs = sorted(list(outputs_set))
        
        # 💡 공통 스페셜 토큰 + 보호할 핵심 키워드 + 분류 타깃용 고유 아웃풋 통째로 병합
        all_special_tokens = self.base_special_tokens + protected_keywords + unique_outputs

        dynamic_vocab_size = self.input_bpe_space + len(all_special_tokens)

        # 동적으로 확장된 크기로 BPE 학습 실행 (보호 키워드가 스페셜 토큰 취급되어 고유 ID를 부여받음)
        trainer = BpeTrainer(vocab_size=dynamic_vocab_size, special_tokens=all_special_tokens)
        self.base_tokenizer.train_from_iterator(inputs_corpus, trainer)

    def save(self):
        self.base_tokenizer.save(self.tokenizer_path)

    def load(self):
        self.base_tokenizer = Tokenizer.from_file(self.tokenizer_path)

    def encode_input(self, text):
        """Input 자연어 텍스트를 정제 후 토큰 ID 리스트로 인코딩"""
        # 특수문자 제거 규칙 고도화 및 소문자 정형화
        clean = text.lower().replace("?", "").replace(",", "").replace(".", "").replace("-", " ").replace("_", " ").strip()
        return self.base_tokenizer.encode(clean).ids

    def encode_output(self, text):
        clean = text.strip()
        token_id = self.base_tokenizer.token_to_id(clean)
        return token_id if token_id is not None else self.unk_id

    def decode_output(self, token_id):
        token_str = self.base_tokenizer.id_to_token(int(token_id))
        return token_str if token_str is not None else "<UNK>"

class PixelArtDataset(Dataset):
    def __init__(self, dataset_path, wrapper_tokenizer):
        self.tokenizer = wrapper_tokenizer
        self.samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "input_ids": torch.tensor(self.tokenizer.encode_input(sample["input"]), dtype=torch.long),
            "target_id": torch.tensor(self.tokenizer.encode_output(sample["output"]), dtype=torch.long)
        }

def get_pixel_collate_fn(pad_id):
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_id"] for item in batch]
        
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
            "target_ids": torch.stack(target_ids)
        }
    return collate_fn