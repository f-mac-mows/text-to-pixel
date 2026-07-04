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
        # 인자값으로 받는 vocab_size는 이제 '순수 자연어(Input) 학습용 기본 공간' 의미로 사용됩니다.
        self.input_bpe_space = vocab_size
        self.tokenizer_path = tokenizer_path
        
        # 기본 공통 스페셜 토큰 정의 (PAD=0, SOS=1, EOS=2, UNK=3)
        self.base_special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        
        self.base_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.base_tokenizer.pre_tokenizer = Whitespace()

        if os.path.exists(self.tokenizer_path):
            self.load()

    @property
    def vocab_size(self):
        """기본 BPE 단어 + 스페셜 토큰으로 등록된 고유 Output 문장이 모두 포함된 총 크기"""
        return self.base_tokenizer.get_vocab_size()

    @property
    def input_vocab_size(self):
        """이전 코드 및 외부 훈련 인스턴스 규칙(bpe_vocab.input_vocab_size) 호환용 프로퍼티"""
        return self.base_tokenizer.get_vocab_size()

    def train_from_dataset(self, dataset_path):
        """데이터셋에서 input BPE 학습 및 output 문장들을 스페셜 토큰으로 등록"""
        inputs_corpus = []
        outputs_set = set()

        # 1. 파일 전체를 읽어 input과 고유 output 수집
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                inputs_corpus.append(data["input"].lower())
                
                # 아웃풋 문장은 형태 그대로 보존하며 중복 제거
                clean_output = data["output"].strip()
                if clean_output:
                    outputs_set.add(clean_output)

        # 2. 고유 아웃풋 리스트 생성 (순서 고정을 위해 정렬)
        unique_outputs = sorted(list(outputs_set))
        
        # 3. 토크나이저가 인식할 전체 스페셜 토큰 목록 구성
        all_special_tokens = self.base_special_tokens + unique_outputs

        # 💡 [핵심 방어선] 데이터 급증으로 인한 사증 크기 부족(ValueError) 원천 차단
        # 초기에 지정한 기본 텍스트 공간 공간에 '추가된 고유 이미지 토큰 개수'를 동적으로 더해줍니다.
        dynamic_vocab_size = self.input_bpe_space + len(all_special_tokens)

        # 4. 동적으로 확장된 크기로 BPE 학습 실행
        trainer = BpeTrainer(vocab_size=dynamic_vocab_size, special_tokens=all_special_tokens)
        self.base_tokenizer.train_from_iterator(inputs_corpus, trainer)

    def save(self):
        """토크나이저 파일 하나에 BPE 어휘와 스페셜 토큰 매핑 정보가 모두 깔끔하게 저장됩니다."""
        self.base_tokenizer.save(self.tokenizer_path)

    def load(self):
        self.base_tokenizer = Tokenizer.from_file(self.tokenizer_path)

    def encode_input(self, text):
        """Input 자연어 텍스트를 BPE 토큰 ID 리스트로 인코딩"""
        clean = text.lower().replace("?", "").replace(",", "").replace(".", "")
        return self.base_tokenizer.encode(clean).ids

    def encode_output(self, text):
        """Output 통문장 전체를 토크나이저 사전에서 찾아 단 하나의 스페셜 토큰 ID로 반환"""
        clean = text.strip()
        token_id = self.base_tokenizer.token_to_id(clean)
        return token_id if token_id is not None else self.unk_id

    def decode_output(self, token_id):
        """모델이 예측한 토큰 ID(정수)를 원래의 통문장으로 복원"""
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