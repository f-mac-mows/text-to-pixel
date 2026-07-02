import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. 픽셀 컴파일러 전용 고유 보카(Vocabulary) 빌더 (안정화 버전)
# ==========================================
class PixelVocabulary:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]

        self.token_to_id = {}
        self.id_to_token = {}

        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token_to_id:
            new_id = len(self.token_to_id)
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token

    def _tokenize_output(self, output_text):
        """✨ 핵심: 대괄호([, ])를 공백으로 격리하여 모델이 괄호 문법을 온전하게 인지하도록 분리합니다."""
        # "[Re2][16W 1L]" -> " [Re2]  [16W   1L] " 형태로 변환 후 쪼갭니다.
        spaced = output_text.replace("[", " [ ").replace("]", " ] ")
        return spaced.split()

    def build_vocab(self, dataset_path):
        """데이터셋을 훑으며 고유 입력 단어와 대괄호가 분리된 RLE 토큰들을 사전에 청정 등록합니다."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # 입력 자연어 등록 (물음표 및 구두점 cleaning 일치)
                clean_input = data["input"].lower().replace("?", "").replace(",", "").replace(".", "")
                for word in clean_input.split():
                    self._add_token(word)

                # 출력 RLE 토큰 등록 (괄호 분리 적용)
                for rle_token in self._tokenize_output(data["output"]):
                    self._add_token(rle_token)

        print(f"[*] 고유 보카 사전 빌드 완료! 최적화된 총 토큰 개수(Vocab Size): {len(self.token_to_id)}")

    def encode(self, text, is_input=True):
        """텍스트 시퀀스를 정수 ID 리스트로 변환합니다."""
        if is_input:
            clean_text = text.lower().replace("?", "").replace(",", "").replace(".", "")
            tokens = clean_text.split()
        else:
            tokens = self._tokenize_output(text)

        ids = []
        if not is_input:
            ids.append(self.token_to_id[self.sos_token])

        for token in tokens:
            ids.append(self.token_to_id.get(token, self.token_to_id[self.unk_token]))

        if not is_input:
            ids.append(self.token_to_id[self.eos_token])

        return ids
    
    def decode(self, ids):
        """✨ ID 매핑 기반으로 특수 토큰(<PAD>, <SOS>, <EOS>)을 확실하게 필터링하여 디코딩합니다."""
        special_ids = {self.token_to_id[self.pad_token], self.token_to_id[self.sos_token], self.token_to_id[self.eos_token]}
        
        decoded_tokens = []
        for idx in ids:
            if idx in special_ids:
                continue
            decoded_tokens.append(self.id_to_token.get(idx, self.unk_token))
            
        # 디코딩 시 분리했던 괄호 구조를 다시 타이트하게 원복시켜 렌더러 호환성을 유지합니다.
        result = " ".join(decoded_tokens)
        return result.replace("[ ", "[").replace(" ]", "]")
    

# ==========================================
# 2. 파이토치 커스텀 Dataset 정의 (기존 유지)
# ==========================================
class PixelArtDataset(Dataset):
    def __init__(self, dataset_path, vocab):
        self.vocab = vocab
        self.samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        input_ids = self.vocab.encode(sample["input"], is_input=True)
        target_ids = self.vocab.encode(sample["output"], is_input=False)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }
    

# ==========================================
# 3. 미니 배치를 위한 패딩 콜레이트(Collate) 함수 (기존 유지)
# ==========================================
def pixel_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(target_ids, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_inputs,
        "target_ids": padded_targets
    }


# ==========================================
# 4. 모듈 검증 테스트 실행
# ==========================================
if __name__ == "__main__":
    # 데이터셋 명칭에 맞게 v2 또는 v3 매핑
    dataset_file = "pixel_dataset_v2.jsonl" 

    vocab = PixelVocabulary()
    vocab.build_vocab(dataset_file)

    pixel_dataset = PixelArtDataset(dataset_file, vocab)
    data_loader = DataLoader(
        pixel_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=pixel_collate_fn
    )

    first_batch = next(iter(data_loader))
    print("\n[+] 패딩 데이터로더 텐서 검증 완료:")
    print(f" - Input Tensor Shape (Batch, Len) : {first_batch['input_ids'].shape}")
    print(f" - Target Tensor Shape (Batch, Len): {first_batch['target_ids'].shape}")
    
    print("\n[*] 실제 변환 매핑 테스트:")
    sample_input_ids = first_batch['input_ids'][0].tolist()
    sample_target_ids = first_batch['target_ids'][0].tolist()
    print(f" - Raw Input IDs  : {sample_input_ids}")
    print(f" - Decoded Input  : {vocab.decode(sample_input_ids)}")
    print(f" - Raw Target IDs : {sample_target_ids}")
    print(f" - Decoded Target : {vocab.decode(sample_target_ids)}")