import torch
import random
import json
import os
from typing import Iterator, Tuple
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer


class SFTStreamDataset(IterableDataset):
    """Streams a 70% Hermes / 30% Cocktail mix for post-Bloom arm specialization."""

    def __init__(self, seq_len: int = 1024) -> None:
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self._init_streams()

    def _init_streams(self) -> None:
        """Initialize Hermes live stream (70%) and local Cocktail (30%)."""
        from datasets import load_dataset
        hf_token = os.environ.get('HF_TOKEN', None)
        print("[SFT Generator] Initializing 70/30 Hermes+Cocktail (Specialization Phase)...")

        # Live Hermes stream — 70% of batches
        try:
            ds_hermes = load_dataset(
                "teknium/OpenHermes-2.5",
                split="train",
                streaming=True,
                token=hf_token
            )
            self.hermes_iter = iter(ds_hermes)
            print("[SFT Generator] OpenHermes-2.5 stream opened.")
        except Exception as e:
            print(f"[SFT Generator] Hermes failed: {e}. Pure GSM8K fallback.")
            ds = load_dataset("gsm8k", "main", split="train", streaming=False).shuffle(seed=42)
            self.hermes_iter = iter(ds)

        # Local Cocktail — 30% of batches, for math/logic arm specialization
        try:
            self.ds_gsm = load_dataset("gsm8k", "main", split="train", streaming=False).shuffle(seed=99)
            self.ds_arc = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=False).shuffle(seed=99)
            self.iter_gsm = iter(self.ds_gsm)
            self.iter_arc = iter(self.ds_arc)
            self.premium_file = open("/hdd_data/mamba_training_data/premium_reasoning.jsonl", "r", encoding="utf-8")
            print("[SFT Generator] Local Cocktail datasets loaded.")
        except Exception as e:
            print(f"[SFT Generator] Cocktail partial load: {e}")

    def _get_hermes_sample(self) -> Tuple[str, str]:
        """Pull and format a sample from the OpenHermes-2.5 stream."""
        try:
            sample = next(self.hermes_iter)
        except StopIteration:
            self._init_streams()
            sample = next(self.hermes_iter)

        try:
            convs = sample.get('conversations', [])
            if not convs:
                return "User: Hello\nAssistant: ", "Hello!<|endoftext|>\n"

            turns = []
            for turn in convs:
                role = turn.get('from', '')
                value = turn.get('value', '').strip()
                if role in ('human', 'user'):
                    turns.append(f"User: {value}")
                elif role in ('gpt', 'assistant'):
                    turns.append(f"Assistant: {value}")

            if len(turns) >= 2:
                user_text = "\n".join(turns[:-1]) + "\n" + turns[-1].split(":")[0] + ": "
                a_text = f"{convs[-1].get('value', '').strip()}<|endoftext|>\n"
            else:
                user_text = "User: Hello\nAssistant: "
                a_text = "Hello!<|endoftext|>\n"
        except Exception:
            user_text = "User: Hello\nAssistant: "
            a_text = "Hello!<|endoftext|>\n"

        return user_text, a_text

    def _get_cocktail_sample(self) -> Tuple[str, str]:
        """Pull and format a math/logic sample from the local Cocktail."""
        domain = random.choices(["gsm", "arc", "premium"], weights=[0.25, 0.25, 0.50])[0]
        try:
            if domain == "gsm":
                try:
                    sample = next(self.iter_gsm)
                except StopIteration:
                    self.iter_gsm = iter(self.ds_gsm)
                    sample = next(self.iter_gsm)
                user_text = f"User: {sample['question']}\nAssistant: "
                a_text = f"<<answer={sample['answer']}>> {sample['answer']}<|endoftext|>\n"

            elif domain == "arc":
                try:
                    sample = next(self.iter_arc)
                except StopIteration:
                    self.iter_arc = iter(self.ds_arc)
                    sample = next(self.iter_arc)
                user_text = f"User: Question: {sample['question']}\n"
                if 'choices' in sample:
                    for lbl, txt in zip(sample['choices']['label'], sample['choices']['text']):
                        user_text += f"{lbl}. {txt}\n"
                user_text += "Answer: \nAssistant: "
                a_text = f"<<answer={sample['answerKey']}>> The correct answer is {sample['answerKey']}.<|endoftext|>\n"

            else:
                line = self.premium_file.readline()
                if not line:
                    self.premium_file.seek(0)
                    line = self.premium_file.readline()
                s = json.loads(line)
                user_text = f"User: {s['prompt']}\nAssistant: "
                a_text = f"{s['answer']}<|endoftext|>\n"

        except Exception:
            user_text = "User: Hello\nAssistant: "
            a_text = "Hello!<|endoftext|>\n"

        return user_text, a_text

    def _get_next_sample(self) -> Tuple[str, str]:
        """90% Hermes, 10% Cocktail for balanced multi-domain arm specialization."""
        if random.random() < 0.90:
            return self._get_hermes_sample()
        return self._get_cocktail_sample()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Pack samples into fixed-length sequences with -100 masking on prompts."""
        buffer_tokens = []
        buffer_targets = []

        while True:
            if getattr(self, 'hermes_iter', None) is None:
                yield (
                    torch.randint(0, 50304, (self.seq_len,), dtype=torch.long),
                    torch.randint(0, 50304, (self.seq_len,), dtype=torch.long)
                )
                continue

            user_text, a_text = self._get_next_sample()

            u_toks = self.tokenizer.encode(user_text)
            a_toks = self.tokenizer.encode(a_text)

            buffer_tokens.extend(u_toks + a_toks)
            buffer_targets.extend([-100] * len(u_toks) + a_toks)

            while len(buffer_tokens) >= self.seq_len + 1:
                seq = buffer_tokens[:self.seq_len + 1]
                tgt = buffer_targets[:self.seq_len + 1]

                x = torch.tensor(seq[:-1], dtype=torch.long)
                y = torch.tensor(tgt[1:], dtype=torch.long)

                yield x, y

                buffer_tokens = buffer_tokens[self.seq_len:]
                buffer_targets = buffer_targets[self.seq_len:]


def get_sft_dataloader(batch_size: int = 4, seq_len: int = 1024) -> DataLoader:
    """Returns the 70/30 Hermes+Cocktail specialization DataLoader."""
    dataset = SFTStreamDataset(seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size)
