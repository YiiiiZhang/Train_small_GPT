import os
import json
import random
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def mixed_corpus_iterator(tiny_path,owe_path,sample_size=200000):
    """
    load dataset from disk and yield text data for tokenizer training
    """
    ds_tiny = load_from_disk(tiny_path)
    ds_owe = load_from_disk(owe_path)
    iter_tiny = iter(ds_tiny["train"])
    iter_owt = iter(ds_owe["train"])
    for _ in range(sample_size):
        # 随机决定从哪个数据集拿一条数据
        if random.random() < 0.5:
            try:
                yield next(iter_tiny)['text']
            except StopIteration:
                pass
        else:
            try:
                yield next(iter_owt)['text']
            except StopIteration:
                pass


def train_tokenizer(bpe_para, dataset_paras, tokenizer_path):
    # 3. 初始化分词器模型
    tokenizer = Tokenizer(BPE(unk_token=bpe_para["unk_token"]))

    # 4. 配置 Pre-tokenizer 和 Decoder
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=bpe_para["vocab_size"],
        min_frequency=bpe_para["min_frequency"],
        special_tokens=bpe_para["special_tokens"],
        initial_alphabet=ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(
        mixed_corpus_iterator(
            tiny_path=os.path.join(dataset_paras['root_dir'], dataset_paras['TinyStories']),
            owe_path=os.path.join(dataset_paras['root_dir'], dataset_paras['openwebtext'])
        ),
        trainer=trainer, 
    )

    tokenizer.save(tokenizer_path)
    print(f"Done! Tokenizer saved to {tokenizer_path}")

# --- 调用示例 ---
if __name__ == "__main__":
    conf = json.load(open("./config.json"))

    train_tokenizer(
        bpe_para=conf['bpe_para'],
        dataset_paras=conf['datasets'],
        tokenizer_path=conf['save_path']
    )