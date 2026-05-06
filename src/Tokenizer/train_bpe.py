import os
import json
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train_tokenizer(hyper_para, dataset_dir, tokenizer_path):
    # 1. 加载本地 Arrow 数据集
    dataset = load_from_disk(dataset_dir)
    
    # 2. 定义数据迭代器 (Generator)
    # 这样可以避免将整个数据集一次性加载到内存中
    def batch_iterator(batch_size=1000):
        # 假设数据在 'train' 分片中，openwebtext 通常直接是 Dataset 对象或包含在 'train' 中
        # 如果 load_from_disk 得到的是 DatasetDict，请确保指向具体分片，如 dataset["train"]
        target_dataset = dataset["train"] if "train" in dataset else dataset
        
        for i in range(0, len(target_dataset), batch_size):
            yield target_dataset[i : i + batch_size]["text"]

    # 3. 初始化分词器模型
    tokenizer = Tokenizer(BPE(unk_token=hyper_para["unk_token"]))

    # 4. 配置 Pre-tokenizer 和 Decoder
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 5. 配置训练器
    trainer = BpeTrainer(
        vocab_size=hyper_para["vocab_size"],
        min_frequency=hyper_para["min_frequency"],
        special_tokens=hyper_para["special_tokens"],
        initial_alphabet=ByteLevel.alphabet()
    )

    # 6. 开始训练 (关键修改：从迭代器训练)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))

    # 7. 保存结果
    tokenizer.save(tokenizer_path)
    print(f"Done! Tokenizer saved to {tokenizer_path}")

# --- 调用示例 ---
if __name__ == "__main__":
    conf = json.load(open("./config.json"))

    # 这里的路径是你之前 save_to_disk 的位置
    train_tokenizer(
        hyper_para=conf['train_para'],
        trainset_dir=os.path.join(conf['datasets']['root_dir'], conf['datasets']['openwebtext']),
        tokenizer_path=conf['save_path']
    )