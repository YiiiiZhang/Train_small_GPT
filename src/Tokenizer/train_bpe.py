import os 
import glob
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def trian_tokenizer(hyper_para,trainset_dir,tokenizer_path):
    #model_initialize
    tokenizer = Tokenizer(BPE(unk_token = hyper_para["unk_token"]))
    #配置 Pre-tokenizer 和 Decoder;add_prefix_space=False 意味着不强制在每个句首加空格
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size = hyper_para["vocab_size"],
        min_frequency = hyper_para["min_frequency"], # 出现少于 2 次的字符对不进行合并
        special_tokens = hyper_para["special_tokens"],
        initial_alphabet=ByteLevel.alphabet()
    )

    #loading_data
    files = glob.glob(trainset_dir)
    #trianing
    tokenizer.train(files, trainer)
    tokenizer.save(tokenizer_path)

    print("Done!")

def validate_bpe(tokenizer_path, valid_file_path):
    # 1. 加载 BPE
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    num_chars = 0
    num_tokens = 0
    num_words = 0
    unk_count = 0
    
    # 2. 逐行读取和处理，彻底解决 OOM 问题
    with open(valid_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): # 跳过空行
                continue
                
            # 统计字符和词数
            num_chars += len(line)
            num_words += len(line.split())
            
            # 3. 逐行进行分词
            encoded = tokenizer.encode(line)
            num_tokens += len(encoded.tokens)
            unk_count += encoded.tokens.count("<UNK>")
    
    # 4. 计算指标 (加入除零保护)
    compression_ratio = num_chars / num_tokens if num_tokens > 0 else 0
    fertility = num_tokens / num_words if num_words > 0 else 0
    
    print("=== BPE 验证报告 ===")
    print(f"验证集总字符数: {num_chars}")
    print(f"产生 Token 总数: {num_tokens}")
    print(f"压缩率 (Chars/Token): {compression_ratio:.2f} (越高越好，通常3~4为佳)")
    print(f"词碎片率 (Tokens/Word): {fertility:.2f} (越接近1越好)")
    print(f"UNK 数量: {unk_count}")

if __name__ == "__main__":
    conf = json.load(open('./config.json',"r"))
    trian_tokenizer(conf['train_para'],conf['datasets']['trainset'],conf['save_path'])
    validate_bpe(conf['save_path'], conf['datasets']['validset'])