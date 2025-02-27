import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ProgramFC.models.program_generator import Reasoning_Program_Generator
from ProgramFC.models.program_execution import Program_Execution
from torch.cuda.amp import autocast, GradScaler

class HoverDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, max_length: int = 2048):  # 减小max_length默认值
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict[str, str]]:
        # 加载Hover数据集
        # 这里需要根据实际的数据格式进行调整
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 过滤出label为refutes的数据
        data = [item for item in data if item.get('label') == 'refutes']
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 处理输入文本和目标文本
        input_text = item['claim']
        evidence = item['evidence']
        id = item['id']
        num_hops = item['num_hops']
        
        # 添加prompt引导生成
        prompt = """You are an experienced expert in correcting erroneous sentences. Please read the following evidence carefully and correct the errors in the original statement.

                Task: Based on the provided evidence, identify the incorrect parts of the original statement and fix them to ensure accuracy.

                Requirements:The generated statement must be a complete sentence.Maintain the same theme and core meaning as the original statement.Correct the erroneous information based on the evidence.Use clear and accurate language.All modifications must be supported by evidence.You can't delete the entire sentence that contains the wrong part of the sentence, but rather fix the error.

                Evidence: {evidence}

                Original statement: {original_statement}

                Corrected statement: """
        input_text = prompt.format(evidence=evidence, original_statement=input_text)
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            evidence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )     
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'raw_input': input_text,
            'raw_evidence': evidence,
            'id' : id,
            'num_hops': num_hops
        }

class GRPO:
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        device: torch.device,
        learning_rate: float = 1e-5,
        eps_clip: float = 0.2,
        similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2'),
        gradient_accumulation_steps: int = 8,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_length: int = 512,  # 减小max_length默认值
        program_generator = Reasoning_Program_Generator(),
        program_executor = Program_Execution()
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eps_clip = eps_clip
        self.similarity_model = similarity_model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length  # 设置max_length属性
        self.global_step = 0
        self.program_generator = program_generator
        self.program_executor = program_executor
        self.scaler = torch.amp.GradScaler(device='cuda')  
        
        # 使用AdaFactor优化器，使用自动学习率调整
        from transformers import Adafactor
        self.optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # 动态调整temperature
        self.temperature = max(0.1, self.temperature * 0.995)
        
        # 生成序列
        with torch.no_grad():
            # 确保输入维度正确
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 添加批次维度如果需要
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            outputs = self.model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
        
        # 使用FP16进行前向传播和反向传播
        with autocast():
            # 解码生成的序列并确保正确处理文本
            generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            reference_texts = batch['raw_input']
            
            try:
                # 从原文本中提取实际的声明内容
                if isinstance(reference_texts, list):
                    original_text = reference_texts[0].split("Original statement:")[-1].strip()
                else:
                    original_text = reference_texts.split("Original statement:")[-1].strip()
                    
                # 确保生成的文本不为空且长度合理
                if not generated_texts or not generated_texts[0].strip():
                    print("警告：生成的文本为空")
                    return {"loss": 0.0, "reward": 0.0}
                    
                if len(generated_texts[0].split()) < 3:
                    print("警告：生成的文本过短")
                    return {"loss": 0.0, "reward": 0.0}
            except Exception as e:
                print(f"处理文本时出错：{str(e)}")
                return {"loss": 0.0, "reward": 0.0}
            evidence_texts = batch['raw_evidence']

            print(f"\n输入给模型的文本为:{reference_texts[0]}")
            print(f"\n原声明为:{original_text}")
            print(f"\n生成的声明为:{generated_texts[0]}\n")
            
            original_embedding = self.similarity_model.encode(original_text)
            corrected_embedding = self.similarity_model.encode(generated_texts[0])

            # 调整向量维度并计算余弦相似度
            original_embedding = original_embedding.reshape(1, -1)
            corrected_embedding = corrected_embedding.reshape(1, -1)
            similarity = cosine_similarity(original_embedding, corrected_embedding)
            similarity = similarity[0][0]
            print(f"\n余弦相似度为:{similarity}\n")

            if similarity < 0.7:
                rewards = 0.0
            else:
                decomposing_output = self.program_generator.batch_generate_programs(generated_texts)
                print(decomposing_output)
                sample = [{
                    "idx":0,
                    "id":batch['id'],
                    "claim":generated_texts,
                    "gold":"",
                    "predicted_programs":decomposing_output,
                    "evidence":evidence_texts
                }]
                final_prediction = self.program_executor.execute_on_dataset(sample)
                if final_prediction:
                    rewards = 1.0
                else:
                    rewards = 0.0
            
            print(f"\nrewards为:{rewards}\n")

            # 计算策略梯度
            old_log_probs = outputs.scores[0].log_softmax(dim=-1)
            new_outputs = self.model(
                input_ids=input_ids,  # 直接使用已经处理过维度的input_ids
                attention_mask=attention_mask,  # 直接使用已经处理过维度的attention_mask
                labels=outputs.sequences.to(self.device)
            )
            new_log_probs = new_outputs.logits.log_softmax(dim=-1)
            
            # 计算比率
            ratio = (new_log_probs - old_log_probs).exp()
            
            # 计算裁剪后的目标
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * rewards
            loss = -torch.min(surr1, surr2).mean()
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps

        # 使用FP16进行反向传播
        self.scaler.scale(loss).backward()
        
        # 更新步骤
        self.global_step += 1
        if self.global_step % self.gradient_accumulation_steps == 0:
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            # 使用FP16更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "reward": rewards
        }

def setup_distributed():
    # 初始化分布式环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    if world_size > 1:
        dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo',
                               rank=rank, world_size=world_size)
    
    return device, rank, world_size

def main():
    # 设置分布式环境
    device, rank, world_size = setup_distributed()
    
    # 初始化模型和tokenizer
    model_name = "google/flan-t5-large"  # 或其他T5模型变体
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16  # 使用FP16
    ).to(device)

    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    if world_size > 1:
        # 在分布式训练中使用DDP包装模型
        model = DDP(model.to(device), device_ids=[rank % torch.cuda.device_count()])
    
    # 初始化数据集和分布式采样器
    train_dataset = HoverDataset(
        data_path='./train.json',
        tokenizer=tokenizer
    )
    sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # 使用较小的batch_size以减少内存使用
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4
    )
    
    # 设置梯度累积步数
    gradient_accumulation_steps = 8  # 增加梯度累积步数
    
    # 初始化训练器
    trainer = GRPO(
        model=model,
        tokenizer=tokenizer,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        total_loss = 0
        total_reward = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 训练步骤
            metrics = trainer.train_step(batch)
            # 根据梯度累积步数缩放损失
            metrics['loss'] = metrics['loss'] / gradient_accumulation_steps
            total_loss += metrics['loss']
            total_reward += metrics['reward']
            
            # 每累积指定步数的梯度后才进行参数更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        if world_size > 1:
            # 在多GPU环境下同步损失和奖励
            dist.all_reduce(torch.tensor([total_loss, total_reward]).to(device))
            total_loss /= world_size
            total_reward /= world_size
        
        # 只在主进程中打印信息和保存模型
        if rank == 0:
            avg_loss = total_loss / len(train_dataloader)
            avg_reward = total_reward / len(train_dataloader)
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Reward: {avg_reward:.4f}')
            
            # 保存模型
            if (epoch + 1) % 5 == 0:
                if isinstance(model, DDP):
                    model.module.save_pretrained(f'model_checkpoint_epoch_{epoch+1}')
                else:
                    model.save_pretrained(f'model_checkpoint_epoch_{epoch+1}')
        
        if world_size > 1:
            dist.barrier()  # 确保所有进程同步

if __name__ == "__main__":
    main()