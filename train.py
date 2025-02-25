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

class HoverDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict[str, str]]:
        # 加载Hover数据集
        # 这里需要根据实际的数据格式进行调整
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 处理输入文本和目标文本
        input_text = item['input_text']
        target_text = item['target_text']
        
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码目标
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.input_ids.squeeze(),
            'raw_input': input_text,
            'raw_target': target_text
        }

class GRPO:
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        reward_model: nn.Module,
        tokenizer: T5Tokenizer,
        device: torch.device,
        learning_rate: float = 1e-5,
        eps_clip: float = 0.2,
        similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2'),
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 0.9
    ):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.eps_clip = eps_clip
        self.similarity_model = similarity_model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.top_p = top_p
        self.global_step = 0
        
        # 使用AdaFactor优化器
        from transformers import Adafactor
        self.optimizer = Adafactor(
            model.parameters(),
            lr=learning_rate,
            scale_parameter=True,
            relative_step=True,
            warmup_init=True
        )

    def compute_rewards(self, generated_outputs: List[str], reference_outputs: List[str]) -> torch.Tensor:
        # 使用奖励模型计算奖励值
        with torch.no_grad():
            rewards = self.reward_model(generated_outputs, reference_outputs)
        return rewards

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # 动态调整temperature
        self.temperature = max(0.1, self.temperature * 0.995)
        
        # 生成序列
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch['input_ids'].unsqueeze(0),
                attention_mask=batch['attention_mask'].unsqueeze(0),
                max_length=self.max_length,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                temperature=self.temperature,
                top_p=self.top_p
            )
        
        # 解码生成的序列
        generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        reference_texts = [batch['raw_target']]
        
        original_embedding = self.similarity_model.encode([reference_texts])
        corrected_embedding = self.similarity_model.encode([generated_texts])

        # 计算余弦相似度
        similarity = cosine_similarity(original_embedding, corrected_embedding)
        similarity = similarity[0][0]

        if similarity < 0.7:
            rewards = 0.0
        else:
            # 计算奖励
            rewards = self.compute_rewards(generated_texts, reference_texts)
        
        # 计算策略梯度
        old_log_probs = outputs.scores[0].log_softmax(dim=-1)
        new_outputs = self.model(
            input_ids=batch['input_ids'].unsqueeze(0),
            attention_mask=batch['attention_mask'].unsqueeze(0),
            labels=outputs.sequences
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
        loss.backward()
        
        # 更新步骤
        self.global_step += 1
        if self.global_step % self.gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "reward": rewards.mean().item()
        }

def setup_distributed():
    # 初始化分布式环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
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
    model_name = "google/flan-t5-xl"  # 或其他T5模型变体
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # 加载奖励模型（需要根据实际情况修改）
    reward_model = torch.load('path_to_reward_model.pt', map_location=device)
    if world_size > 1:
        reward_model = DDP(reward_model, device_ids=[rank])
    
    # 初始化数据集和分布式采样器
    train_dataset = HoverDataset(
        data_path='path_to_hover_dataset.json',
        tokenizer=tokenizer
    )
    sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # GRPO通常使用小批量
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4
    )
    
    # 初始化训练器
    trainer = GRPO(
        model=model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device
    )
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        total_loss = 0
        total_reward = 0
        for batch in train_dataloader:
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 训练步骤
            metrics = trainer.train_step(batch)
            total_loss += metrics['loss']
            total_reward += metrics['reward']
        
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