import torch 
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding 
from peft import LoraConfig, get_peft_model, TaskType

peft_model_name = 'roberta-base-peft' 
modified_base = 'roberta-base-modified' 
base_model = 'roberta-base' 

class LORA(nn.Module):
    def __init__(self):
        super(LORA, self).__init__()

        base_model = RobertaModel.from_pretrained('/media/cygzz/data/guanjl/Code_self/roberta-base/')
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],  
            bias="none",            inference_mode=False
        )
        
        
        for param in base_model.parameters():
            param.requires_grad = False
        
        self.model = get_peft_model(base_model, lora_config)
        
        
    
        self.model.print_trainable_parameters()

    def forward(self, features, attention_mask):
        """
        参数说明：
        - features: 输入特征，shape [batch_size, seq_len, hidden_size]（如 [2,25,768]）
        - attention_mask: 注意力掩码，shape [batch_size, seq_len]（如 [2,25]）
        """
        # 关键：用关键字传参调用模型，避免解析为 input_ids
        outputs = self.model(
            inputs_embeds=features,       # 明确指定预计算特征（替代 input_ids）
            attention_mask=attention_mask, # 明确指定注意力掩码
            output_hidden_states=False     # 不返回所有隐藏层，节省显存
        )
        # 获取最后一层隐藏状态（输出维度 [2,25,768]）
        encoder_hidden = outputs.last_hidden_state
        return encoder_hidden
  

        





if __name__ == "__main__":
    
    model=LORA()
    model=model.cuda()
    text_features=torch.ones((2,25,768)).cuda()
    id=torch.ones((2,25)).cuda()
    out=model(text_features,id)
    print("finish")