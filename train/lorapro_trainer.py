import peft
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from train.peta.optim import AdamW
from transformers import TrainingArguments, Trainer 

# Configure custom trainer with lora-pro optimizer
class LoraProTrainer(Trainer):
    def __init__(self, *args, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)

        self.scaling_factor = scaling_factor
        self.lora_modules = []
        self.find_modules(self.model, self.lora_modules)

    def find_modules(self, module ,lora_modules):
        for sub_module in module.children():
            if isinstance(sub_module, peft.tuners.lora.layer.Linear):
                lora_modules.append(sub_module)
            elif list(sub_module.children()):
                self.find_modules(sub_module, lora_modules)
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = AdamW(self.model.named_parameters(), lr=self.args.learning_rate, scaling_factor=self.scaling_factor, betas=(0.9, 0.999), weight_decay=self.args.weight_decay, mode="efficient", X_mode='sylvester')
        
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)