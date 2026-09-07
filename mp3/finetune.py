import logging
import numpy as np
from torch import nn
import torch
from vision_transformer import vit_b_32, ViT_B_32_Weights


def get_encoder(name):
    if name == 'vit_b_32':
        torch.hub.set_dir("model")
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    return model


class ViTLinear(nn.Module):
    def __init__(self, n_classes, encoder_name, num_prompts=None):
        super(ViTLinear, self).__init__()
        
        self.vit_b = [get_encoder(encoder_name)]
        
        # Reinitialize the head with a new layer
        self.vit_b[0].heads[0] = nn.Identity()
        self.linear = nn.Linear(768, n_classes)
    
    def to(self, device):
        super(ViTLinear, self).to(device)
        self.vit_b[0] = self.vit_b[0].to(device)
        return self
    
    def forward(self, x):
        with torch.no_grad():
            out = self.vit_b[0](x)
        y = self.linear(out)
        return y


class VPTDeep(nn.Module):
    """
    Visual Prompt Tuning (Deep) implementation.
    Adds learnable prompts at every transformer layer's input.
    """
    def __init__(self, n_classes, encoder_name='vit_b_32', num_prompts=10, prompt_dim=768):
        super(VPTDeep, self).__init__()
        
        # Load pre-trained ViT
        torch.hub.set_dir("model")
        self.vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        
        # Freeze the entire ViT backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Get model parameters
        self.num_layers = 12  # ViT-B has 12 layers
        self.hidden_dim = prompt_dim  # 768 for ViT-B
        self.num_prompts = num_prompts
        
        # Initialize learnable prompts for each layer
        # Shape: (1, num_layers, num_prompts, hidden_dim)
        # Using Xavier uniform initialization as suggested
        v = (6.0 / (self.hidden_dim + self.num_prompts)) ** 0.5
        self.prompts = nn.Parameter(
            torch.zeros(1, self.num_layers, self.num_prompts, self.hidden_dim)
        )
        nn.init.uniform_(self.prompts, -v, v)
        
        # Replace the classification head
        self.vit.heads = nn.Identity()
        self.head = nn.Linear(self.hidden_dim, n_classes)
        
        # Log number of trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"Num prompts per layer: {num_prompts}")
        
    def forward(self, x):
        # Process input through conv_proj and add class token
        x = self.vit._process_input(x)
        n = x.shape[0]
        
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Add positional embedding
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        
        # Pass through encoder layers with prompts
        for i, layer in enumerate(self.vit.encoder.layers):
            # Get prompts for this layer and expand to batch size
            layer_prompts = self.prompts[:, i, :, :].expand(n, -1, -1)
            
            # Concatenate prompts with the input
            # Format: [CLS token, prompts, image patches]
            x_with_prompts = torch.cat([
                x[:, :1, :],  # CLS token
                layer_prompts,  # Prompts for this layer
                x[:, 1:, :]   # Image patch embeddings
            ], dim=1)
            
            # Pass through the transformer layer
            x_with_prompts = layer(x_with_prompts)
            
            # Remove prompts from output for next layer
            # Keep only [CLS] and image patches
            x = torch.cat([
                x_with_prompts[:, :1, :],  # CLS token
                x_with_prompts[:, self.num_prompts + 1:, :]  # Image patches
            ], dim=1)
        
        # Apply final layer normalization
        x = self.vit.encoder.ln(x)
        
        # Use only the CLS token for classification
        x = x[:, 0]
        
        # Classification head
        x = self.head(x)
        
        return x
    
    def to(self, device):
        super(VPTDeep, self).to(device)
        self.vit = self.vit.to(device)
        return self


def test(test_loader, model, device):
    model.eval()
    total_loss, correct, n = 0., 0., 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        correct += (y_hat.argmax(dim=1) == y).float().mean().item()
        loss = nn.CrossEntropyLoss()(y_hat, y)
        total_loss += loss.item()
        n += 1
    accuracy = correct / n
    loss = total_loss / n
    return loss, accuracy


def inference(test_loader, model, device, result_path):
    """Generate predicted labels for the test set."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            y_hat = model(x)
            pred = y_hat.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    
    with open(result_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {result_path}")


class Trainer():
    def __init__(self, model, train_loader, val_loader, writer,
                 optimizer, lr, wd, momentum, 
                 scheduler, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.writer = writer
        
        self.model.to(self.device)
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=lr, weight_decay=wd,
                                             momentum=momentum)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=lr, weight_decay=wd)
            
        # 改进的学习率调度
        if scheduler == 'multi_step':
            # 延后学习率下降时机，使用更温和的下降
            self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[80, 110], gamma=0.1)
        elif scheduler == 'cosine':
            # Cosine annealing 学习率
            self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler == 'step':
            self.lr_schedule = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=40, gamma=0.1)
    
    def train_epoch(self):
        self.model.train()
        total_loss, correct, n = 0., 0., 0
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            total_loss += loss.item()
            correct += (y_hat.argmax(dim=1) == y).float().mean().item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            n += 1
        return total_loss / n, correct / n
    
    def val_epoch(self):
        self.model.eval()
        total_loss, correct, n = 0., 0., 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                correct += (y_hat.argmax(dim=1) == y).float().mean().item()
                loss = nn.CrossEntropyLoss()(y_hat, y)
                total_loss += loss.item()
                n += 1
        accuracy = correct / n
        loss = total_loss / n
        return loss, accuracy
    
    def train(self, model_file_name, best_val_acc=-np.inf):
        best_epoch = np.nan
        for epoch in range(self.epochs):
            logging.info(f'Training Epoch {epoch}')
            train_loss, train_acc = self.train_epoch()
            logging.info(f'Validating at Epoch {epoch}')
            val_loss, val_acc = self.val_epoch()
            
            # Log metrics
            self.writer.add_scalar('lr', self.lr_schedule.get_last_lr()[0], epoch)
            self.writer.add_scalar('val_acc', val_acc, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)
            self.writer.add_scalar('train_acc', train_acc, epoch)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_file_name)
                logging.info(f'*** New best model at epoch {epoch}: val_acc = {val_acc:.4f} ***')
            
            self.lr_schedule.step()
        
        return best_val_acc, best_epoch