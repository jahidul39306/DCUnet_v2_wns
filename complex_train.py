import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio

from complex_model import DeepComplexUNet
from complex_data_prep import get_dataloaders

# --- TARGET CLUSTER CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "./complex_checkpoints"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_BASE = os.path.join(BASE_DIR, "dataset")
CLEAN_DIR = os.path.join(DATA_BASE, "clean")
NOISY_DIR = os.path.join(DATA_BASE, "noisy")

BATCH_SIZE = 16
NUM_EPOCHS = 300 
LEARNING_RATE = 0.0001
# LEARNING_RATE = 5e-5 # Lowered for final precision fine-tuning
# loss function from the original paper
class wSDRLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super(wSDRLoss, self).__init__()
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop_length, normalized=True
        )
        self.istft.to(DEVICE)
    # when call criterion(a,b,c,d,e,f) it will call this function

    # this recieve 6 tensors the real imag part of the mix, clean and predicted spectrograms
    def forward(self, noisy_real, noisy_imag, clean_real, clean_imag, pred_real, pred_imag):    
        noisy_complex = torch.complex(noisy_real, noisy_imag) 
        clean_complex = torch.complex(clean_real, clean_imag) 
        pred_complex = torch.complex(pred_real, pred_imag) 
        
        noisy_wav = self.istft(noisy_complex)
        clean_wav = self.istft(clean_complex)
        pred_wav = self.istft(pred_complex)
        
        noise_wav = noisy_wav - clean_wav
        pred_noise_wav = noisy_wav - pred_wav # pred_noise_wav IS NEARLY ZERO if the model removed the noise ; no noise it left
        

        clean_wav = clean_wav.flatten(1)
        pred_wav = pred_wav.flatten(1)
        noise_wav = noise_wav.flatten(1)
        pred_noise_wav = pred_noise_wav.flatten(1)
        
        eps = 1e-8 # tiny number used as a safelty buffer , add to denominator so that we can avoid dividing by zero and crash
        
        def sdr(target, pred):
            #  NUMERICAL STABILITY SHIELD
            num = torch.sum(target**2, dim=1) + eps
            den = torch.sum((target - pred)**2, dim=1) + eps
            # Clamp ratio to prevent log10(0) or log10(inf)
            ratio = torch.clamp(num / den, min=1e-7, max=1e7)
            return 10 * torch.log10(ratio)
            
        s_target = sdr(clean_wav, pred_wav) # how well speech is reconstructed
        n_target = sdr(noise_wav, pred_noise_wav) # how well is noise seperated
        
        clean_energy = torch.sum(clean_wav**2, dim=1) + eps
        noise_energy = torch.sum(noise_wav**2, dim=1) + eps
        # alpha is weight , if clean energy is high alpha is high focus on speech; if alpha is low noise is high concetrate on removing noise
        alpha = clean_energy / (clean_energy + noise_energy + eps)
        
        loss = - (alpha * s_target + (1 - alpha) * n_target)
        
        # FINAL SHIELD: Filter out any NaNs that managed to break through
        loss = torch.nan_to_num(loss, nan=0.0)
        if loss.numel() == 0:
            return None
            
        return torch.mean(loss)

def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train() # tell pytorch we are in training mode now ;turn on batchnorm and dropout(well we are not using dropout)
    running_loss = 0.0 # initialise the running loss ; this is used to track average loss over the epoch
    loop = tqdm(dataloader, total=len(dataloader), leave=False)

    for mix_real, mix_imag, clean_real, clean_imag in loop:
        
        mix_real, mix_imag = mix_real.to(DEVICE), mix_imag.to(DEVICE)
        clean_real, clean_imag = clean_real.to(DEVICE), clean_imag.to(DEVICE)
        
        mask_real, mask_imag = model(mix_real, mix_imag)
        pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
        pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real
        
        loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
        
        # If loss is still NaN, skip this batch entirely
        if loss is None:
            continue
            
        optimizer.zero_grad() # clean the gradient from the previous iteration
        loss.backward() # calculate the gradient of the loss with respect to model params
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step() # chnage the brain weights based on the gradients; note this is not going to be in the validate_one_epoch 
        
        running_loss += loss.item() # add the loss to the running loss
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
        
    return running_loss / (len(dataloader) + 1e-8) # eps to prevent div-by-zero

# (Rest of validation and main remain standard)
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad(): # tell pytorch do not calculate gradient since we are checking performance here; not tracking;note : nograd makes validation faster and use half gpu memory
        for mix_real, mix_imag, clean_real, clean_imag in dataloader:
            mix_real, mix_imag = mix_real.to(DEVICE), mix_imag.to(DEVICE)
            clean_real, clean_imag = clean_real.to(DEVICE), clean_imag.to(DEVICE)
            mask_real, mask_imag = model(mix_real, mix_imag)
            pred_clean_real = mask_real * mix_real - mask_imag * mix_imag
            pred_clean_imag = mask_real * mix_imag + mask_imag * mix_real
            loss = criterion(mix_real, mix_imag, clean_real, clean_imag, pred_clean_real, pred_clean_imag)
            
            if loss is None: 
                continue
            running_loss += loss.item()
            
    return running_loss / (len(dataloader) + 1e-8)

def save_checkpoint(model, optimizer, epoch, best_val_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, filename)

def main():
    print(f"TRAINING: Using full dataset from: {DATA_BASE}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = None
    if os.path.exists(SAVE_DIR):
        # 1. Check for the absolute latest (safest for Slurm timeouts)
        latest_path = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
        else:
            # 2. Fallback to numbered backups
            checkpoints = [f for f in os.listdir(SAVE_DIR) if f.startswith('dcunet_epoch_') and f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(SAVE_DIR, checkpoints[-1])
    
    model = DeepComplexUNet(n_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = wSDRLoss()
    

    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE

    train_loader, val_loader = get_dataloaders(CLEAN_DIR, NOISY_DIR, batch_size=BATCH_SIZE)
    
    log_file = os.path.join(SAVE_DIR, "training_log.csv")
    if not os.path.exists(log_file) and start_epoch == 0:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_wSDR_Loss", "Val_wSDR_Loss"])
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Current Learning Rate: {current_lr:.2e}")
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] wSDR Train: {avg_loss:.4f} | Val: {val_loss:.4f}")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, val_loss])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_val_loss, best_path)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
            
        # 1. Save "latest" for every single epoch (Safety snapshot)
        latest_save_path = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
        save_checkpoint(model, optimizer, epoch, val_loss, latest_save_path)
        
if __name__ == "__main__":
    main()
