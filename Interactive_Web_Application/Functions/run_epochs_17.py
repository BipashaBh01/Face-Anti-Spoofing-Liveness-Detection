# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *
from Functions.full_model_pipeline_13 import *
from Functions.loss_functions_14 import *
from Functions.optimiser_and_scheduler_15 import *
from Functions.PAD_metrics_16 import *




#=========================================================
# MAIN FUNCTION FOR RUNNING AN EPOCH
#=========================================================

# Defines a function that runs one epoch (one full pass through the data). 
# Takes the model, data loader, optimizer, loss function, device, optional mixed-precision scaler, and a flag for train vs. validation.
def run_epoch(model, loader, optimizer, criterion, device, scaler=None, is_train=True, epoch=0):
    """
    Run one full epoch (training or validation).

    Args:
        model (torch.nn.Module):
            Multi-stream baseline model (RGB, frequency, optical flow).
        loader (DataLoader):
            Yields batches of (rgb, freq, flow, labels, attack_types, usable_frac).
        optimizer (torch.optim.Optimizer):
            Optimizer used for training.
        criterion (callable):
            Loss function returning (total_loss, component_dict).
        device (torch.device):
            Device to run computations on (CPU/GPU).
        scaler (torch.cuda.amp.GradScaler, optional):
            Gradient scaler for mixed precision training.
        is_train (bool, optional):
            Training mode (True) or validation mode (False).

    Returns:
        tuple:
            - avg_loss (float)
            - metrics (dict): PAD metrics (APCER, BPCER, ACER, Acc, AUC)
            - all_probs (np.array)
            - all_labels (np.array)
            - loss_breakdown (dict): average loss components
    """

    # Set the model's mode. 
    # train() enables dropout and batch-norm updates (for training)
    # eval() disables them (for validation, so results are consistent)
    model.train() if is_train else model.eval()

    # Running sum of the loss across all batches (to average later)
    total_loss = 0.0
    # Empty lists to collect every prediction and true label across the epoch (for computing metrics at the end),
    # and the attack type labels for each sample
    all_probs, all_labels, all_attacks = [], [], []

    # Track the loss components used in the baseline (focal + supcon)
    loss_breakdown = {'focal': 0.0, 'supcon': 0.0}

    # Progress bar: Wrap the loader in a progress bar, labeled "Train" or "Val"
    phase = 'Train' if is_train else 'Val'
    pbar = tqdm(loader, desc=f'{phase} (ep {epoch+1}/{CFG.N_EPOCHS})', leave=False)

    # Choose a context: training tracks gradients (enable_grad); validation doesn't (no_grad, saves memory and speed)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    # Enter that gradient context for the whole loop
    with ctx:
        # Loop over every batch the loader serves
        for batch in pbar:
            # Unpack the 6 items from each batch: the three stream tensors, the labels, the attack types, and the quality signal
            rgb, freq, flow, labels, attack_types, usable_frac = batch

            # Move data to device
            # non_blocking=True allows asynchronous transfer (faster when paired with pin_memory)
            # Labels are converted to float (needed for the loss)
            rgb    = rgb.to(device, non_blocking=True)
            freq   = freq.to(device, non_blocking=True)
            flow   = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            # Before computing new gradients, clear old ones (gradients accumulate by default in PyTorch, so reset each batch). 
            # Only in training.
            if is_train:
                optimizer.zero_grad()

            # -------------------- Forward Pass (Mixed Precision) --------------------
            # Enable mixed precision if a scaler is provided. Inside this block, operations use faster 16-bit math where safe.
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                # Forward pass: run the three streams through the model --> get raw scores, probabilities, and the contrastive embeddings
                logits, probs, embeddings = model(rgb, freq, flow)

                # Compute the loss: the criterion returns the total loss and a breakdown dict (comps = {'focal':…, 'supcon':…})
                loss, comps = criterion(logits, probs, embeddings, labels)

            # -------------------- Backward Pass (Training Only) --------------------
            # Only update weights during training
            if is_train:
                # check if mixed-precision scaler is used
                if scaler:
                    # Scale the loss up before backprop (mixed precision can make tiny gradients vanish)
                    # scaling prevents that, then compute gradients
                    scaler.scale(loss).backward()
                    # Unscale the gradients back to normal size before clipping (so clipping uses true gradient values)
                    scaler.unscale_(optimizer)
                    # Gradient clipping: if gradients are too large, shrink them to a max norm (CFG.GRAD_CLIP). 
                    # Prevents unstable "exploding gradient" updates
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                    # Apply the weight update (the optimizer moves the weights)
                    scaler.step(optimizer)
                    # Adjust the scaler's scaling factor for next time
                    scaler.update()
                else:
                    # The non-mixed-precision path: standard backprop, clip gradients, update weights. Same logic without scaling
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                    optimizer.step()

            # -------------------- Logging per batch --------------------
            # Add this batch's loss to the running total 
            # .item() extracts the number from the tensor
            total_loss += loss.item()

            # Accumulate each loss component (focal, supcon) from this batch
            for k in loss_breakdown:
                loss_breakdown[k] += comps.get(k, 0)

            # Save this batch's predictions and labels. 
            # .detach() removes them from the gradient graph, 
            # .cpu() moves to CPU, .numpy() converts to arrays. 
            # Collected for metrics later.
            all_probs.extend(probs.detach().cpu().float().numpy())
            all_labels.extend(labels.cpu().float().numpy())

            # Collect attack types for this batch.
            # The DataLoader usually batches strings as a tuple/list; tensors are also handled.
            # extend() flattens them so we get exactly one entry per sample.
            if isinstance(attack_types, torch.Tensor):
                all_attacks.extend(attack_types.cpu().tolist())
            else:
                all_attacks.extend(list(attack_types))

            # Show the current batch loss live on the progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    
    # -------------------- Epoch Summary (after all batches) --------------------
    # Number of batches (to average over).
    n = len(loader)

    # Compute the anti-spoofing metrics (APCER, BPCER, ACER, accuracy, AUC) from all collected labels and predictions. 
    # Labels are rounded to ints (in case of float).
    metrics = compute_pad_metrics(
        np.array(all_labels).round().astype(int),
        np.array(all_probs)
    )

    # Average each loss component over all batches
    for k in loss_breakdown:
        loss_breakdown[k] /= n

    # Return: the average loss, the metrics, all predictions, all labels, the loss breakdown, and the attack-type list
    return total_loss / n, metrics, np.array(all_probs), np.array(all_labels), loss_breakdown, all_attacks

# =======================================================================================
# Display training and validation metrics
# =======================================================================================
def plot_training_history(history, title, save_path):
    """
    Plots a 9-panel training history dashboard (loss, ACER, BPCER, APCER
    vs BPCER, AUC, loss components, learning rate, and a summary box) and
    saves it to disk.
 
    Parameters
    ----------
    history : dict
        Must contain: train_loss, val_loss, train_acer, val_acer,
        train_bpcer, val_bpcer, val_apcer, train_auc, val_auc, focal,
        supcon, val_focal, val_supcon, lr — one list per key, one value
        per epoch.
    title : str
        Figure title (e.g. identifies which model/run this is).
    save_path : str or Path
        Where to save the resulting PNG.
    """
    # X-axis values = epoch numbers (1, 2, 3, ...)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(title, fontsize=14, fontweight='bold')
 
    # ---- [0,0] Total Loss (train vs val) ----
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', ms=3, label='Train')
    axes[0, 0].plot(epochs, history['val_loss'],   'r-o', ms=3, label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
 
    # ---- [0,1] ACER % (train vs val) ----
    axes[0, 1].plot(epochs, [a*100 for a in history['train_acer']], 'b-o', ms=3, label='Train')
    axes[0, 1].plot(epochs, [a*100 for a in history['val_acer']],   'r-o', ms=3, label='Val')
    axes[0, 1].set_title('ACER % (lower=better)')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('ACER (%)')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
 
    # ---- [0,2] BPCER % (train vs val) ----
    axes[0, 2].plot(epochs, [a*100 for a in history['train_bpcer']], 'b-o', ms=3, label='Train')
    axes[0, 2].plot(epochs, [a*100 for a in history['val_bpcer']],   'r-o', ms=3, label='Val')
    axes[0, 2].set_title('BPCER % (lower=better)')
    axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('BPCER (%)')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)
 
    # ---- [1,0] APCER vs BPCER % (validation) ----
    axes[1, 0].plot(epochs, [a*100 for a in history['val_apcer']], 'g-o', ms=3, label='APCER (Val)')
    axes[1, 0].plot(epochs, [a*100 for a in history['val_bpcer']], 'm-o', ms=3, label='BPCER (Val)')
    axes[1, 0].set_title('APCER vs BPCER on Validation (lower=better)')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Error rate (%)')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
 
    # ---- [1,1] ROC-AUC (train vs val) ----
    axes[1, 1].plot(epochs, history['train_auc'], 'b-o', ms=3, label='Train')
    axes[1, 1].plot(epochs, history['val_auc'],   'r-o', ms=3, label='Val')
    axes[1, 1].set_title('ROC-AUC (higher=better)')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_ylim([0.5, 1.01])
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
 
    # ---- [1,2] Loss Components: Training ----
    axes[1, 2].plot(epochs, history['focal'],  'b-', lw=2, label='Focal')
    axes[1, 2].plot(epochs, history['supcon'], 'r-', lw=2, label='SupCon')
    axes[1, 2].set_title('Loss Components: Training')
    axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('Component Loss')
    axes[1, 2].legend(); axes[1, 2].grid(alpha=0.3)
 
    # ---- [2,0] Loss Components: Validation ----
    axes[2, 0].plot(epochs, history['val_focal'],  'b-', lw=2, label='Focal')
    axes[2, 0].plot(epochs, history['val_supcon'], 'r-', lw=2, label='SupCon')
    axes[2, 0].set_title('Loss Components: Validation')
    axes[2, 0].set_xlabel('Epoch'); axes[2, 0].set_ylabel('Component Loss')
    axes[2, 0].legend(); axes[2, 0].grid(alpha=0.3)
 
    # ---- [2,1] Learning Rate (log scale) ----
    axes[2, 1].plot(epochs, history['lr'], 'g-', lw=2)
    axes[2, 1].set_title('Learning Rate')
    axes[2, 1].set_xlabel('Epoch'); axes[2, 1].set_ylabel('Learning Rate (log scale)')
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(alpha=0.3)
 
    # ---- [2,2] Summary text box (no axes) ----
    axes[2, 2].axis('off')
    text  = f"Best Val ACER: {min(history['val_acer'])*100:.2f}%\n"
    text += f"Best Val AUC:  {max(history['val_auc']):.4f}\n"
    text += f"Epochs trained: {len(epochs)}"
    axes[2, 2].text(0.1, 0.5, text, transform=axes[2, 2].transAxes,
                    fontsize=14, verticalalignment='center')
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Training history plot saved to {save_path}')

