# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *


#=========================================================
# MAIN CLASS FOR FOCAL LOSS
#=========================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Purpose:
    - Handles class imbalance by down-weighting easy examples
    - Focuses training on hard/misclassified samples

    Parameters:
    - alpha: weight for positive class (balances class importance)
    - gamma: focusing parameter (higher = more focus on hard samples)

    Formula:
    FL = alpha_t * (1 - p_t)^gamma * BCE
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # weight for positive class
        self.gamma = gamma  # focusing strength (hard examples)

    def forward(self, logits, targets):
        """
        Defines how loss is computed during training
        
        Args:
            logits  : raw model outputs (before sigmoid), shape (B,)
            targets : ground truth labels (0 or 1), shape (B,)

        Returns:
            Scalar focal loss
        """

        # Convert targets to float for BCE computation
        targets = targets.float()

        # Compute Binary Cross Entropy loss per sample (no reduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Convert BCE loss to probability of correct classification (p_t)
        # Low loss: high p_t (correct prediction), High loss: low p_t (wrong prediction)
        # p_t = exp(-BCE) gives higher values for correct predictions
        p_t = torch.exp(-bce)

        # Assign class weights
        # If target = 1 -> alpha
        # If target = 0 -> (1 - alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply focal loss formula
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


#=========================================================
# MAIN CLASS FOR SUPCON LOSS
#=========================================================

# -------- Supervised Contrastive Loss -----------------------------------------
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)

    Key Idea:
    - Pull embeddings of SAME class closer
    - Push embeddings of DIFFERENT class apart

    Why useful for FAS (Face Anti-Spoofing):
    - Learns "realness" structure in embedding space
    - More generalisable than just binary classification

    Difference from Cross-Entropy:
    - CE: classify each sample independently
    - SupCon: learn relationships between samples
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature  # scaling factor for similarity (temperature controls how sharp similarity scores are)

    def forward(self, features, labels):
        """
        Args:
            features : (B, D) L2-normalised embeddings
            labels   : (B,) integer labels (0=real, 1=fake)

        Returns:
            Scalar SupCon loss
        """

        B = features.shape[0]   # Get batch size

        # If batch too small, contrastive learning not possible
        if B < 2:    # Need at least 2 samples for comparison
            return torch.tensor(0.0, device=features.device)

        # Compute cosine similarity matrix (B x B)
        # Since features are normalised --> dot product = cosine similarity
        # Each entry = similarity between two samples
        sim = torch.matmul(features, features.T) / self.temperature

        # Reshape labels for pairwise comparison
        labels = labels.view(-1, 1)

        # Positive mask: 1 if same class, else 0
        pos_mask = (labels == labels.T).float()

        # Remove self-comparisons (diagonal)
        pos_mask.fill_diagonal_(0)

        # Negative mask: 1 if different class
        neg_mask = (labels != labels.T).float()

        # Numerical stability trick (log-sum-exp)
        # Subtract max value per row before exponentiation
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_exp = torch.exp(sim - sim_max.detach())

        # Denominator: sum over all non-self similarities
        denom = sim_exp.sum(dim=1, keepdim=True) - sim_exp.diagonal().unsqueeze(1)

        # Compute log-probabilities
        log_prob = sim - sim_max.detach() - torch.log(denom + 1e-8)

        # Count number of positive samples per anchor
        n_pos = pos_mask.sum(dim=1)

        # Compute loss only over positive pairs
        loss = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)

        # Average only valid anchors (with at least 1 positive)
        return loss[n_pos > 0].mean() if (n_pos > 0).any() else torch.tensor(0.0, device=features.device)


#=========================================================
# MAIN CLASS FOR COMPOUND LOSS
#=========================================================

# ------------ Compound loss function (baseline: Focal + SupCon) ---------------
class CompoundFASLoss(nn.Module):
    """
    Combined loss for Face Anti-Spoofing (FAS) — baseline version.

    Components:
    1. Focal Loss  : robust binary classification (handles class imbalance).
    2. SupCon Loss : embedding-structure learning (pulls same-class together,
                     pushes different-class apart).

    Total Loss:
        L = lambda_focal * Focal + lambda_supcon * SupCon

    NOTE: The depth (geometry) term has been removed for the baseline. It can be
    re-added later as a fine-tuning enhancement (needs MiDaS pseudo-labels and a
    depth head on the model).
    """

    def __init__(self, lambda_focal=CFG.LAMBDA_FOCAL, lambda_supcon=CFG.LAMBDA_SUPCON):
        super().__init__()
        # Individual loss components
        self.focal  = FocalLoss(alpha=0.75, gamma=2.0)
        self.supcon = SupervisedContrastiveLoss(temperature=0.07)
        # Loss weights (hyperparameters)
        self.l_focal  = lambda_focal
        self.l_supcon = lambda_supcon

    def forward(self, logits, probs, embeddings, labels):
        """
        Args:
            logits     : (B,)    raw model outputs (pre-sigmoid)
            probs      : (B,)    sigmoid(logits)  [unused here; kept for API symmetry]
            embeddings : (B, D)  L2-normalised embeddings (the model's `proj`)
            labels     : (B,)    ground-truth labels (0=real, 1=fake)

        Returns:
            total (scalar tensor), and a dict of component values for logging.
        """
        # --- Classification loss ---
        loss_focal = self.focal(logits, labels)

        # --- Representation-learning loss ---
        loss_supcon = self.supcon(embeddings, labels)

        # --- Total weighted loss ---
        total = self.l_focal * loss_focal + self.l_supcon * loss_supcon

        # Return total + components for logging
        return total, {
            'focal':  loss_focal.item(),
            'supcon': loss_supcon.item(),
        }