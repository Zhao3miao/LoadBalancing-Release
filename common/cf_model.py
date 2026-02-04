import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BASE_STATIONS = 3
NUM_DEVICES = 50
RSRP_DIM = NUM_BASE_STATIONS * NUM_DEVICES
device = "cuda" if torch.cuda.is_available() else "cpu"


def flatten_rsrp_tensor(t):
    """t: (..., bs, dev) -> (..., bs*dev)"""
    return t.reshape(*t.shape[:-2], -1)


def unflatten_rsrp_tensor(v, bs=NUM_BASE_STATIONS, dev=NUM_DEVICES):
    return v.reshape(*v.shape[:-1], bs, dev)


def gaussian_nll(x, mu, logvar, reduce="mean"):
    """
    Negative log-likelihood for diagonal Gaussian.
    x, mu, logvar shape: (B, D)
    logvar is natural log of variance.
    """
    var = torch.exp(logvar)
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (x - mu) ** 2 / var)
    if reduce == "mean":
        return nll.mean()
    elif reduce == "sum":
        return nll.sum()
    else:
        return nll


class GlobalEncoder(nn.Module):
    """
    Encodes the global context (motion pattern/intent) from the trajectory.
    Input: (B, seq_len, RSRP_DIM)
    Output: z_global_mean, z_global_logvar (B, z_global_dim)
    """

    def __init__(self, z_dim=32, hidden_dim=128):
        super(GlobalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=RSRP_DIM,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # Attention mechanism to aggregate temporal features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, z_dim)

    def forward(self, traj):
        # traj: (B, seq_len, RSRP_DIM)
        if traj.dim() > 3:
            traj = flatten_rsrp_tensor(traj)

        out, _ = self.gru(traj)  # (B, T, 2*H)

        # Attention pooling
        attn_weights = F.softmax(self.attention(out), dim=1)  # (B, T, 1)
        context = torch.sum(out * attn_weights, dim=1)  # (B, 2*H)

        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


class LocalEncoder(nn.Module):
    """
    Encodes the local state (current position/status) from a single frame.
    Input: (B, RSRP_DIM)
    Output: z_local_mean, z_local_logvar (B, z_local_dim)
    """

    def __init__(self, z_dim=32, hidden_dim=128):
        super(LocalEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(RSRP_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = flatten_rsrp_tensor(x)
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


class CausalTransition(nn.Module):
    """
    Predicts next local state given current local state and global context.
    z_local_{t+1} ~ P(z_local_{t+1} | z_local_t, z_global)
    """

    def __init__(self, z_local_dim, z_global_dim, hidden_dim=256):
        super(CausalTransition, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_local_dim + z_global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_local_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_local_dim)

        # Residual connection weight
        self.res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_local, z_global):
        # z_local: (B, z_local_dim)
        # z_global: (B, z_global_dim)
        inp = torch.cat([z_local, z_global], dim=-1)
        h = self.net(inp)

        delta_mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)

        # Residual prediction: next = curr + delta
        mu = z_local + self.res_weight * delta_mu
        return mu, logvar


class Decoder(nn.Module):
    """
    Decodes local state back to observation.
    x_t ~ P(x_t | z_local_t)
    """

    def __init__(self, z_local_dim, hidden_dim=128):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, RSRP_DIM)
        self.fc_logvar = nn.Linear(hidden_dim, RSRP_DIM)

    def forward(self, z_local):
        h = self.net(z_local)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


class CounterfactualModel(nn.Module):
    """
    Disentangled Causal World Model.
    Maintains the same interface as the previous model for compatibility.
    """

    def __init__(self, traj_len=20, z_dim=64, hidden_dim=256):
        super(CounterfactualModel, self).__init__()
        # We split z_dim into global and local parts
        self.z_global_dim = z_dim
        self.z_local_dim = z_dim

        self.global_enc = GlobalEncoder(z_dim=self.z_global_dim, hidden_dim=hidden_dim)
        self.local_enc = LocalEncoder(z_dim=self.z_local_dim, hidden_dim=hidden_dim)
        self.transition = CausalTransition(
            self.z_local_dim, self.z_global_dim, hidden_dim
        )
        self.decoder = Decoder(self.z_local_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, traj, current_rsrp):
        """
        Training forward pass.
        traj: (B, T, D) - used to extract global context
        current_rsrp: (B, D) - current state
        Returns: next_mean, next_logvar (in observation space), info dict
        """
        # 1. Extract Global Context (Intent)
        z_g_mu, z_g_logvar = self.global_enc(traj)
        z_g = self.reparameterize(z_g_mu, z_g_logvar)

        # 2. Extract Local State (Position)
        z_l_mu, z_l_logvar = self.local_enc(current_rsrp)
        z_l = self.reparameterize(z_l_mu, z_l_logvar)

        # 3. Predict Next Local State
        z_l_next_mu, z_l_next_logvar = self.transition(z_l, z_g)
        z_l_next = self.reparameterize(z_l_next_mu, z_l_next_logvar)

        # 4. Decode to Observation
        next_obs_mean, next_obs_logvar = self.decoder(z_l_next)

        info = {
            "z_g_mu": z_g_mu,
            "z_g_logvar": z_g_logvar,
            "z_g": z_g,
            "z_l_mu": z_l_mu,
            "z_l_logvar": z_l_logvar,
            "z_l": z_l,
            "z_l_next_mu": z_l_next_mu,
            "z_l_next_logvar": z_l_next_logvar,
            "z_l_next": z_l_next,
        }
        return next_obs_mean, next_obs_logvar, info

    def encode_state(self, x):
        # Helper for consistency loss if needed
        mu, _ = self.local_enc(x)
        return mu

    def counterfactual_sample(self, traj, current_rsrp, n_samples=5, noise_std=0.00):
        """
        Generate counterfactual samples.
        We fix Z_global (from traj) and sample Z_local transitions.
        """
        # 1. Extract Global Context (Deterministic or Sampled)
        z_g_mu, z_g_logvar = self.global_enc(traj)
        # Use mean for stable counterfactuals, or sample for diversity
        z_g = z_g_mu

        # 2. Extract Local State
        z_l_mu, z_l_logvar = self.local_enc(current_rsrp)
        z_l = z_l_mu  # Start from deterministic current state

        samples = []
        for _ in range(n_samples):
            # Sample transition noise
            # In this model, transition is probabilistic: P(z_next | z_curr, z_global)
            z_l_next_mu, z_l_next_logvar = self.transition(z_l, z_g)

            # Sample from the transition distribution
            z_l_next = self.reparameterize(z_l_next_mu, z_l_next_logvar)

            if noise_std > 0:
                z_l_next = z_l_next + torch.randn_like(z_l_next) * noise_std

            # Decode
            pred_obs, _ = self.decoder(z_l_next)
            samples.append(pred_obs.unsqueeze(1))

        return torch.cat(samples, dim=1)  # (B, n_samples, D)
