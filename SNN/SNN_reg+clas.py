import snntorch as snn
from snntorch import surrogate
from pathlib import Path
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: None
import numpy as np
import torch
import dataset as ds
import SNN_func_modified as snnfn
import torch.nn as nn

print('empieza')


def save_figure(fig, output_path, **savefig_kwargs):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, **savefig_kwargs)

batch_size   = 50
num_epochs   = 100
multiplicity = 4

labels_map = {
    0: "proton",
    1: "kaon",
    2: "pion",
    3: "other",
}
nClasses = len(labels_map)
population = 20
REG_TARGETS = 7

_leaky_params = {
    "beta"            : 0.5,
    "learn_beta"      : True,
    "threshold"       : 1.0,
    "learn_threshold" : True,
    "spike_grad"      : surrogate.atan(),
}


net_desc_reg = {
    "layers"    : [multiplicity * 100, 300, 200, REG_TARGETS * population],
    "timesteps" : 100,
    "neuron_params" : {
        1: [snn.Leaky, _leaky_params],
        2: [snn.Leaky, _leaky_params],
        3: [snn.Leaky, _leaky_params],
    },
    "output"    : "spike",
}



def spikegen_multi(data, multiplicity=4):
    data       = torch.where(data > 0, data, torch.zeros_like(data))
    og_shape   = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity * og_shape[2],device=data.device,)
    for i in range(multiplicity):
        condition = data > np.power(10, i + 2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity * sensor_idx + i] = 1
    return spike_data


def comp_accuracy(log_probs, targets):
    predicted = log_probs.argmax(dim=1)
    return (predicted == targets).to(torch.float32)

class KendallLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

        self.loss_cls = nn.NLLLoss()
        self.loss_reg = nn.MSELoss()

    def forward(self, outputs, targets):
        log_probs, reg_pred = outputs
        target_cls, target_reg = targets

        l_cls = self.loss_cls(log_probs, target_cls)
        l_reg = self.loss_reg(reg_pred, target_reg.float())

        prec_cls = torch.exp(-self.log_var_cls)
        prec_reg = torch.exp(-self.log_var_reg)

        loss = (
            prec_cls       * l_cls + self.log_var_cls
            + 0.5 * prec_reg * l_reg + 0.5 * self.log_var_reg
        )

        return loss, l_cls.detach(), l_reg.detach()


class SpikingClassificationHead(nn.Module):
    def __init__(self, layers, neuron_params, out_population=1):
        super().__init__()
        self.linears = nn.ModuleList(nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1))
        self.neurons = nn.ModuleList(snn.Leaky(**neuron_params) for _ in range(len(layers) - 1))

        self.out_population = out_population
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, spiketrain, reg_feats):
        mem = []
        for neuron in self.neurons:
            res = neuron.reset_mem()
            mem.append(list(res) if isinstance(res, tuple) else [res])

        spk_rec = []
        for t in range(spiketrain.shape[0]):
            spk = torch.cat([spiketrain[t], reg_feats], dim=1)
            for i, (linear, neuron) in enumerate(zip(self.linears, self.neurons)):
                cur = linear(spk)
                spk, *(mem[i]) = neuron(cur, *(mem[i]))
            spk_rec.append(spk)

        spike_counts = torch.stack(spk_rec, dim=0).sum(dim=0)
        spike_counts = spike_counts.view(
            spike_counts.shape[0], -1, self.out_population
        ).mean(dim=2)
        return self.log_softmax(spike_counts)


class Hybrid_Net(nn.Module):
    def __init__(
        self,
        snn_reg    : nn.Module,
        n_reg      : int,
        n_classes  : int,
        population  : int,
        spikegen_fn,
    ):
        super().__init__()
        self.snn_reg    = snn_reg
        self.n_reg      = n_reg
        self.population = population
        self.spikegen_fn = spikegen_fn

        head_in = multiplicity * 100 + n_reg
        self.head_cls = SpikingClassificationHead(
            layers=[head_in, 120, 100, multiplicity * n_classes],
            neuron_params=_leaky_params,
            out_population=multiplicity,
        )

    def forward(self, data):
        spikes_reg = self.snn_reg(data)

        reg_out = spikes_reg.sum(dim=0).view(
            -1, self.n_reg, self.population
        ).mean(dim=2)

        spiketrain = self.spikegen_fn(data)

        log_probs = self.head_cls(spiketrain, reg_out)

        return log_probs, reg_out



dataset = ds.build_dataset(
    path         = "./Data/PrimaryOnly/Uniform",
    max_files    = 10000,
    primary_only = True,
    target       = ["particle", "energy", "centroid", "dispersion"],
)
train_loader, test_loader, val_loader = ds.build_loaders(
    dataset, split=(0.7, 0.15), batch_size=batch_size, shuffle=True
)


snn_reg = snnfn.Spiking_Net(net_desc_reg, spikegen_multi)

modelo_completo = Hybrid_Net(
    snn_reg     = snn_reg,
    n_reg       = REG_TARGETS,
    n_classes   = nClasses,
    population  = population,
    spikegen_fn = spikegen_multi,
)

kendall_loss = KendallLoss()

all_params = list(modelo_completo.parameters()) + list(kendall_loss.parameters())
optimizer  = torch.optim.Adam(all_params, lr=5e-3, betas=(0.9, 0.999), weight_decay=0)
scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

device = snnfn.device
modelo_completo.to(device)
kendall_loss.to(device)

history = {
    "train_loss" : [], "val_loss"  : [],
    "val_acc"    : [], "l_cls"     : [],
    "l_reg"      : [], "sigma_cls" : [], "sigma_reg" : [],
}


def target_to_2d_tensor(value):
    if torch.is_tensor(value):
        value = value.float()
        return value.unsqueeze(1) if value.dim() == 1 else value

    if isinstance(value, (list, tuple)):
        parts = [torch.as_tensor(v).float() for v in value]
        return torch.stack(parts, dim=1)

    value = torch.as_tensor(value).float()
    return value.unsqueeze(1) if value.dim() == 1 else value


def build_reg_target(targets):
    energy = target_to_2d_tensor(targets[1])
    energy = torch.log10(torch.clamp(energy, min=1e-12))
    centroid = target_to_2d_tensor(targets[2])
    dispersion = target_to_2d_tensor(targets[3])
    return torch.cat([energy, centroid, dispersion], dim=1).float()


def loss_scales(loss_module):
    sigma_cls = torch.exp(0.5 * loss_module.log_var_cls).item()
    sigma_reg = torch.exp(0.5 * loss_module.log_var_reg).item()
    return sigma_cls, sigma_reg


def run_epoch(loader, train=True):
    modelo_completo.train(train)
    total_loss = total_cls = total_reg = total_acc = 0.0
    n_samples  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            data_batch = batch[0].to(device)
            targets    = batch[1]
            target_cls = torch.as_tensor(targets[0]).long().to(device)
            if target_cls.min().item() < 0 or target_cls.max().item() >= nClasses:
                raise ValueError(
                    f"target_cls fuera de rango para nClasses={nClasses}: "
                    f"min={target_cls.min().item()}, max={target_cls.max().item()}"
                )

            target_reg = build_reg_target(targets).to(device)

            outputs               = modelo_completo(data_batch)
            loss, l_cls, l_reg    = kendall_loss(outputs, (target_cls, target_reg))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            log_probs, _ = outputs
            correct      = comp_accuracy(log_probs, target_cls)
            batch_size_actual = target_cls.numel()

            total_loss += loss.item() * batch_size_actual
            total_cls  += l_cls.item() * batch_size_actual
            total_reg  += l_reg.item() * batch_size_actual
            total_acc  += correct.sum().item()
            n_samples  += batch_size_actual

    return (
        total_loss / n_samples,
        total_cls  / n_samples,
        total_reg  / n_samples,
        total_acc  / n_samples,
    )

print(f"\n{'Época':>5}  {'Train L':>9}  {'Val L':>9}  {'Val Acc':>8}  "
      f"{'L_cls':>8}  {'L_reg':>10}  {'σ_cls':>7}  {'σ_reg':>7}  {'Best':>4}")
print("-" * 88)

best_val_acc = -float("inf")
best_val_cls = float("inf")
best_epoch = 0
last_metrics = {}
best_model_state = deepcopy(modelo_completo.state_dict())
best_loss_state  = deepcopy(kendall_loss.state_dict())

for epoch in range(1, num_epochs + 1):
    tr_loss, tr_cls, tr_reg, _          = run_epoch(train_loader, train=True)
    va_loss, va_cls, va_reg, va_acc     = run_epoch(val_loader,   train=False)
    scheduler.step()

    sig_cls, sig_reg = loss_scales(kendall_loss)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    history["l_cls"].append(va_cls)
    history["l_reg"].append(va_reg)
    history["sigma_cls"].append(sig_cls)
    history["sigma_reg"].append(sig_reg)

    last_metrics = {
        "train_loss": tr_loss,
        "val_loss": va_loss,
        "val_acc": va_acc,
        "val_cls_loss": va_cls,
        "val_reg_loss": va_reg,
        "sigma_cls": sig_cls,
        "sigma_reg": sig_reg,
    }

    is_best = va_acc > best_val_acc or (
        np.isclose(va_acc, best_val_acc) and va_cls < best_val_cls
    )
    if is_best:
        best_val_acc = va_acc
        best_val_cls = va_cls
        best_epoch = epoch
        best_model_state = deepcopy(modelo_completo.state_dict())
        best_loss_state  = deepcopy(kendall_loss.state_dict())

    print(f"{epoch:>5}  {tr_loss:>9.4f}  {va_loss:>9.4f}  {va_acc:>8.4f}  "
          f"{va_cls:>8.4f}  {va_reg:>10.2f}  "
          f"{sig_cls:>7.4f}  {sig_reg:>7.4f}  "
          f"{'*' if is_best else '':>4}")


modelo_completo.load_state_dict(best_model_state)
kendall_loss.load_state_dict(best_loss_state)
print(
    f"\nCargados los mejores pesos: epoch={best_epoch}, "
    f"val_acc={best_val_acc:.4f}"
)


te_loss, te_cls, te_reg, te_acc = run_epoch(test_loader, train=False)
print(f"\nTest → loss={te_loss:.4f}  acc={te_acc:.4f}  "
      f"l_cls={te_cls:.4f}  l_reg={te_reg:.2f}")
test_sig_cls, test_sig_reg = loss_scales(kendall_loss)
print(f"       σ_cls={test_sig_cls:.4f}  σ_reg={test_sig_reg:.4f}")

epochs_range = range(1, num_epochs + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(epochs_range, history["train_loss"], label="Train")
axes[0, 0].plot(epochs_range, history["val_loss"],   label="Val")
axes[0, 0].set_title("Pérdida total (Kendall)"); axes[0, 0].set_xlabel("Época")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(epochs_range, history["val_acc"])
axes[0, 1].set_title("Exactitud clasificación (val)"); axes[0, 1].set_xlabel("Época")
axes[0, 1].grid(alpha=0.3)

ax2 = axes[1, 0]
ax2b = ax2.twinx()
ax2.plot(epochs_range,  history["l_cls"], color="tab:blue",   label="L_cls (NLL)")
ax2b.plot(epochs_range, history["l_reg"], color="tab:orange", label="L_reg (MSE)", linestyle="--")
ax2.set_ylabel("L_cls (NLL)", color="tab:blue")
ax2b.set_ylabel("L_reg (MSE)", color="tab:orange")
ax2.set_title("Pérdidas individuales (val)"); ax2.set_xlabel("Época")
ax2.grid(alpha=0.3)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2)

axes[1, 1].plot(epochs_range, history["sigma_cls"], label="σ_cls (clasificación)")
axes[1, 1].plot(epochs_range, history["sigma_reg"], label="σ_reg (regresión)")
axes[1, 1].set_title("Incertidumbres Kendall aprendidas"); axes[1, 1].set_xlabel("Época")
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

fig.tight_layout()
save_figure(fig, "training_curves_reg+clas_nocls_100_epocas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

class_names = list(labels_map.values())

modelo_completo.eval()
all_log_probs = []
all_targets   = []
all_reg_pred  = []
all_reg_true  = []
with torch.no_grad():
    for batch in test_loader:
        data_batch = batch[0].to(device)
        target_cls = torch.as_tensor(batch[1][0]).long().to(device)
        target_reg = build_reg_target(batch[1]).to(device)

        log_probs, reg_pred = modelo_completo(data_batch)
        all_log_probs.append(log_probs.cpu())
        all_targets.append(target_cls.view(-1).cpu())
        all_reg_pred.append(reg_pred.cpu())
        all_reg_true.append(target_reg.cpu())

all_log_probs = torch.cat(all_log_probs, dim=0)
all_targets   = torch.cat(all_targets,   dim=0).numpy()
all_probs     = torch.exp(all_log_probs).numpy()
all_preds     = all_log_probs.argmax(dim=1).numpy()
all_reg_pred  = torch.cat(all_reg_pred,  dim=0).numpy()
all_reg_true  = torch.cat(all_reg_true,  dim=0).numpy()

cm = np.zeros((nClasses, nClasses), dtype=int)
for t, p in zip(all_targets, all_preds):
    cm[t, p] += 1
np.savetxt("confusion_matrix_reg+clas_nocls.txt", cm, fmt="%d")

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
fig_cm.colorbar(im, ax=ax_cm)
tick_marks = np.arange(nClasses)
ax_cm.set_xticks(tick_marks); ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
ax_cm.set_yticks(tick_marks); ax_cm.set_yticklabels(class_names)
ax_cm.set_xlabel("Predicción"); ax_cm.set_ylabel("Valor real")
ax_cm.set_title("Matriz de confusión")
for i in range(nClasses):
    for j in range(nClasses):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if im.norm(cm[i, j]) > 0.5 else "black")
fig_cm.tight_layout()
save_figure(fig_cm, "confusion_matrix_SNN_reg+clas_nocls_100_epocas.png", dpi=300, bbox_inches="tight")
plt.close(fig_cm)

fig, axes = plt.subplots(1, nClasses, figsize=(5 * nClasses, 5), sharex=True, sharey=True)
for true_cls in range(nClasses):
    ax   = axes[true_cls]
    mask = all_targets == true_cls
    print(f"  {class_names[true_cls]}: {mask.sum()} eventos en test")
    for pred_cls in range(nClasses):
        ax.hist(all_probs[mask, pred_cls], bins=100, range=(0, 1),
                density=True, histtype="step", linewidth=2,
                label=f"P({class_names[pred_cls]})")
    ax.set_title(f"Eventos reales = {class_names[true_cls]}")
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Densidad")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.suptitle("Distribución de probabilidades predichas condicionada a la clase real", y=0.98)
fig.tight_layout()
save_figure(fig, "probabilidades_reg+clas_nocls_100_epocas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

reg_names = [
    r"$\log_{10}(E/\mathrm{MeV})$",
    "Centroide X [celdas]", "Centroide Y [celdas]", "Centroide Z [celdas]",
    r"$\sigma^2_x$ [celdas$^2$]", r"$\sigma^2_y$ [celdas$^2$]", r"$\sigma^2_z$ [celdas$^2$]",
]


residual_stats = []
for i in range(REG_TARGETS):
    residuos = all_reg_pred[:, i] - all_reg_true[:, i]
    mu    = residuos.mean()
    sigma = residuos.std()
    eps_abs = np.abs(residuos).mean()
    seguro  = np.abs(all_reg_true[:, i]) > 1e-9   
    eps_rel = np.abs(residuos[seguro] / all_reg_true[seguro, i]).mean() * 100
    residual_stats.append((mu, sigma))
    print(f"{reg_names[i]:<16} {mu:>10.4f} {sigma:>10.4f} {eps_abs:>10.4f} {eps_rel:>11.2f}")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i in range(REG_TARGETS):
    residuos = all_reg_pred[:, i] - all_reg_true[:, i]
    mu, sigma = residual_stats[i]
    axes[i].hist(residuos, bins=80, density=True,
                 color="steelblue", edgecolor="none", alpha=0.8)
    axes[i].axvline(0, color="crimson", linewidth=1.5, linestyle="--")
    axes[i].set_title(f"Residuo: {reg_names[i]}")
    axes[i].text(0.97, 0.97, f"$\\mu = {mu:.3f}$\n$\\sigma = {sigma:.3f}$",
                 transform=axes[i].transAxes, ha="right", va="top", fontsize=11,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    axes[i].set_xlabel("Pred − Real"); axes[i].set_ylabel("Densidad")
    axes[i].grid(alpha=0.3)
axes[-1].set_visible(False)

fig.suptitle("Distribución de residuos de regresión (test)", y=0.98)
fig.tight_layout()
save_figure(fig, "residuos_regresion_reg+clas_nocls_100_epocas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("terminado")
