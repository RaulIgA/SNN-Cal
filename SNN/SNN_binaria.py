import snntorch as snn
from snntorch import surrogate
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: None
import numpy as np
import torch
from typing import Callable
import visualize
import dataset as ds
import SNN_func_modified as snnfn
from torch.utils.data import DataLoader
import torch.nn as nn
from snntorch import spikeplot as splt
import torchvision as tv
from sklearn.metrics import roc_curve, auc


print('empieza')
nCublets = 1000
nSensors = 100
max_t = 20
dt = 0.2
timesteps = int(max_t/dt)
batch_size = 50
num_epochs = 100
idx = 100


PAIR = (1, 2)
PAIR_NAMES = ("kaon", "pion")

labels_map = {0: PAIR_NAMES[0], 1: PAIR_NAMES[1]}
nClasses = len(labels_map)


population = 20

net_desc = {
    "layers" : [400, 120 ,100 , nClasses*population],
    "timesteps": 100,
    "neuron_params" : {

                1: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],
                    
                2: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],

                3: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],

                4: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],

                },
    }



net_desc_spikefreq = deepcopy(net_desc)
net_desc_spikefreq["output"] = "spike"



def spikegen_multi(data, multiplicity=4):
    data = torch.where(data > 0, data, torch.zeros_like(data))
    og_shape = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity*og_shape[2], device = data.device)
    for i in range(multiplicity):
        condition = data > np.power(10, i+2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity*sensor_idx+i] = 1

    return spike_data


def predict_spikefreq(output):
    return output

def comp_accuracy(output, targets, *args, **kwargs):
    predicted = output.argmax(dim=1)
    correct = (predicted == targets).to(torch.float32)

    return correct


class Spiking_Classifier(nn.Module):
    def __init__(self, snn_network, n_classes, population):
        super().__init__()

        self.snn = snn_network
        self.n_classes = n_classes
        self.population = population
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):

        spk_rec = self.snn(data)
        rate = spk_rec.mean(dim=0)
        rate = rate.view(rate.shape[0], self.n_classes, self.population).mean(dim=2)

        return self.softmax(rate)


class BinaryDataset(torch.utils.data.Dataset):
    """Filtra el dataset a las dos clases de PAIR y las reetiqueta como 0/1."""

    def __init__(self, base, class_a, class_b):
        self.base = base
        self.mapping = {class_a: 0, class_b: 1}
        self.indices = [i for i in range(len(base))
                        if int(base[i][1]) in self.mapping]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        data, label = self.base[self.indices[i]]
        return data, self.mapping[int(label)]


dataset_base = ds.build_dataset( path="./Data/PrimaryOnly/Uniform", max_files=10000, primary_only = True, target="particle")
dataset = BinaryDataset(dataset_base, *PAIR)
print(f"Eventos binarios ({PAIR_NAMES[0]} vs {PAIR_NAMES[1]}): {len(dataset)} de {len(dataset_base)}")

train_loader, test_loader, val_loader = ds.build_loaders(dataset, split=(0.7, 0.15), batch_size=batch_size, shuffle=True)

net_Epos_spk = snnfn.Spiking_Net(net_desc_spikefreq, spikegen_multi)
modelo_completo = Spiking_Classifier(net_Epos_spk, nClasses, population)

Pred_Epos_spk = snnfn.Predictor(predict_spikefreq, comp_accuracy)
loss_Epos = nn.NLLLoss()
opt_Epos_spk = torch.optim.Adam(modelo_completo.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=0)
sche_Epos_spk = torch.optim.lr_scheduler.ExponentialLR(opt_Epos_spk, gamma=0.9)
train_Epos_spk = snnfn.Trainer(modelo_completo, loss_Epos, opt_Epos_spk, Pred_Epos_spk,
                    train_loader, val_loader, test_loader, task = "Accuracy")

train_Epos_spk.train(num_epochs)
train_Epos_spk.predict.accuracy_fn = lambda p, t: comp_accuracy(p, t)
train_Epos_spk.test("test")

train_epochs = sorted(train_Epos_spk.loss_hist["train"].keys())
train_loss_curve = [np.mean(train_Epos_spk.loss_hist["train"][e]) for e in train_epochs]
val_epochs = sorted(train_Epos_spk.loss_hist["validation"].keys())
val_loss_curve = [train_Epos_spk.loss_hist["validation"][e] for e in val_epochs]
val_acc_curve = [train_Epos_spk.acc_hist["validation"][e] for e in val_epochs]

fig_tc, axes_tc = plt.subplots(1, 2, figsize=(12, 5))

axes_tc[0].plot([e + 1 for e in train_epochs], train_loss_curve, label="Train")
axes_tc[0].plot(val_epochs, val_loss_curve, label="Val")
axes_tc[0].set_title("Pérdida (NLL)"); axes_tc[0].set_xlabel("Época")
axes_tc[0].legend(); axes_tc[0].grid(alpha=0.3)

axes_tc[1].plot(val_epochs, val_acc_curve)
axes_tc[1].set_title("Exactitud clasificación (val)"); axes_tc[1].set_xlabel("Época")
axes_tc[1].grid(alpha=0.3)

fig_tc.tight_layout()
fig_tc.savefig("training_curves_SNN_simple_binaria_100_epocas.png", dpi=300, bbox_inches="tight")
plt.close(fig_tc)

cm_metric = train_Epos_spk.ConfusionMatrix(num_classes=nClasses)
cm = cm_metric.compute().cpu().numpy()
np.savetxt("confusion_matrix_spiking_binaria.txt", cm, fmt="%d")

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
fig_cm.colorbar(im, ax=ax_cm)

ax_cm.set_xlabel("Predicción")
ax_cm.set_ylabel("Valor real")
ax_cm.set_title("Matriz de confusión")

class_names = list(labels_map.values())
tick_marks = np.arange(nClasses)
ax_cm.set_xticks(tick_marks)
ax_cm.set_yticks(tick_marks)
ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
ax_cm.set_yticklabels(class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(
            j,
            i,
            int(cm[i, j]),
            ha="center",
            va="center",
            color="white" if im.norm(cm[i, j]) > 0.5 else "black",
        )

fig_cm.tight_layout()
fig_cm.savefig("confusion_matrix_SNN_simple_binaria_pion_kaon_50_epocas.png", dpi=300, bbox_inches="tight")



modelo_completo.eval()

all_probs = []
all_targets = []

with torch.no_grad():
    for data_batch, targets_batch in test_loader:
        data_batch = data_batch.to(snnfn.device)
        targets_batch = targets_batch.to(snnfn.device)

        log_probs = modelo_completo(data_batch)

        probs = torch.exp(log_probs)

        all_probs.append(probs.cpu())
        all_targets.append(targets_batch.view(-1).cpu())

all_probs = torch.cat(all_probs, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()


fig, axes = plt.subplots(1, nClasses, figsize=(7 * nClasses, 5), sharex=True, sharey=True)
axes = axes.flatten()

for true_cls in range(nClasses):
    ax = axes[true_cls]
    mask = (all_targets == true_cls)

    print(mask.sum())
    for pred_cls in range(nClasses):
        ax.hist(
            all_probs[mask, pred_cls],
            bins=1000,
            range=(0.0, 1.0),
            density=True,
            histtype="step",
            linewidth=2,
            label=f"P({class_names[pred_cls]})"
        )

    ax.set_title(f"Eventos reales = {class_names[true_cls]}")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Densidad")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle("Distribución de probabilidades predichas condicionada a la clase real", y=0.98)
fig.tight_layout()
fig.savefig("probabilidades_SNN_simple_binaria_pion_kaon_50_epocas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Curva ROC (clase positiva: PAIR_NAMES[1])
fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
roc_auc = auc(fpr, tpr)
print(f"AUC ROC = {roc_auc:.4f}")

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
ax_roc.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
ax_roc.set_xlabel("Tasa de falsos positivos")
ax_roc.set_ylabel("Tasa de verdaderos positivos")
ax_roc.set_title(f"Curva ROC ({PAIR_NAMES[1]} como clase positiva)")
ax_roc.legend(loc="lower right")
ax_roc.grid(alpha=0.3)
fig_roc.tight_layout()
fig_roc.savefig("roc_SNN_simple_binaria_pion_kaon_50_epocas.png", dpi=300, bbox_inches="tight")
plt.close(fig_roc)

print("terminado")
