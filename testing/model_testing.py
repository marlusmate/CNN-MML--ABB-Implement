import tensorflow as tf
import json
#import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import roc_curve



# Dataset Parameters
param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
output_proc_shape = (len(param_list),)
batch_size = 15
output_img_shape = (128, 128, 1)
no_classes = 3
no_epochs = 1

# Model Name
model_name = 'Test-HybridFusion2-CnnOutput'

# Paths

y_true_list = 'labels.pickle'
y_pred_list = f'predictions/y_pred_{model_name}.pickle'


# Load Predictions, True Labels
with open(y_pred_list, 'rb') as f:
    pred = pickle.load(f)

with open(y_true_list, 'rb') as f:
    label = pickle.load(f)
    print("Test-Verteilung:\n", "0:", label.count(0),"\n1:", label.count(1),"\n2:", label.count(2))


if len(pred[0]) == 1:
    pred = pred
else:
    pred = pred[0]

one_hot_encoder = tf.one_hot(range(no_classes), no_classes)
label_sparse = np.asarray([one_hot_encoder[lb].numpy() for lb in label])
print("Daten geladen")

prediction = np.argmax(pred, axis=-1)
prediction_sparse = np.asarray([one_hot_encoder[lb].numpy() for lb in prediction])


# Get Categorical Accuracy
acc = tf.keras.metrics.CategoricalAccuracy()
acc.update_state(label_sparse, pred)
print("Test Genauigkeit: ", acc.result().numpy())

# Confusion Matrix
# Rows: True / Label
# Columns: Predicition
cf_mat = tf.math.confusion_matrix(label, prediction, num_classes=no_classes)
row_sums = tf.math.reduce_sum(input_tensor= cf_mat, axis=1)
cf_mat_norm = cf_mat / row_sums

#Viz
fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(cf_mat.numpy(), cmap=plt.cm.Blues, alpha=0.5)
for m in range(cf_mat.shape[0]):
    for n in range(cf_mat.shape[1]):
        px.text(x=m,y=n,s=cf_mat.numpy()[n, m], va='center', ha='center', size='xx-large')
# Sets the labels
plt.xticks(ticks=np.arange(no_classes), labels=['flooded', 'loaded', 'dispersed'],fontsize='large', rotation=30)
plt.yticks(ticks=np.arange(no_classes), labels=['flooded', 'loaded', 'dispersed'],fontsize='large')
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('CM LeNet20x50 Regime Detection', fontsize=15)
conf1 = plt.gcf()
#plt.show()
# save conf mat
conf1.savefig(f"../Figures/{model_name}/ConfusionMatrixNormed")

# Get Metrics - Precision, Recall, F1
# get one vs rest labels
pred_0 = pred[:,0]
prediction_0 = prediction_sparse[:,0]
label_0 = label_sparse[:,0]

pred_1 = pred[:,1]
prediction_1 = prediction_sparse[:,1]
label_1 = label_sparse[:,1]

pred_2 = pred[:,2]
prediction_2 = prediction_sparse[:,2]
label_2 = label_sparse[:,2]

# Calc Precision, Recall, F1
# Label 0
precision_0 = Precision()
recall_0 = Recall()

precision_0.update_state(y_true=label_0, y_pred=pred_0)
recall_0.update_state(y_true=label_0, y_pred=pred_0)
f1_0 = 2 * (precision_0.result().numpy() * recall_0.result().numpy()) / (precision_0.result().numpy() + recall_0.result().numpy())


# Label 1
precision_1 = Precision()
recall_1 = Recall()

precision_1.update_state(y_true=label_1, y_pred=pred_1)
recall_1.update_state(y_true=label_1, y_pred=pred_1)
f1_1 = 2 * (precision_1.result().numpy() * recall_1.result().numpy()) / (precision_1.result().numpy() + recall_1.result().numpy())

# Label 2
precision_2 = Precision()
precision_2.update_state(y_true=label_2, y_pred=pred_2)

recall_2 = Recall()
recall_2.update_state(y_true=label_2, y_pred=pred_2)
f1_2 = 2 * (precision_2.result().numpy() * recall_2.result().numpy()) / (precision_2.result().numpy() + recall_2.result().numpy())

# Plot Metrics
metrics_names = ["Precision", "Recall"]
class_0 = [precision_0.result().numpy(), recall_0.result().numpy()]
class_1 = [precision_1.result().numpy(), recall_1.result().numpy()]
class_2 = [precision_2.result().numpy(), recall_2.result().numpy()]

x_axis = np.arange(len(metrics_names))
plt.figure()
plt.bar(x_axis -0.3, class_0, width=0.2, label = 'flooded')
plt.bar(x_axis -0.1, class_1, width=0.2, label = 'loaded')
plt.bar(x_axis +0.1, class_2, width=0.2, label = 'dispersed')
plt.xticks(x_axis, metrics_names)
plt.legend()
plt.title("Precision, Recall LeNet20x50 Regime Detecion")
bar1 = plt.gcf()
#plt.show()
bar1.savefig(f"../Figures/{model_name}/Barplot_Precsion-Recall")


# ROC, AUC, log loss?
# y_pred value for true class of instance needed
#y_true_proba = get_pred_proba(pred_proba=pred_finaloutput_sparse, y_true=label)

fpr_0, tpr_0, thresholds_0 = roc_curve(label_0, pred_0)
fpr_1, tpr_1, thresholds_1 = roc_curve(label_1, pred_1)
fpr_2, tpr_2, thresholds_2= roc_curve(label_2, pred_2)
plt.figure()
plt.plot(fpr_0, tpr_0, label=0)
plt.plot(fpr_1, tpr_1, label=1)
plt.plot(fpr_2, tpr_2, label=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(["flooded", "loaded", "dispersed"])
plt.title("ROC-Curve LeNext20x50 Regime Detection")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
roc1 = plt.gcf()
#plt.show()
roc1.savefig(f"../Figures/{model_name}/Roc_curves")

# Calc AUC
auc_0 = tf.keras.metrics.AUC()
auc_0.update_state(label_0, pred_0)
auc_0 = auc_0.result().numpy()

auc_1 = tf.keras.metrics.AUC()
auc_1.update_state(label_1, pred_1)
auc_1 = auc_1.result().numpy()

auc_2 = tf.keras.metrics.AUC()
auc_2.update_state(label_0, pred_0)
auc_2 = auc_2.result().numpy()

# Confidence Score
pred_max = []
pred_sc_max = []
for ps in pred:
    ps.sort()
    pred_max.append(ps[2])
    pred_sc_max.append(ps[1])
pred_max = np.asarray(pred_max)
pred_sc_max = np.asarray(pred_sc_max)

# Plot Metrics
fig, ax = plt.subplots()

# Example data
metrics_names = ["F1-Score", "Recall"]
class_0 = [f1_0, auc_0]
class_1 = [f1_1, auc_1]
class_2 = [f1_2, auc_2]

x_axis = np.arange(len(metrics_names))

plt.bar(x_axis -0.3, class_0, width=0.2, label = 'flooded')
plt.bar(x_axis -0.1, class_1, width=0.2, label = 'loaded')
plt.bar(x_axis +0.1, class_2, width=0.2, label = 'dispersed')
plt.xticks(x_axis, metrics_names)
plt.legend()
plt.title("F1-Score, AUC LeNet20x50 Regime Detecion")
bar1 = plt.gcf()
plt.show()
bar1.savefig(f"../Figures/{model_name}/Barplot_F1-AUC.png")

# Save Metrics to json-file
metrics = {'precision_0': precision_0.result().numpy().astype('float64'), 'precision_1': precision_1.result().numpy().astype('float64'), 'precision_2': precision_2.result().numpy().astype('float64'),
           'recall_0': recall_0.result().numpy().astype('float64'), 'recall_1': recall_1.result().numpy().astype('float64'), 'recall_2': recall_2.result().numpy().astype('float64'),
           'f1-score_0': f1_0, 'f1-score_1': f1_1, 'f1-score_2': f1_2,
           'auc_0': auc_0.astype('float64'), 'auc_1': auc_1.astype('float64'), 'auc_2': auc_2.astype('float64'),
           'mean_1pred_prob': pred_max.mean().astype('float64'), 'mean_2pred_prob': pred_sc_max.mean().astype('float64'),
           'med_1pred_prob': np.median(pred).astype('float64'), 'med_2pred_prob': np.median(pred_sc_max).astype('float64')
           }



with open(f'metrics/metrics_{model_name}.json', 'w', encoding="utf8") as f: json.dump(metrics, f)




