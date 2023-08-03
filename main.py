import torch
import os
from torch import nn
import warnings
import utils
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    train_loader, test_loader = utils.prepare_dataset()

    # create a model instance
    model_fp32 = utils.LeNet5()

    # # Print model parameters
    # for name, parameter in model_fp32.named_parameters():
    #     print(name, ": ", parameter.dtype)

    # Check saved model with 32 bits exists, otherwise train it
    if not os.path.isfile("models/model_32.pth"):
        losses = utils.train(model_fp32, train_loader, num_epochs=10)
        # Save weights
        torch.save(model_fp32.state_dict(), "models/model_32.pth")
    else:
        # Load model
        model_fp32.load_state_dict(torch.load("models/model_32.pth"))

    results = []
    num_experiments = 10
    avg_model_size, avg_time_evaluation, avg_accuracy = utils.calc_model_metrics(model_fp32, test_loader, num_experiments)
    print('Model type: Original - Size (KB): {}'.format(avg_model_size))
    print('''Accuracy: {}% - Elapsed time (seconds): {}'''.format(avg_accuracy, avg_time_evaluation))
    results.append([avg_model_size, avg_time_evaluation, avg_accuracy])

    torch.backends.quantized.engine = 'qnnpack'
    quantitation_method = 'x86'

    # Dynamic Quantitation
    dquant_avg_model_size, dquant_avg_time_evaluation, dquant_avg_accuracy = utils.calc_dynamic_quant_metrics(model_fp32, test_loader, num_experiments)
    print('Model type: Dynamic-Quantitation - Size (KB): {}'.format(dquant_avg_model_size))
    print('''Accuracy: {}% - Elapsed time (seconds): {}'''.format(dquant_avg_accuracy, dquant_avg_time_evaluation))
    results.append([dquant_avg_model_size, dquant_avg_time_evaluation, dquant_avg_accuracy])

    # Post-Training Static Quantization
    quantitation_method = 'fbgemm'
    stat_quant_avg_model_size, stat_quant_avg_time_evaluation, stat_quant_avg_accuracy = utils.calc_post_training_static_quant_metrics(model_fp32, test_loader, quantitation_method, num_experiments)
    print('Model type: Static-Quant - Size (KB): {}'.format(stat_quant_avg_model_size))
    print('''Accuracy: {}% - Elapsed time (seconds): {}'''.format(stat_quant_avg_accuracy, stat_quant_avg_time_evaluation))
    results.append([stat_quant_avg_model_size, stat_quant_avg_time_evaluation, stat_quant_avg_accuracy])

    # Quantization Aware Training
    quant_aware_avg_model_size, quant_aware_avg_time_evaluation, quant_aware_avg_accuracy = utils.calc_quant_aware_training_metrics(model_fp32, train_loader, test_loader, quantitation_method,
                                                                                                                                    num_experiments)
    print('Model type: Quant Aware Training (QAT) - Size (KB): {}'.format(quant_aware_avg_model_size))
    print('''Accuracy: {}% - Elapsed time (seconds): {}'''.format(quant_aware_avg_accuracy, quant_aware_avg_time_evaluation))
    results.append([quant_aware_avg_model_size, quant_aware_avg_time_evaluation, quant_aware_avg_accuracy])

df = pd.DataFrame(results, columns=['model_size', 'inference_time', 'accuracy'], index=["Original", "DQuant", "PT Static Quant", "QAT"])
print(df)
df = df[['model_size', 'accuracy']].copy()
# Create matplotlib figure
fig = plt.figure()
# Create matplotlib axes
ax = fig.add_subplot(111)
# Create another axes that shares the same x-axis as ax.
ax2 = ax.twinx()
width = 0.2
df.model_size.plot(kind='bar', color='blue', ax=ax, width=width, position=0)
df.accuracy.plot(kind='bar', color='red', ax=ax2, width=width, position=1)
# Create custom legend for 2 yaxis
blue_patch = mpatches.Patch(color='blue', label='Model size')
red_patch = mpatches.Patch(color='red', label='Accuracy')
plt.legend(handles=[red_patch, blue_patch])
for bars in ax.containers:
    ax.bar_label(bars, labels=['   %.2f' % value for value in bars.datavalues], color='b', fontsize=10)
for bars in ax2.containers:
    ax2.bar_label(bars,labels=['%.2f   ' % value for value in bars.datavalues], color='r', fontsize=10)
# Set y labels, axis limits, rotation x-axis, and figure title
ax.set_ylabel('Model Size (KB)')
ax.set_ylim(0, 300)
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 100)
ax.xaxis.set_tick_params(rotation=0)
plt.title("Comparison Model Size - Accuracy between Quantization Method")
plt.savefig('results/model_comparison.png')
plt.show()
