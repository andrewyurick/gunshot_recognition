import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from panns_inference import AudioTagging
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from panns_inference.config import labels_csv_path, test_data_folder, labels


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix\n',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class', fontsize=11)
    plt.xlabel('Predicted Class\n\n\nAccuracy = {:0.4f}'.format(accuracy), fontsize=11)
    plt.show()


def traverse_folder(fd):
    paths = []
    names = []

    for root, dirs, files in os.walk(fd):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)

    return names, paths


def print_audio_tagging_result(clipwise_output):
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    print('\nRanked Recognition Results:')
    for k in range(10):
        print('    {}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))

    return np.array(labels)[sorted_indexes[0]]


if __name__ == '__main__':

    device = 'cpu' # 'cuda' | 'cpu'
    audio_files, paths = traverse_folder(test_data_folder)
    y_pred = []
    y_true = []

    for audio_file in audio_files:
        audio_path = os.path.join(test_data_folder, audio_file)
        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)

        print('\n------------------ Gunshot Recognition ------------------')
        at = AudioTagging(checkpoint_path=None, device=device)
        (clipwise_output, embedding) = at.inference(audio)

        print('Input labels: ' + labels_csv_path)
        print('Input test data:  ' + test_data_folder)
        print('Input audio:  ' + audio_path)
        y_pred.append(print_audio_tagging_result(clipwise_output[0]))
        y_true.append(audio_file.split('.')[0])

    print('\nAccuracy Score = ' + str(accuracy_score(y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, normalize=False, target_names=labels, title='Confusion Matrix\n')
