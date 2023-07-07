import matplotlib.pyplot as plt
import os
from plotly.offline import plot
import plotly.graph_objects as go



def showLossCurves(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(os.path.join(os.getcwd(), 'training_loss.png'))


def showAccuracyCurves(history):

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.savefig(os.path.join(os.getcwd(), 'training_accuracy.png'))






def plot_confusion_matrix(confusion_mtx, classes):

    fig = go.Figure(data=go.Heatmap(z=confusion_mtx,
                                    x=classes,
                                    y=classes,
                                    colorscale='Blues'))
    fig.update_layout(title='Confusion Matrix',
                      xaxis_title='Predicted label',
                      yaxis_title='True label',
                      xaxis=dict(type='category', automargin=True),
                      yaxis=dict(type='category', automargin=True),
                      autosize=True,
                      width=2000,
                      height=2000
                      )
    plot(fig, filename="confusion_matrix.html", auto_open=False)