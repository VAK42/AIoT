import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
def buildDataset():
  classes = ['CleanAir', 'H2S', 'NH3']
  labelMap = {c: i for i, c in enumerate(classes)}
  csvFiles = {'CleanAir': 'data/CleanAir.csv', 'H2S': 'data/H2S.csv', 'NH3': 'data/NH3.csv'}
  windows, gLabels, ppmLabels = [], [], []
  for gas, fpath in csvFiles.items():
    if os.path.exists(fpath):
      df = pd.read_csv(fpath)
      vals = df['voltage2'].values.astype(np.float32)
      ppms = df['ppm'].values.astype(np.float32) if 'ppm' in df.columns else np.zeros(len(vals), dtype=np.float32)
      for i in range(0, len(vals) - 20 + 1, 10):
        windows.append(vals[i:i + 20])
        gLabels.append(labelMap[gas])
        ppmLabels.append(float(ppms[i + 19]))
  X = np.expand_dims(np.array(windows, dtype=np.float32), axis=-1)
  yG = np.array(gLabels, dtype=np.int32)
  yP = np.array(ppmLabels, dtype=np.float32)
  np.random.seed(42)
  idx = np.random.permutation(len(X))
  split = int(len(X) * 0.8)
  return X[idx[:split]], yG[idx[:split]], yP[idx[:split]], X[idx[split:]], yG[idx[split:]], yP[idx[split:]], classes
def trainModel():
  XTr, yGTr, yPTr, XVa, yGVa, yPVa, classes = buildDataset()
  rawInput = tf.keras.layers.Input(shape=(20, 1), name='rawInput')
  x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(rawInput)
  x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
  x = tf.keras.layers.GlobalAveragePooling1D()(x)
  x = tf.keras.layers.Dense(32, activation='relu')(x)
  gasOutput = tf.keras.layers.Dense(len(classes), activation='softmax', name='gasOutput')(x)
  ppmOutput = tf.keras.layers.Dense(1, activation='softplus', name='ppmOutput')(x)
  model = tf.keras.Model(inputs=rawInput, outputs=[gasOutput, ppmOutput])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss={'gasOutput': 'sparse_categorical_crossentropy', 'ppmOutput': 'huber'}, loss_weights={'gasOutput': 1.0, 'ppmOutput': 0.1}, metrics={'gasOutput': 'accuracy', 'ppmOutput': 'mae'})
  history = model.fit(XTr, [yGTr, yPTr], validation_data=(XVa, [yGVa, yPVa]), epochs=20, batch_size=32, verbose=0)
  evalRes = model.evaluate(XVa, [yGVa, yPVa], verbose=0)
  model.save('backend/model.keras')
  with open('backend/classes.json', 'w', encoding='utf-8') as f:
    json.dump({'classes': classes, 'windowSize': 20, 'architecture': 'MultiTask1DCNN'}, f, indent=2)
  stats = {
    'epochs': len(history.history['loss']),
    'loss': [round(float(v), 4) for v in history.history['loss']],
    'valLoss': [round(float(v), 4) for v in history.history.get('val_loss', [])],
    'accuracy': [round(float(v) * 100, 2) for v in history.history.get('gasOutput_accuracy', [])],
    'valAccuracy': [round(float(v) * 100, 2) for v in history.history.get('val_gasOutput_accuracy', [])],
    'finalLoss': round(float(evalRes[0]), 4),
    'finalAccuracy': round(float(evalRes[3]) * 100, 2),
    'finalMAE': round(float(evalRes[4]), 3),
    'classes': classes,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
  }
  with open('backend/stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)
  return stats
if __name__ == '__main__':
  trainModel()