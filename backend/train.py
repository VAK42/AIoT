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
  cycles, gLabels, ppmLabels = [], [], []
  targetGrid = np.arange(250)
  for gas, fpath in csvFiles.items():
    if os.path.exists(fpath):
      df = pd.read_csv(fpath, on_bad_lines='skip')
      zeroIdxs = list(df.index[df['point'] == 0])
      zeroIdxs.append(len(df))
      for k in range(len(zeroIdxs) - 1):
        sub = df.iloc[zeroIdxs[k]:zeroIdxs[k + 1]]
        if sub['point'].max() >= 240 and len(sub) >= 200:
          sub = sub.drop_duplicates(subset=['point'])
          pts = sub['point'].values
          v1 = sub['voltage1'].values.astype(np.float32)
          v2 = sub['voltage2'].values.astype(np.float32)
          ppmVal = float(sub['ppm'].mean()) if 'ppm' in sub.columns else 0.0
          v1Interp = np.interp(targetGrid, pts, v1).astype(np.float32)
          v2Interp = np.interp(targetGrid, pts, v2).astype(np.float32)
          cycles.append(np.stack([v1Interp, v2Interp], axis=-1))
          gLabels.append(labelMap[gas])
          ppmLabels.append(ppmVal)
  X = np.array(cycles, dtype=np.float32)
  yG = np.array(gLabels, dtype=np.int32)
  yP = np.array(ppmLabels, dtype=np.float32)
  np.random.seed(42)
  idx = np.random.permutation(len(X))
  split = int(len(X) * 0.8)
  return X[idx[:split]], yG[idx[:split]], yP[idx[:split]], X[idx[split:]], yG[idx[split:]], yP[idx[split:]], classes
def trainModel():
  XTr, yGTr, yPTr, XVa, yGVa, yPVa, classes = buildDataset()
  rawInput = tf.keras.layers.Input(shape=(250, 2), name='rawInput')
  x = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(rawInput)
  x = tf.keras.layers.MaxPooling1D(2)(x)
  x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
  x = tf.keras.layers.MaxPooling1D(2)(x)
  x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
  x = tf.keras.layers.GlobalAveragePooling1D()(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  gasOutput = tf.keras.layers.Dense(len(classes), activation='softmax', name='gasOutput')(x)
  ppmOutput = tf.keras.layers.Dense(1, activation='softplus', name='ppmOutput')(x)
  model = tf.keras.Model(inputs=rawInput, outputs=[gasOutput, ppmOutput])
  model.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss={'gasOutput': 'sparse_categorical_crossentropy', 'ppmOutput': 'huber'}, loss_weights={'gasOutput': 1.0, 'ppmOutput': 0.1}, metrics={'gasOutput': 'accuracy', 'ppmOutput': 'mae'})
  fitKw = {'validation_data': (XVa, [yGVa, yPVa]), 'epochs': 25, 'batch_size': 16, 'verbose': 0}
  history = model.fit(XTr, [yGTr, yPTr], **fitKw)
  evalRes = model.evaluate(XVa, [yGVa, yPVa], verbose=0)
  model.save('backend/model.keras')
  with open('backend/classes.json', 'w', encoding='utf-8') as f:
    json.dump({'classes': classes, 'windowSize': 250, 'channels': 2, 'architecture': 'MultiTask1DCNN'}, f, indent=2)
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