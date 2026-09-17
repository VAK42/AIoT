import os
import json
import time
import asyncio
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.firebase import FirebaseService
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from backend.train import trainModel
classesPath = 'backend/classes.json'
modelPath = 'backend/model.keras'
statsPath = 'backend/stats.json'
if not (os.path.exists(classesPath) and os.path.exists(modelPath)):
  trainModel()
with open(classesPath, 'r', encoding='utf-8') as f:
  meta = json.load(f)
classes = meta['classes']
model = tf.keras.models.load_model(modelPath)
appConfig = {'retrainMinutes': 15, 'lastRetrain': time.time(), 'isRetraining': False}
async def retrainCronTask():
  global model
  while True:
    await asyncio.sleep(10)
    if not appConfig['isRetraining'] and time.time() - appConfig['lastRetrain'] >= appConfig['retrainMinutes'] * 60:
      appConfig['isRetraining'] = True
      try:
        trainModel()
        model = tf.keras.models.load_model(modelPath)
        appConfig['lastRetrain'] = time.time()
      except Exception:
        pass
      finally:
        appConfig['isRetraining'] = False
@asynccontextmanager
async def lifespan(fastApp: FastAPI):
  cronTask = asyncio.create_task(retrainCronTask())
  yield
  cronTask.cancel()
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/dashboard', StaticFiles(directory='dashboard', html=True), name='dashboard')
firebaseService = FirebaseService()
@app.get('/')
def getRoot():
  return RedirectResponse(url='/dashboard/index.html')
@app.get('/health')
def getHealth():
  return {'status': 'ok', 'model': 'MultiTask1DCNN', 'classes': classes}
@app.get('/api/stats')
def getTrainingStats():
  if os.path.exists(statsPath):
    with open(statsPath, 'r', encoding='utf-8') as f:
      return json.load(f)
  return {'status': 'none'}
@app.post('/api/retrain')
def postRetrain():
  global model
  appConfig['isRetraining'] = True
  try:
    stats = trainModel()
    model = tf.keras.models.load_model(modelPath)
    appConfig['lastRetrain'] = time.time()
    appConfig['isRetraining'] = False
    return {'status': 'ok', 'stats': stats}
  except Exception as e:
    appConfig['isRetraining'] = False
    return JSONResponse(status_code=500, content={'error': str(e)})
@app.get('/api/settings')
def getSettings():
  status = firebaseService.getStatus()
  return {
    'retrainMinutes': appConfig['retrainMinutes'],
    'lastRetrain': appConfig['lastRetrain'],
    'secondsUntilRetrain': max(0, int(appConfig['retrainMinutes'] * 60 - (time.time() - appConfig['lastRetrain']))),
    'isRetraining': appConfig['isRetraining'],
  }
@app.get('/api/status')
def getStatus():
  return firebaseService.getStatus()
@app.get('/api/checkconnections')
def getCheckConnections():
  return firebaseService.checkAllConnections()
@app.get('/api/connections')
def getConnections():
  return firebaseService.connectionStatus
@app.post('/api/settings')
async def postSettings(payload: dict):
  if 'retrainMinutes' in payload:
    appConfig['retrainMinutes'] = max(1, int(payload['retrainMinutes']))
  if 'activeAir' in payload:
    firebaseService.setAirType(payload['activeAir'])
  return getSettings()
def extractFeatures(w):
  n = len(w)
  meanVal = float(np.mean(w))
  stdVal = float(np.std(w))
  slopeVal = float((w[-1] - w[0]) / n)
  minVal = float(np.min(w))
  maxVal = float(np.max(w))
  rngVal = maxVal - minVal
  deltaVal = float(w[-1] - w[0])
  sortedW = np.sort(w)
  q25 = float(sortedW[int(n * 0.25)])
  q75 = float(sortedW[int(n * 0.75)])
  diffs = [w[i] - w[i - 1] for i in range(1, n)]
  diffMean = float(np.mean(diffs)) if diffs else 0.0
  diffStd = float(np.std(diffs)) if diffs else 0.0
  return [meanVal, stdVal, maxVal, minVal, rngVal, deltaVal, slopeVal, diffMean, diffStd, q75 - q25]
def runInference(windowValues, compS3, activeAir):
  tStart = time.perf_counter()
  w = list(windowValues)
  if len(w) < 20:
    fillVal = w[0] if w else float(compS3)
    w = [fillVal] * (20 - len(w)) + w
  rawTensor = np.array(w[-20:], dtype=np.float32).reshape(1, 20, 1)
  gasPreds, ppmPreds = model(rawTensor, training=False)
  gasProbs = gasPreds.numpy()[0]
  bestIdx = int(np.argmax(gasProbs))
  predGas = classes[bestIdx]
  conf = int(round(float(gasProbs[bestIdx]) * 100))
  probMap = {classes[i]: round(float(gasProbs[i]), 4) for i in range(len(classes))}
  estimatedppm = round(float(max(0.0, ppmPreds.numpy()[0][0])), 2)
  latencyMs = round((time.perf_counter() - tStart) * 1000, 2)
  return {'gas': predGas, 'confidence': conf, 'probabilities': probMap, 'estimatedppm': estimatedppm, 'latencyMs': latencyMs}
@app.websocket('/stream')
async def streamWs(ws: WebSocket):
  await ws.accept()
  state = {'isPlaying': True, 'speedMs': 250, 'window': [], 'lastEMA': None}
  async def listenCommands():
    try:
      while True:
        msg = await ws.receive_text()
        data = json.loads(msg)
        action = data.get('action')
        if action == 'setActiveAir':
          firebaseService.setAirType(data.get('airType', 'CleanAir'))
        elif action == 'setSpeed':
          state['speedMs'] = max(50, int(data.get('speedMs', 250)))
    except Exception:
      pass
  cmdTask = asyncio.create_task(listenCommands())
  try:
    while True:
      if state['isPlaying']:
        activeAir = firebaseService.activeAir
        fbItem = firebaseService.getLatest(activeAir)
        if fbItem is None:
          state['window'].clear()
          state['lastEMA'] = None
          packet = {
            'online': False,
            'timestamp': time.strftime('%I:%M:%S %p'),
            's3': 0.0,
            's3Raw': 0.0,
            'temperature': 28.5,
            'humidity': 64.0,
            'trueGas': activeAir,
            'trueppm': 0.0,
            'phase': 'Offline',
            'gas': activeAir,
            'activeAir': activeAir,
            'confidence': 0,
            'probabilities': {'CleanAir': 0.0, 'H2S': 0.0, 'NH3': 0.0},
            'estimatedppm': 0.0,
            'riskLevel': 'Offline',
            'features': None,
            'prognostics': {
              'trajectoryPoints': [0.0, 0.0, 0.0, 0.0],
              'maxProjectedS3': 0.0,
              'isImminentEmergency': False,
              'timeToEmergencySec': None
            },
            'latencyMs': 0.0,
            'frameIndex': 0,
            'totalFrames': 250,
            'mode': 'liveFirebase',
            'isPlaying': state['isPlaying'],
            'sensor': {
              'voltage1': 0.0,
              'voltage2': 0.0,
              'sensor1': 0,
              'sensor2': 0,
              'dacVoltage': 0.0,
              'pulse': 0,
              'point': 0,
              'waveType': 0
            },
            'cron': {
              'retrainMinutes': appConfig['retrainMinutes'],
              'secondsUntilRetrain': max(0, int(appConfig['retrainMinutes'] * 60 - (time.time() - appConfig['lastRetrain']))),
              'ramCounts': {k: len(v) for k, v in firebaseService.ramBuffers.items()},
              'secondsUntilFlush': max(0, int(firebaseService.flushInterval - (time.time() - firebaseService.lastFlush)))
            },
            'connections': firebaseService.connectionStatus
          }
          await ws.send_text(json.dumps(packet))
          await asyncio.sleep(state['speedMs'] / 1000.0)
          continue
        voltage1 = float(fbItem.get('voltage1', 0.0))
        voltage2 = float(fbItem.get('voltage2', 0.0))
        rawVal = voltage2
        temp = float(fbItem.get('temperature', 28.5))
        hum = float(fbItem.get('humidity', 64.0))
        activeAir = firebaseService.activeAir
        trueGas = fbItem.get('gasType', activeAir)
        truePPM = float(fbItem.get('ppm', 0.0))
        phase = 'LiveSensor'
        dacVoltage = float(fbItem.get('dacVoltage', fbItem.get('dac_voltage', 0.0)))
        pulse = int(fbItem.get('pulse', 0))
        point = int(fbItem.get('point', 0))
        waveType = int(fbItem.get('waveType', fbItem.get('wave_type', 0)))
        sensor1 = int(fbItem.get('sensor1', fbItem.get('raw_sensor1', 0)))
        sensor2 = int(fbItem.get('sensor2', fbItem.get('raw_sensor2', 0)))
        idx = point
        totalFrames = 250
        if state['lastEMA'] is None:
          state['lastEMA'] = rawVal
        else:
          state['lastEMA'] = 0.2 * rawVal + 0.8 * state['lastEMA']
        compFactor = 1.0 + 0.0035 * (temp - 25.0) + 0.0015 * (hum - 60.0)
        compVal = state['lastEMA'] / compFactor
        if not state['window']:
          state['window'] = [compVal] * 20
        else:
          state['window'].append(compVal)
          if len(state['window']) > 20:
            state['window'].pop(0)
        feat = extractFeatures(state['window'])
        if activeAir == 'MixedAir':
          inf = runInference(state['window'], compVal, activeAir)
          gas = inf['gas']
          ppmVal = inf['estimatedppm']
          risk = 'Normal'
          if gas == 'H2S':
            if ppmVal >= 10.0 or compVal >= 0.95:
              risk = 'Emergency'
            elif ppmVal >= 5.0 or compVal >= 0.80:
              risk = 'Hazardous'
            elif ppmVal >= 1.0 or compVal >= 0.70:
              risk = 'Warning'
          elif gas == 'NH3':
            if ppmVal >= 50.0 or compVal >= 1.20:
              risk = 'Emergency'
            elif ppmVal >= 25.0 or compVal >= 0.90:
              risk = 'Hazardous'
            elif ppmVal >= 10.0 or compVal >= 0.75:
              risk = 'Warning'
          dt = state['speedMs'] / 1000.0
          slopePerStep = feat[6] if feat else 0.0
          slopeVal = slopePerStep / max(0.01, dt)
          horizonSteps = [5, 10, 15, 20]
          traj = [round(float(compVal + slopeVal * s), 3) for s in horizonSteps]
          maxProj = max(compVal, *traj)
          isEmerg = (gas in ['H2S', 'NH3'] and maxProj >= 0.95) or (gas == 'H2S' and slopeVal > 0.008)
          tte = None
          if isEmerg:
            rate = max(0.002, slopeVal)
            tte = max(3, min(60, round((0.95 - compVal) / rate)))
          confidence = inf['confidence']
          probabilities = inf['probabilities']
          latencyMs = inf['latencyMs']
        else:
          gas = activeAir
          ppmVal = 0.0
          risk = 'Normal'
          traj = [compVal, compVal, compVal, compVal]
          maxProj = compVal
          isEmerg = False
          tte = None
          confidence = 100
          probabilities = {'CleanAir': 1.0 if activeAir == 'CleanAir' else 0.0, 'H2S': 1.0 if activeAir == 'H2S' else 0.0, 'NH3': 1.0 if activeAir == 'NH3' else 0.0}
          latencyMs = 0.0
        packet = {
          'online': True,
          'timestamp': time.strftime('%I:%M:%S %p'),
          's3': round(compVal, 4),
          's3Raw': round(rawVal, 4),
          'temperature': temp,
          'humidity': hum,
          'trueGas': trueGas,
          'trueppm': truePPM,
          'phase': phase,
          'gas': gas,
          'activeAir': activeAir,
          'confidence': confidence,
          'probabilities': probabilities,
          'estimatedppm': ppmVal,
          'riskLevel': risk,
          'features': {
            'mean': round(feat[0], 5),
            'std': round(feat[1], 5),
            'max': round(feat[2], 5),
            'min': round(feat[3], 5),
            'range': round(feat[4], 5),
            'delta': round(feat[5], 5),
            'slope': round(feat[6], 5),
            'diffmean': round(feat[7], 5),
            'diffstd': round(feat[8], 5),
            'iqr': round(feat[9], 5)
          } if feat else None,
          'prognostics': {
            'trajectoryPoints': traj,
            'maxProjectedS3': round(maxProj, 3),
            'isImminentEmergency': isEmerg,
            'timeToEmergencySec': tte
          },
          'latencyMs': latencyMs,
          'frameIndex': idx,
          'totalFrames': totalFrames,
          'mode': 'liveFirebase',
          'isPlaying': state['isPlaying'],
          'sensor': {
            'voltage1': voltage1,
            'voltage2': voltage2,
            'sensor1': sensor1,
            'sensor2': sensor2,
            'dacVoltage': dacVoltage,
            'pulse': pulse,
            'point': point,
            'waveType': waveType
          },
          'cron': {
            'retrainMinutes': appConfig['retrainMinutes'],
            'secondsUntilRetrain': max(0, int(appConfig['retrainMinutes'] * 60 - (time.time() - appConfig['lastRetrain']))),
            'ramCounts': {k: len(v) for k, v in firebaseService.ramBuffers.items()},
            'secondsUntilFlush': max(0, int(firebaseService.flushInterval - (time.time() - firebaseService.lastFlush)))
          },
          'connections': firebaseService.connectionStatus
        }
        await ws.send_text(json.dumps(packet))
      await asyncio.sleep(state['speedMs'] / 1000.0)
  except WebSocketDisconnect:
    pass
  finally:
    cmdTask.cancel()
if __name__ == '__main__':
  uvicorn.run('backend.app:app', host='127.0.0.1', port=8001, reload=True)