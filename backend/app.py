import os
import json
import time
import asyncio
import uvicorn
import configparser
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
configIniPath = 'config.ini'
iniConfig = configparser.ConfigParser()
iniConfig.optionxform = str
def loadIniSettings():
  if not os.path.exists(configIniPath):
    iniConfig['Settings'] = {'retrainMinutes': '15', 'cleanMinutes': '5', 'speedMs': '250', 'activeAir': 'CleanAir', 'autoCleanEnabled': 'True', 'autoRetrainEnabled': 'True'}
    with open(configIniPath, 'w', encoding='utf-8') as cf:
      iniConfig.write(cf)
    with open(configIniPath, 'r', encoding='utf-8') as cf:
      lines = [l.strip() for l in cf if l.strip()]
    with open(configIniPath, 'w', encoding='utf-8') as cf:
      cf.write('\n'.join(lines))
  else:
    iniConfig.read(configIniPath, encoding='utf-8')
loadIniSettings()
if not (os.path.exists(classesPath) and os.path.exists(modelPath)):
  trainModel()
with open(classesPath, 'r', encoding='utf-8') as f:
  meta = json.load(f)
classes = meta['classes']
model = tf.keras.models.load_model(modelPath)
retrainMinFromIni = int(iniConfig.get('Settings', 'retrainMinutes', fallback='15'))
cleanMinFromIni = int(iniConfig.get('Settings', 'cleanMinutes', fallback='5'))
speedMsFromIni = int(iniConfig.get('Settings', 'speedMs', fallback='250'))
autoCleanFromIni = iniConfig.getboolean('Settings', 'autoCleanEnabled', fallback=True)
autoRetrainFromIni = iniConfig.getboolean('Settings', 'autoRetrainEnabled', fallback=True)
appConfig = {'retrainMinutes': retrainMinFromIni, 'lastRetrain': time.time(), 'isRetraining': False, 'speedMs': speedMsFromIni, 'autoRetrainEnabled': autoRetrainFromIni}
cleanerConfig = {'cleanMinutes': cleanMinFromIni, 'lastClean': time.time(), 'isCleaning': False, 'totalCleanCycles': 0, 'totalRemovedCycles': 0, 'lastReport': {}, 'autoCleanEnabled': autoCleanFromIni}
def saveIniSettings():
  if not iniConfig.has_section('Settings'):
    iniConfig.add_section('Settings')
  iniConfig.set('Settings', 'retrainMinutes', str(appConfig['retrainMinutes']))
  iniConfig.set('Settings', 'cleanMinutes', str(cleanerConfig['cleanMinutes']))
  iniConfig.set('Settings', 'speedMs', str(appConfig.get('speedMs', 250)))
  iniConfig.set('Settings', 'autoCleanEnabled', str(cleanerConfig.get('autoCleanEnabled', True)))
  iniConfig.set('Settings', 'autoRetrainEnabled', str(appConfig.get('autoRetrainEnabled', True)))
  if 'firebaseService' in globals() and firebaseService:
    iniConfig.set('Settings', 'activeAir', str(firebaseService.activeAir))
  with open(configIniPath, 'w', encoding='utf-8') as cf:
    iniConfig.write(cf)
  with open(configIniPath, 'r', encoding='utf-8') as cf:
    lines = [l.strip() for l in cf if l.strip()]
  with open(configIniPath, 'w', encoding='utf-8') as cf:
    cf.write('\n'.join(lines))
def cleanAllCsvFiles():
  cleanerConfig['isCleaning'] = True
  try:
    gases = ['CleanAir', 'H2S', 'NH3', 'MixedAir']
    targetGrid = np.arange(250)
    report = {}
    totalKept = 0
    totalRemoved = 0
    with firebaseService.lock:
      for gas in gases:
        fpath = f'data/{gas}.csv'
        if os.path.exists(fpath):
          df = pd.read_csv(fpath, on_bad_lines='skip')
          zeroIdxs = list(df.index[df['point'] == 0])
          zeroIdxs.append(len(df))
          validSubframes = []
          removedCount = 0
          for i in range(len(zeroIdxs) - 1):
            sub = df.iloc[zeroIdxs[i]:zeroIdxs[i + 1]]
            if len(sub) == 250 and np.array_equal(sub['point'].values, targetGrid):
              validSubframes.append(sub)
            else:
              removedCount += 1
          if validSubframes:
            cleanDf = pd.concat(validSubframes, ignore_index=True)
            cleanDf.to_csv(fpath, index=False)
          else:
            emptyDf = pd.DataFrame(columns=df.columns)
            emptyDf.to_csv(fpath, index=False)
          report[gas] = {'kept': len(validSubframes), 'removed': removedCount}
          totalKept += len(validSubframes)
          totalRemoved += removedCount
    cleanerConfig['totalCleanCycles'] = totalKept
    cleanerConfig['totalRemovedCycles'] = totalRemoved
    cleanerConfig['lastReport'] = report
    cleanerConfig['lastClean'] = time.time()
    global lastDatasetLoadMtime
    lastDatasetLoadMtime = 0.0
    return {'status': 'ok', 'report': report, 'totalKept': totalKept, 'totalRemoved': totalRemoved}
  except Exception as e:
    return {'status': 'error', 'message': str(e)}
  finally:
    cleanerConfig['isCleaning'] = False
async def retrainCronTask():
  global model
  while True:
    await asyncio.sleep(10)
    if not appConfig.get('autoRetrainEnabled', True):
      continue
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
async def csvCleanerCronTask():
  while True:
    await asyncio.sleep(10)
    if not cleanerConfig.get('autoCleanEnabled', True):
      continue
    if not cleanerConfig['isCleaning'] and time.time() - cleanerConfig['lastClean'] >= cleanerConfig['cleanMinutes'] * 60:
      cleanAllCsvFiles()
@asynccontextmanager
async def lifespan(fastApp: FastAPI):
  cronTask = asyncio.create_task(retrainCronTask())
  cleanerTask = asyncio.create_task(csvCleanerCronTask())
  yield
  cronTask.cancel()
  cleanerTask.cancel()
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/dashboard', StaticFiles(directory='dashboard', html=True), name='dashboard')
firebaseService = FirebaseService()
activeAirFromIni = iniConfig.get('Settings', 'activeAir', fallback='CleanAir')
if activeAirFromIni in firebaseService.dbUrls:
  firebaseService.setAirType(activeAirFromIni)
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
@app.get('/api/clean')
def getCleanCsv():
  return {
    'cleanMinutes': cleanerConfig['cleanMinutes'],
    'secondsUntilClean': max(0, int(cleanerConfig['cleanMinutes'] * 60 - (time.time() - cleanerConfig['lastClean']))),
    'isCleaning': cleanerConfig['isCleaning'],
    'totalCleanCycles': cleanerConfig['totalCleanCycles'],
    'totalRemovedCycles': cleanerConfig['totalRemovedCycles'],
    'lastReport': cleanerConfig['lastReport']
  }
@app.post('/api/clean')
def postCleanCsv():
  return cleanAllCsvFiles()
datasetEventsCache = []
lastDatasetLoadMtime = 0.0
def getDatasetEventsList():
  global datasetEventsCache, lastDatasetLoadMtime
  gases = ['MixedAir', 'NH3', 'H2S', 'CleanAir']
  latestMtime = 0.0
  for g in gases:
    fp = f'data/{g}.csv'
    if os.path.exists(fp):
      latestMtime = max(latestMtime, os.path.getmtime(fp))
  if datasetEventsCache and latestMtime <= lastDatasetLoadMtime:
    return datasetEventsCache
  targetGrid = np.arange(250)
  events = []
  for gas in gases:
    fp = f'data/{gas}.csv'
    if not os.path.exists(fp):
      continue
    mtime = os.path.getmtime(fp)
    try:
      df = pd.read_csv(fp, on_bad_lines='skip')
      if 'point' not in df.columns:
        continue
      zeroIdxs = list(df.index[df['point'] == 0])
      zeroIdxs.append(len(df))
      valid = []
      for i in range(len(zeroIdxs) - 1):
        sub = df.iloc[zeroIdxs[i]:zeroIdxs[i + 1]]
        if len(sub) == 250 and np.array_equal(sub['point'].values, targetGrid):
          valid.append(sub)
      n = len(valid)
      for idx, sub in enumerate(valid):
        v1 = [round(float(x), 3) for x in sub['voltage1'].values]
        v2 = [round(float(x), 3) for x in sub['voltage2'].values]
        ppmVal = round(float(sub['ppm'].mean()), 2) if 'ppm' in sub.columns else 0.0
        channelName = 'Clean Air' if gas == 'CleanAir' else ('Main' if gas == 'MixedAir' else gas)
        if gas == 'MixedAir':
          try:
            t = tf.convert_to_tensor(np.stack([np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)], axis=-1)[np.newaxis, ...], dtype=tf.float32)
            gasPreds, ppmPreds = model(t, training=False)
            probs = gasPreds.numpy()[0]
            bestIdx = int(np.argmax(probs))
            predGas = classes[bestIdx]
            conf = int(round(float(probs[bestIdx]) * 100))
            estPpm = round(float(max(0.0, ppmPreds.numpy()[0][0])), 2)
            predGasLabel = 'Clean Air' if predGas == 'CleanAir' else predGas
            msg = f'{predGasLabel} {estPpm:.2f} PPM ({conf}%)'
            predProbMap = {classes[j]: round(float(probs[j]), 4) for j in range(len(classes))}
          except Exception:
            predGas = 'CleanAir'
            conf = 100
            estPpm = ppmVal
            msg = f'Clean Air {estPpm:.2f} PPM (100%)'
            predProbMap = {'CleanAir': 1.0, 'H2S': 0.0, 'NH3': 0.0}
          waveObj = {
            'cycleIndex': idx + 1,
            'timestamp': '--:--:--',
            'channel': 'MixedAir',
            'gas': predGas,
            'ppm': estPpm,
            'confidence': conf,
            'probabilities': predProbMap,
            'risk': 'Normal',
            'speedMs': appConfig.get('speedMs', 250),
            'v1': v1,
            'v2': v2
          }
        else:
          minV1, maxV1 = min(v1), max(v1)
          minV2, maxV2 = min(v2), max(v2)
          msg = f'V1: {minV1:.2f}-{maxV1:.2f}V - V2: {minV2:.2f}-{maxV2:.2f}V'
          waveObj = {
            'cycleIndex': idx + 1,
            'timestamp': '--:--:--',
            'channel': gas,
            'gas': gas,
            'ppm': ppmVal,
            'confidence': 100,
            'probabilities': {gas: 1.0},
            'risk': 'Normal',
            'speedMs': appConfig.get('speedMs', 250),
            'v1': v1,
            'v2': v2
          }
        events.append({
          'id': f'ds{gas}Cycle{idx + 1}',
          'cycleIndex': idx + 1,
          'channel': channelName,
          'channelKey': gas,
          'time': '--:--:--',
          'telemetry': msg,
          'status': '200',
          'cycleWave': waveObj
        })
    except Exception:
      pass
  datasetEventsCache = events
  lastDatasetLoadMtime = latestMtime
  return datasetEventsCache
@app.get('/api/events')
def getEvents(channel: str = 'CleanAir'):
  evts = getDatasetEventsList()
  if channel and channel != 'All':
    matched = [e for e in evts if e['channelKey'] == channel]
    matched.sort(key=lambda e: e['cycleIndex'], reverse=True)
    return matched
  return evts
@app.get('/api/settings')
def getSettings():
  return {
    'retrainMinutes': appConfig['retrainMinutes'],
    'lastRetrain': appConfig['lastRetrain'],
    'secondsUntilRetrain': max(0, int(appConfig['retrainMinutes'] * 60 - (time.time() - appConfig['lastRetrain']))),
    'isRetraining': appConfig['isRetraining'],
    'autoRetrainEnabled': appConfig.get('autoRetrainEnabled', True),
    'cleanMinutes': cleanerConfig['cleanMinutes'],
    'secondsUntilClean': max(0, int(cleanerConfig['cleanMinutes'] * 60 - (time.time() - cleanerConfig['lastClean']))),
    'isCleaning': cleanerConfig['isCleaning'],
    'autoCleanEnabled': cleanerConfig.get('autoCleanEnabled', True),
    'totalCleanCycles': cleanerConfig['totalCleanCycles'],
    'totalRemovedCycles': cleanerConfig['totalRemovedCycles'],
    'lastReport': cleanerConfig['lastReport'],
    'speedMs': appConfig.get('speedMs', 250),
    'activeAir': firebaseService.activeAir
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
  if 'cleanMinutes' in payload:
    cleanerConfig['cleanMinutes'] = max(1, int(payload['cleanMinutes']))
  if 'speedMs' in payload:
    appConfig['speedMs'] = max(50, int(payload['speedMs']))
  if 'autoCleanEnabled' in payload:
    cleanerConfig['autoCleanEnabled'] = bool(payload['autoCleanEnabled'])
  if 'autoRetrainEnabled' in payload:
    appConfig['autoRetrainEnabled'] = bool(payload['autoRetrainEnabled'])
  if 'activeAir' in payload:
    firebaseService.setAirType(payload['activeAir'])
  saveIniSettings()
  return getSettings()
def extractFeatures(w):
  n = len(w)
  if n == 0:
    return [0.0] * 10
  meanVal = float(np.mean(w))
  stdVal = float(np.std(w))
  slopeVal = float((w[-1] - w[0]) / max(1, n))
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
def runInference(cycleTensor):
  tStart = time.perf_counter()
  gasPreds, ppmPreds = model(cycleTensor, training=False)
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
  state = {
    'isPlaying': True,
    'speedMs': appConfig.get('speedMs', 250),
    'cyclePoints': {},
    'lastPoint': -1,
    'cycleIndex': 0,
    'cycleGas': 'CleanAir',
    'cyclePpm': 0.0,
    'cycleConfidence': 100,
    'cycleProbabilities': {'CleanAir': 1.0, 'H2S': 0.0, 'NH3': 0.0},
    'cycleRisk': 'Normal',
    'lastCycleWave': None,
    'lastEMA1': None,
    'lastEMA2': None,
    'lastLatency': 0.0
  }
  async def listenCommands():
    try:
      while True:
        msg = await ws.receive_text()
        data = json.loads(msg)
        action = data.get('action')
        if action == 'setActiveAir':
          firebaseService.setAirType(data.get('airType', 'CleanAir'))
          saveIniSettings()
        elif action == 'setSpeed':
          state['speedMs'] = max(50, int(data.get('speedMs', 250)))
          appConfig['speedMs'] = state['speedMs']
          saveIniSettings()
        elif action == 'toggleAutoClean':
          cleanerConfig['autoCleanEnabled'] = bool(data.get('enabled', True))
          saveIniSettings()
        elif action == 'toggleAutoRetrain':
          appConfig['autoRetrainEnabled'] = bool(data.get('enabled', True))
          saveIniSettings()
        elif action == 'triggerClean':
          cleanAllCsvFiles()
    except Exception:
      pass
  cmdTask = asyncio.create_task(listenCommands())
  try:
    while True:
      if state['isPlaying']:
        activeAir = firebaseService.activeAir
        fbItem = firebaseService.getLatest(activeAir)
        if fbItem is None:
          state['cyclePoints'].clear()
          state['lastEMA1'] = None
          state['lastEMA2'] = None
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
            'prognostics': {'trajectoryPoints': [0.0, 0.0, 0.0, 0.0], 'maxProjectedS3': 0.0, 'isImminentEmergency': False, 'timeToEmergencySec': None},
            'latencyMs': 0.0,
            'frameIndex': 0,
            'totalFrames': 250,
            'mode': 'liveFirebase',
            'isPlaying': state['isPlaying'],
            'isCycleComplete': False,
            'cycleIndex': state['cycleIndex'],
            'cycleWave': None,
            'sensor': {'voltage1': 0.0, 'voltage2': 0.0, 'sensor1': 0, 'sensor2': 0, 'dacVoltage': 0.0, 'pulse': 0, 'point': 0, 'waveType': 0},
            'cron': {
              'retrainMinutes': appConfig['retrainMinutes'],
              'secondsUntilRetrain': max(0, int(appConfig['retrainMinutes'] * 60 - (time.time() - appConfig['lastRetrain']))),
              'autoRetrainEnabled': appConfig.get('autoRetrainEnabled', True),
              'cleanMinutes': cleanerConfig['cleanMinutes'],
              'secondsUntilClean': max(0, int(cleanerConfig['cleanMinutes'] * 60 - (time.time() - cleanerConfig['lastClean']))),
              'autoCleanEnabled': cleanerConfig.get('autoCleanEnabled', True),
              'totalCleanCycles': cleanerConfig['totalCleanCycles'],
              'totalRemovedCycles': cleanerConfig['totalRemovedCycles'],
              'isCleaning': cleanerConfig['isCleaning'],
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
        temp = float(fbItem.get('temperature', 28.5))
        hum = float(fbItem.get('humidity', 64.0))
        trueGas = fbItem.get('gasType', activeAir)
        truePPM = float(fbItem.get('ppm', 0.0))
        phase = 'LiveSensor'
        dacVoltage = float(fbItem.get('dacVoltage', fbItem.get('dac_voltage', 0.0)))
        pulse = int(fbItem.get('pulse', 0))
        point = int(fbItem.get('point', 0))
        waveType = int(fbItem.get('waveType', fbItem.get('wave_type', 0)))
        sensor1 = int(fbItem.get('sensor1', fbItem.get('raw_sensor1', 0)))
        sensor2 = int(fbItem.get('sensor2', fbItem.get('raw_sensor2', 0)))
        if state['lastEMA1'] is None:
          state['lastEMA1'] = voltage1
        else:
          state['lastEMA1'] = 0.2 * voltage1 + 0.8 * state['lastEMA1']
        if state['lastEMA2'] is None:
          state['lastEMA2'] = voltage2
        else:
          state['lastEMA2'] = 0.2 * voltage2 + 0.8 * state['lastEMA2']
        compFactor = 1.0 + 0.0035 * (temp - 25.0) + 0.0015 * (hum - 60.0)
        v1Comp = state['lastEMA1'] / compFactor
        v2Comp = state['lastEMA2'] / compFactor
        state['cyclePoints'][point] = (v1Comp, v2Comp)
        isCycleComplete = False
        if (point == 249) or (state['lastPoint'] >= 240 and point < 10):
          if len(state['cyclePoints']) >= 20:
            knownPts = sorted(state['cyclePoints'].keys())
            knownV1 = [state['cyclePoints'][p][0] for p in knownPts]
            knownV2 = [state['cyclePoints'][p][1] for p in knownPts]
            targetGrid = np.arange(250)
            v1Interp = [round(float(v), 4) for v in np.interp(targetGrid, knownPts, knownV1)]
            v2Interp = [round(float(v), 4) for v in np.interp(targetGrid, knownPts, knownV2)]
            cycleTensor = np.stack([v1Interp, v2Interp], axis=-1).astype(np.float32).reshape(1, 250, 2)
            if activeAir == 'MixedAir':
              inf = runInference(cycleTensor)
              state['cycleGas'] = inf['gas']
              state['cyclePpm'] = inf['estimatedppm']
              state['cycleConfidence'] = inf['confidence']
              state['cycleProbabilities'] = inf['probabilities']
              state['lastLatency'] = inf['latencyMs']
              if state['cycleGas'] == 'H2S':
                if state['cyclePpm'] >= 10.0 or v2Comp >= 0.95:
                  state['cycleRisk'] = 'Emergency'
                elif state['cyclePpm'] >= 5.0 or v2Comp >= 0.80:
                  state['cycleRisk'] = 'Hazardous'
                elif state['cyclePpm'] >= 1.0 or v2Comp >= 0.70:
                  state['cycleRisk'] = 'Warning'
                else:
                  state['cycleRisk'] = 'Normal'
              elif state['cycleGas'] == 'NH3':
                if state['cyclePpm'] >= 50.0 or v2Comp >= 1.20:
                  state['cycleRisk'] = 'Emergency'
                elif state['cyclePpm'] >= 25.0 or v2Comp >= 0.90:
                  state['cycleRisk'] = 'Hazardous'
                elif state['cyclePpm'] >= 10.0 or v2Comp >= 0.75:
                  state['cycleRisk'] = 'Warning'
                else:
                  state['cycleRisk'] = 'Normal'
              else:
                state['cycleRisk'] = 'Normal'
            else:
              state['cycleGas'] = activeAir
              state['cyclePpm'] = 0.0
              state['cycleConfidence'] = 100
              state['cycleProbabilities'] = {'CleanAir': 1.0 if activeAir == 'CleanAir' else 0.0, 'H2S': 1.0 if activeAir == 'H2S' else 0.0, 'NH3': 1.0 if activeAir == 'NH3' else 0.0}
              state['cycleRisk'] = 'Normal'
              state['lastLatency'] = 0.0
            state['cycleIndex'] += 1
            isCycleComplete = True
            state['lastCycleWave'] = {
              'cycleIndex': state['cycleIndex'],
              'timestamp': time.strftime('%I:%M:%S %p'),
              'channel': activeAir,
              'gas': state['cycleGas'],
              'ppm': state['cyclePpm'],
              'confidence': state['cycleConfidence'],
              'probabilities': state['cycleProbabilities'],
              'risk': state['cycleRisk'],
              'speedMs': state['speedMs'],
              'v1': v1Interp,
              'v2': v2Interp
            }
            state['cyclePoints'].clear()
        state['lastPoint'] = point
        if activeAir == 'MixedAir':
          gas = state['cycleGas']
          ppmVal = state['cyclePpm']
          risk = state['cycleRisk']
          confidence = state['cycleConfidence']
          probabilities = state['cycleProbabilities']
          latencyMs = state['lastLatency']
          recentV2 = [state['cyclePoints'][p][1] for p in sorted(state['cyclePoints'].keys())]
          feat = extractFeatures(recentV2)
          slopePerStep = feat[6] if feat else 0.0
          dt = state['speedMs'] / 1000.0
          slopeVal = slopePerStep / max(0.01, dt)
          traj = [round(float(v2Comp + slopeVal * s), 3) for s in [5, 10, 15, 20]]
          maxProj = max(v2Comp, *traj)
          isEmerg = (gas in ['H2S', 'NH3'] and maxProj >= 0.95) or (gas == 'H2S' and slopeVal > 0.008)
          tte = None
          if isEmerg:
            rate = max(0.002, slopeVal)
            tte = max(3, min(60, round((0.95 - v2Comp) / rate)))
        else:
          gas = activeAir
          ppmVal = 0.0
          risk = 'Normal'
          confidence = 100
          probabilities = {'CleanAir': 1.0 if activeAir == 'CleanAir' else 0.0, 'H2S': 1.0 if activeAir == 'H2S' else 0.0, 'NH3': 1.0 if activeAir == 'NH3' else 0.0}
          latencyMs = 0.0
          feat = None
          traj = [round(v2Comp, 3)] * 4
          maxProj = round(v2Comp, 3)
          isEmerg = False
          tte = None
        packet = {
          'online': True,
          'timestamp': time.strftime('%I:%M:%S %p'),
          's3': round(v2Comp, 4),
          's3Raw': round(voltage2, 4),
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
          'frameIndex': point,
          'totalFrames': 250,
          'mode': 'liveFirebase',
          'isPlaying': state['isPlaying'],
          'isCycleComplete': isCycleComplete,
          'cycleIndex': state['cycleIndex'],
          'cycleWave': state['lastCycleWave'] if isCycleComplete else None,
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
            'autoRetrainEnabled': appConfig.get('autoRetrainEnabled', True),
            'cleanMinutes': cleanerConfig['cleanMinutes'],
            'secondsUntilClean': max(0, int(cleanerConfig['cleanMinutes'] * 60 - (time.time() - cleanerConfig['lastClean']))),
            'autoCleanEnabled': cleanerConfig.get('autoCleanEnabled', True),
            'totalCleanCycles': cleanerConfig['totalCleanCycles'],
            'totalRemovedCycles': cleanerConfig['totalRemovedCycles'],
            'isCleaning': cleanerConfig['isCleaning'],
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