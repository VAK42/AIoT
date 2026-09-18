import os
import csv
import json
import time
import math
import threading
import urllib.error
import urllib.request
class FirebaseService:
  def __init__(self):
    self.dbUrls = {
      'CleanAir': 'https://enose-1aeb7-default-rtdb.asia-southeast1.firebasedatabase.app/sensor/latest.json?auth=qbtMtFdVKPlt5kBvQoD7ELUITrqs1qoPPuNmgQ0y',
      'H2S': 'https://enose-h2s-default-rtdb.asia-southeast1.firebasedatabase.app/sensor/latest.json?auth=qbtMtFdVKPlt5kBvQoD7ELUITrqs1qoPPuNmgQ0y',
      'NH3': 'https://enose-nh3-default-rtdb.asia-southeast1.firebasedatabase.app/sensor/latest.json?auth=qbtMtFdVKPlt5kBvQoD7ELUITrqs1qoPPuNmgQ0y',
      'MixedAir': 'https://enose-1aeb7-default-rtdb.asia-southeast1.firebasedatabase.app/sensor/latest.json?auth=qbtMtFdVKPlt5kBvQoD7ELUITrqs1qoPPuNmgQ0y'
    }
    self.activeAir = 'CleanAir'
    self.connectionStatus = {}
    self.ramBuffers = {g: [] for g in self.dbUrls.keys()}
    self.csvFiles = {g: f'data/{g}.csv' for g in self.dbUrls.keys()}
    self.latest = {g: None for g in self.dbUrls.keys()}
    self.lastFlush = time.time()
    self.flushInterval = 60
    self.running = True
    self.lock = threading.Lock()
    self.checkAllConnections()
    for g in self.dbUrls.keys():
      threading.Thread(target=self.streamGas, args=(g,), daemon=True).start()
    self.connThread = threading.Thread(target=self.connectionCheckLoop, daemon=True)
    self.connThread.start()
  def setAirType(self, airType):
    with self.lock:
      if airType in self.dbUrls:
        self.activeAir = airType
  def getLatest(self, airType = None):
    with self.lock:
      target = airType if airType in self.dbUrls else self.activeAir
      return self.latest.get(target)
  def checkConnection(self, gas):
    url = self.dbUrls.get(gas, '')
    if not url:
      return {'connected': False, 'code': 404, 'status': 'Offline - 404', 'url': '', 'latencyMs': 0.0}
    tStart = time.perf_counter()
    try:
      req = urllib.request.Request(url, headers={'Accept': 'application/json'})
      with urllib.request.urlopen(req, timeout=2.5) as resp:
        code = resp.getcode()
        latency = round((time.perf_counter() - tStart) * 1000, 1)
        return {'connected': code == 200, 'code': code, 'status': f'Connected - {code}' if code == 200 else f'HTTP {code}', 'url': url, 'latencyMs': latency}
    except urllib.error.HTTPError as err:
      latency = round((time.perf_counter() - tStart) * 1000, 1)
      return {'connected': False, 'code': err.code, 'status': f'Offline - {err.code}', 'url': url, 'latencyMs': latency}
    except Exception:
      latency = round((time.perf_counter() - tStart) * 1000, 1)
      return {'connected': False, 'code': 404, 'status': 'Offline - 404', 'url': url, 'latencyMs': latency}
  def checkAllConnections(self):
    results = {}
    for gas in self.dbUrls.keys():
      results[gas] = self.checkConnection(gas)
    with self.lock:
      self.connectionStatus = results
    return results
  def connectionCheckLoop(self):
    while self.running:
      self.checkAllConnections()
      time.sleep(60)
  def flushDisk(self):
    with self.lock:
      for gas, buf in self.ramBuffers.items():
        if len(buf) > 0:
          path = self.csvFiles[gas]
          exists = os.path.exists(path)
          keys = ['point', 'waveType', 'dacVoltage', 'pulse', 'rawSensor1', 'rawSensor2', 'sensor1', 'sensor2', 'voltage1', 'voltage2', 'gasType', 'ppm']
          with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not exists or os.path.getsize(path) == 0:
              writer.writeheader()
            for r in buf:
              row = {k: r.get(k, 0) for k in keys}
              writer.writerow(row)
          buf.clear()
      self.lastFlush = time.time()
  def parseGasPacket(self, rawData, gasName):
    v2 = float(rawData.get('voltage2', 0.0))
    v1 = float(rawData.get('voltage1', 0.0))
    pt = int(rawData.get('point', 0))
    pl = int(rawData.get('pulse', 0))
    dac = float(rawData.get('dac_voltage', 0.0))
    wt = int(rawData.get('wave_type', 0))
    s1 = int(rawData.get('sensor1', rawData.get('raw_sensor1', 0)))
    s2 = int(rawData.get('sensor2', rawData.get('raw_sensor2', 0)))
    ppmVal = float(rawData.get('ppm', 0.0))
    tNow = time.strftime('%Y-%m-%d %H:%M:%S')
    return {'point': pt, 'waveType': wt, 'dacVoltage': dac, 'pulse': pl, 'rawSensor1': s1, 'rawSensor2': s2, 'sensor1': s1, 'sensor2': s2, 'voltage1': v1, 'voltage2': v2, 'gasType': gasName, 'ppm': ppmVal, 'timestamp': tNow}
  def streamGas(self, gasName):
    while self.running:
      url = self.dbUrls.get(gasName, '')
      try:
        req = urllib.request.Request(url, headers={'Accept': 'text/event-stream'})
        with urllib.request.urlopen(req, timeout=30) as resp:
          for rawLine in resp:
            if not self.running:
              break
            line = rawLine.decode('utf-8', errors='ignore').strip()
            if line.startswith('data:'):
              jsonStr = line[5:].strip()
              if jsonStr and jsonStr != 'null':
                evt = json.loads(jsonStr)
                packetData = evt.get('data') if isinstance(evt, dict) and 'data' in evt else evt
                if isinstance(packetData, dict) and 'voltage2' in packetData:
                  item = self.parseGasPacket(packetData, gasName)
                  with self.lock:
                    self.latest[gasName] = item
                    self.ramBuffers[gasName].append(item)
                  if time.time() - self.lastFlush >= self.flushInterval:
                    self.flushDisk()
      except Exception:
        with self.lock:
          self.latest[gasName] = None
        time.sleep(5)
  def getStatus(self):
    with self.lock:
      return {
        'activeAir': self.activeAir,
        'latest': self.latest.get(self.activeAir),
        'ramCounts': {k: len(v) for k, v in self.ramBuffers.items()},
        'secondsUntilFlush': max(0, int(self.flushInterval - (time.time() - self.lastFlush))),
        'connections': self.connectionStatus
      }