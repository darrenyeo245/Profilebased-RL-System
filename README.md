# Profilbasiertes Reinforcement Learning System

Dieses Projekt stellt eine dynamische, profilbasierte Reinforcement-Learning-Umgebung für das MediaIP Lab bereit. Das System kann zur Laufzeit Konfigurationen laden, daraus eine Gymnasium-Umgebung erzeugen, ein PPO-Modell mit `stable-baselines3` trainieren und trainierte Modelle später für Inference (den Live-Einsatz) einsetzen.

Die Kommunikation mit dem Labor erfolgt über OSC. Konfigurationsdateien, Reward-Skripte und gespeicherte Modelle liegen nicht fest im Projekt, sondern werden aus einem externen GitLab-Repository (`mediaip-storage`) geladen und dorthin gespeichert.

## Grundidee

Das System trennt drei Dinge klar voneinander:

- **Pattern** definieren, welche OSC-Signale gelesen und welche OSC-Signale gesendet werden.
- **Reward-Skripte** bewerten, ob die gesendeten Aktionen gut waren.
- **Setups** verbinden Pattern, Reward, Reward-Konfiguration und Trainingsparameter.

Dadurch kann das RL-System für unterschiedliche Laboraufgaben genutzt werden, ohne den Code jedes Mal anpassen zu müssen. Neue Trainingsszenarien entstehen durch neue JSON-Konfigurationen und Reward-Skripte im externen Storage-Repository.

## Externes GitLab-Repository

Das externe Repository wird über die GitLab API angesprochen. Es wird kein vollständiger Git-Clone benötigt. Das System lädt gezielt nur die Dateien, die für ein Setup oder ein Modell gebraucht werden.

Die erwartete Struktur im Storage-Repository sieht wie folgt aus:

```text
local_rl-system/
  models/
    model_<id>/
      model_<id>.zip
      manifest.json
      pattern_<id>.json
      reward_<id>.py
      setup_<id>.json
  patterns/
    pattern_<id>.json
  rewards/
    reward_<id>.py
  setups/
    setup_<id>.json
```

Der Name des Root-Verzeichnisses, die GitLab-URL, Projekt-ID, Branch und der Zugriffstoken werden über Umgebungsvariablen in der .env gesetzt.

## Aufbau und Funktion der Konfigurationsdateien

### `pattern_<id>.json`

Ein Pattern beschreibt die OSC-Schnittstelle einer Aufgabe. Es enthält:

- Observation-Signale, die das RL-System empfängt
- Action-Signale, die das RL-System sendet
- OSC-Adressen
- Vektorgrößen
- minimale und maximale Wertebereiche

Aus diesen Angaben werden automatisch der Observation Space und der Action Space für die Gymnasium-Umgebung erzeugt.

Beispiel für ein Pattern:
```python
{
    "id": "1",
    "name": "spotlights_follow_actor",
    "description": "OSC-Pattern für Actor-Position und Spotlight-Zielpositionen.",
    "observations": [
        {
            "name": "actor_position",
            "address": "/adm/obj/101/xyz",
            "size": 3,
            "low": -1.0,
            "high": 1.0
        }
    ],
    "actions": [
        {
            "name": "spot_1_target",
            "address": "/adm/obj/1/xyz",
            "size": 3,
            "low": -1.0,
            "high": 1.0
        }
    ]
}
```

### `reward_<id>.py`

Ein Reward-Skript berechnet aus den aktuellen OSC-Informationen einen numerischen Reward. Dieser Reward sagt dem Lernalgorithmus, ob eine Aktion gut oder schlecht war.

Die Reward-Funktion arbeitet mit den benannten Signalen aus dem `info`-Objekt. Dort stehen unter anderem die empfangenen Observations und die gesendeten Actions.

Beispiel für ein Reward-Skript:

```python
import numpy as np

class RewardFunction:

    def __init__(self, config=None):
        self.config = config or {}
        self.input = self.config.get("input")
        self.output = self.config.get("output")
        self.objective = self.config.get("objective")
        self.max_distance = float(self.config.get("max_distance", 3.4641016151)) # maximale Distanz im 3D-Raum von -1 bis 1

    def reset(self):
        pass

    def compute(self, info):
        input_value = np.asarray(info["observations"][self.input], dtype=np.float32)
        output_value = np.asarray(info["actions"][self.output], dtype=np.float32)

        distance = float(np.linalg.norm(input_value - output_value))
        normalized_distance = min(distance / self.max_distance, 1.0)

        if self.objective == "maximize_distance":
            return normalized_distance
        if self.objective == "minimize_distance":
            return 1.0 - normalized_distance
        raise ValueError(f"Unknown objective: {self.objective}")
    
def create_reward(config=None):
    return RewardFunction(config)
```

Diese berechnet einen Reward basierend auf der Distanz zwischen einer Beobachtung und einer Aktion. 
Je nach Zielsetzung kann die Distanz maximiert oder minimiert werden.

### `setup_<id>.json`

Ein Setup verbindet ein Pattern mit einem Reward und den Trainingsparametern. Es legt fest:

- welches Pattern verwendet wird ("pattern_id")
- welcher Reward geladen wird ("reward_id")
- welche Reward-Konfiguration gilt ("reward_config")
- wie lange trainiert wird ("total_timesteps")
- wie lang eine Episode maximal ist ("max_steps")

Ein Setup ist damit die zentrale Beschreibung einer Trainingsaufgabe.

Beispiel für ein Setup:

```python
{
  "setup_id": "1",
  "name": "follow_actor",
  "description": "Scheinwerfer sollen auf die Actor-Position zeigen.",
  "pattern_id": "1",
  "reward_id": "1",
  "reward_config": {
    "input": "actor_position",
    "output": "spot_1_target",
    "objective": "minimize_distance",
    "max_distance": 3.4641016151
  },
  "training": {
    "algorithm": "ppo",
    "total_timesteps": 10000,
    "max_steps": 500
  }
}
```

### Modellbundle

Beim Speichern eines Modells wird ein Modellbundle im externen Repository abgelegt:

```text
models/model_<id>/
  model_<id>.zip
  manifest.json
  setup_<setup_id>.json
  pattern_<pattern_id>.json
  reward_<reward_id>.py
```

Das Modell selbst liegt als `.zip`-Datei vor. Zusätzlich werden die zugehörigen Setup-, Pattern- und Reward-Dateien mitgespeichert, damit nachvollziehbar bleibt, womit das Modell erzeugt wurde.

Die `manifest.json` beschreibt das Bundle. Sie enthält unter anderem:

- `model_id`
- `setup_id`
- `pattern_id`
- `reward_id`
- Dateinamen des Modells und der Konfigurationen
- Erstellungszeitpunkt
- Trainingsalgorithmus
- Trainingsparameter
- Reward-Konfiguration

Wenn ein Modell unter einer bereits vorhandenen `model_id` gespeichert wird, werden die vorhandenen Dateien im aktuellen GitLab-Branch aktualisiert. Die alte Version bleibt nur noch über die GitLab-Commit-Historie nachvollziehbar.

## Laufzeitverhalten

Beim Start des Systems werden `.env` und `config/system_config.json` geladen. Danach startet ein OSC-Server, der Steuerbefehle und Laborsignale empfängt.

Typischer Ablauf:

```text
1. Setup laden (rl/set/config/setup/<id>)
2. Training starten (rl/start/training)
3. OSC-Observations empfangen 
4. Actions per OSC senden 
5. Training stoppen (rl/stop/training oder automatisch nach total_timesteps)
6. Modell lokal speichern (wird automatisch unter runtime/models/current_model.zip abgelegt)
7. Modell in mediaip-storage speichern (rl/savemodel/<id>)
8. Modell später für Inference laden (rl/loadmodel/<id>)
9. Inference starten (rl/start/inference)
```

Beim Training wird nach einem Reset zuerst auf die erste neue Observation gewartet. Erst danach wird die erste Action berechnet und gesendet. Danach kann das Training mit dem zuletzt bekannten Zustand weiterlaufen.

Bei der Inference wird ein geladenes Modell regelmäßig mit der aktuellen Observation aufgerufen. Die vorhergesagte Action wird per OSC an die im Pattern definierte Action-Adresse gesendet.

## OSC-Befehle

Das System wird über OSC-Befehle gesteuert:

```text
/rl/set/config/setup/<id>
```

Lädt `setup_<id>.json` aus dem externen Storage und dazu das referenzierte Pattern und Reward-Skript.

```text
/rl/start/training
```

Startet ein Training mit dem aktuell geladenen Setup.

```text
/rl/stop/training
```

Stoppt das laufende Training sauber und speichert das aktuelle Modell lokal unter `runtime/models/current_model.zip`.

```text
/rl/savemodel/<id>
```

Speichert das lokale Modell als Bundle im externen GitLab-Repository unter `models/model_<id>/`.

```text
/rl/loadmodel/<id>
```

Lädt ein gespeichertes Modellbundle aus dem externen Repository. Für Inference werden Modell und Pattern verwendet.

```text
/rl/start/inference
```

Startet die Inference mit dem geladenen Modell.

```text
/rl/stop/inference
```

Stoppt die laufende Inference.

```text
/rl/status
```

Gibt den aktuellen Systemzustand zurück, zum Beispiel geladenes Setup, geladenes Modell, Trainingsstatus, Inference-Status und letzte Fehlermeldung.

## Lokale Konfiguration

Die lokale Systemkonfiguration liegt in:

```text
config/system_config.json
```

Sie definiert:

- maximale Größen für Observation- und Action-Spaces
- lokale Runtime-Pfade
- Inference-Takt
- ob Inference deterministisch ausgeführt wird

Die Verbindung zum externen GitLab-Storage und die OSC-Netzwerkeinstellungen werden über `.env` gesetzt.

Wichtige Umgebungsvariablen:

```text
MEDIAIP_STORAGE_GITLAB_URL
MEDIAIP_STORAGE_PROJECT_ID
MEDIAIP_STORAGE_BRANCH
MEDIAIP_STORAGE_ROOT
MEDIAIP_STORAGE_TOKEN

RASPI_HOST
RASPI_PORT
BROADCAST_IP
BROADCAST_PORT
```

## Lokaler Runtime-Cache

Geladene Dateien und lokale Modelle werden unter `runtime/` abgelegt. Dieser Ordner ist ein Cache und gehört nicht in die Versionsverwaltung.

Wichtige lokale Pfade:

```text
runtime/mediaip_storage/
runtime/models/current_model.zip
```

`current_model.zip` ist das lokale Arbeitsmodell. Es wird beim Training aktualisiert und kann anschließend mit `/rl/savemodel/<id>` in das externe Repository übertragen werden.

## Installation

Die benötigten Python Packages stehen in `requirements.txt`:

```text
gymnasium
numpy
python-osc
stable-baselines3
torch
python-dotenv
```
Typischer Start:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

Danach kann das System über OSC-Befehle gesteuert werden.

## Hinweise

- Ein geladenes Setup ist Voraussetzung für Training.
- Ein geladenes Modell ist Voraussetzung für Inference.
- Das Pattern im Modellbundle muss zu den Modell-Spaces passen.
- Für Training muss das Reward-Skript zur aktuellen Reward-Schnittstelle passen.
- Für reine Inference wird der Reward nicht berechnet.
- Wenn ein Modell unter derselben `model_id` gespeichert wird, wird der aktuelle Stand im GitLab-Repository überschrieben.
