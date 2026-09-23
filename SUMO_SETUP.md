# SUMO + CARLA Indian traffic

The CARLA bridge now starts SUMO automatically and mirrors SUMO vehicles into CARLA. Those vehicles are rendered by the existing LiDAR, perception, tactical guidance, and OpenCV dashboard path.

## Windows setup

1. Install SUMO from https://sumo.dlr.de/docs/Installing/index.html.
2. Add the SUMO `bin` directory to PATH. It contains `sumo.exe` and `netconvert.exe`.
3. Install the Python client in this project's environment:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If SUMO is not on PATH, set its executable explicitly:

```powershell
$env:SUMO_BINARY = 'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe'
```

## Run

Start CARLA first, then run only the bridge:

```powershell
& .\.venv\Scripts\python.exe .\carla_test_bridge.py
```

The bridge exports the active CARLA map to OpenDRIVE, converts it to a SUMO network, generates Indian-style flows, and synchronizes them at 20 Hz. The generated traffic includes motorcycles, auto-rickshaw-like compact vehicles, cars, buses, and trucks.

To temporarily use the bridge without SUMO:

```powershell
$env:LIDFORGE_SUMO = '0'
& .\.venv\Scripts\python.exe .\carla_test_bridge.py
```

Do not run `traffic_generator.py` at the same time as SUMO mode, because it adds a second independent traffic population to CARLA.
