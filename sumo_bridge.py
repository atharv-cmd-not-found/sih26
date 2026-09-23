"""SUMO <-> CARLA traffic adapter for the Indian-road scenario."""

import os
import random
import math
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import carla


class SumoIndianTraffic:
    """Generate SUMO traffic from CARLA's active OpenDRIVE map and mirror it into CARLA."""

    _route_anchor = None

    def __init__(self, client, world, anchor_location=None, seed=26):
        self.client = client
        self.world = world
        self.anchor_location = anchor_location
        self.seed = seed
        self.sumo = None
        self.traci = None
        self.work_dir = None
        self.actors = {}
        self.pedestrian_actors = []
        self.started = False
        self.step_count = 0

    @staticmethod
    def _find_binary(name):
        configured = os.environ.get("SUMO_BINARY")
        if configured and Path(configured).exists():
            return configured
        found = shutil.which(name)
        if found:
            return found
        for install_dir in (
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Eclipse SUMO" / "bin",
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Eclipse" / "Sumo" / "bin",
        ):
            candidate = install_dir / f"{name}.exe"
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"{name} was not found. Install SUMO and add its bin directory to PATH, "
            "or set SUMO_BINARY to the executable path."
        )

    @staticmethod
    def _connected_routes(net_path, route_count=8):
        root = ET.parse(net_path).getroot()
        edges = {}
        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            if edge_id.startswith(":") or edge.get("function") == "internal":
                continue
            start = edge.get("from")
            end = edge.get("to")
            if edge_id and start and end:
                shape = edge.find("lane").get("shape", "") if edge.find("lane") is not None else ""
                first_point = shape.split(" ")[0].split(",") if shape else []
                x = float(first_point[0]) if len(first_point) == 2 else 0.0
                y = float(first_point[1]) if len(first_point) == 2 else 0.0
                edges[edge_id] = (start, end, x, y)

        next_edges = {}
        for edge_id, (_, end, _, _) in edges.items():
            choices = [candidate for candidate, (start, _, _, _) in edges.items() if start == end]
            if choices:
                next_edges[edge_id] = choices

        routes = []
        candidates = list(next_edges)
        if hasattr(SumoIndianTraffic, "_route_anchor") and SumoIndianTraffic._route_anchor is not None:
            anchor = SumoIndianTraffic._route_anchor
            candidates.sort(
                key=lambda edge_id: (edges[edge_id][2] - anchor.x) ** 2 + (edges[edge_id][3] - anchor.y) ** 2
            )
        random.seed(26)
        if SumoIndianTraffic._route_anchor is None:
            random.shuffle(candidates)
        for first in candidates:
            chain = [first]
            current = first
            for _ in range(5):
                choices = next_edges.get(current, [])
                if not choices:
                    break
                current = random.choice(choices)
                if current in chain:
                    break
                chain.append(current)
            if len(chain) >= 2:
                routes.append(" ".join(chain))
            if len(routes) >= route_count:
                break
        return routes

    def _write_scenario(self):
        self.work_dir = Path(tempfile.mkdtemp(prefix="lidforge_sumo_"))
        xodr_path = self.work_dir / "carla.xodr"
        net_path = self.work_dir / "carla.net.xml"
        route_path = self.work_dir / "indian_traffic.rou.xml"
        config_path = self.work_dir / "indian_traffic.sumocfg"

        xodr_path.write_text(self.world.get_map().to_opendrive(), encoding="utf-8")
        netconvert = self._find_binary("netconvert")
        subprocess.run(
            [
                netconvert,
                "--opendrive-files", str(xodr_path),
                "--proj.utm", "false",
                "--junctions.join", "true",
                "--ramps.guess", "true",
                "--output-file", str(net_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        SumoIndianTraffic._route_anchor = self.anchor_location
        routes = self._connected_routes(net_path)
        if not routes:
            raise RuntimeError("SUMO could not find connected drivable routes in the CARLA map.")

        route_xml = [
            '<routes>',
            '  <vType id="motorcycle" vClass="motorcycle" accel="3.0" decel="5.0" maxSpeed="16.0" minGap="0.8" sigma="0.6"/>',
            '  <vType id="auto" vClass="passenger" accel="2.2" decel="4.5" maxSpeed="13.0" minGap="1.0" sigma="0.7"/>',
            '  <vType id="car" vClass="passenger" accel="2.6" decel="4.5" maxSpeed="15.0" minGap="1.5" sigma="0.5"/>',
            '  <vType id="bus" vClass="bus" accel="1.3" decel="3.5" maxSpeed="11.0" minGap="2.5" sigma="0.6"/>',
            '  <vType id="truck" vClass="truck" accel="1.1" decel="3.0" maxSpeed="10.0" minGap="2.5" sigma="0.7"/>',
        ]
        for index, route in enumerate(routes):
            route_xml.append(f'  <route id="r{index}" edges="{route}"/>')

        flows = [("motorcycle", 300, 0), ("auto", 100, 1), ("car", 180, 2), ("bus", 35, 3), ("truck", 45, 4)]
        for vehicle_type, rate, route_index in flows:
            route_xml.append(
                f'  <flow id="flow_{vehicle_type}" type="{vehicle_type}" route="r{route_index % len(routes)}" '
                f'begin="0" end="3600" vehsPerHour="{rate}" departLane="best" departSpeed="max" '
                'reroute="true"/>'
            )
        route_xml.append("</routes>")
        route_path.write_text("\n".join(route_xml), encoding="utf-8")
        config_path.write_text(
            "<configuration>\n"
            f"  <input><net-file value=\"{net_path.name}\"/><route-files value=\"{route_path.name}\"/></input>\n"
            "  <time><begin value=\"0\"/><end value=\"3600\"/><step-length value=\"0.05\"/></time>\n"
            "  <processing><time-to-teleport value=\"-1\"/></processing>\n"
            "</configuration>\n",
            encoding="utf-8",
        )
        return config_path

    def start(self):
        try:
            import traci

            self.traci = traci
            sumo_binary = self._find_binary(os.environ.get("SUMO_GUI", "sumo"))
            config_path = self._write_scenario()
            traci.start(
                [
                    sumo_binary,
                    "-c", str(config_path),
                    "--seed", str(self.seed),
                    "--no-step-log", "true",
                    "--duration-log.disable", "true",
                    "--collision.action", "none",
                ],
                label=f"lidforge_{os.getpid()}",
            )
            self.started = True
            print("[✓] SUMO Indian-road traffic connected to CARLA.")
            return True
        except Exception as exc:
            print(f"[!] SUMO traffic unavailable: {exc}")
            self.close()
            return False

    def _blueprint_for(self, sumo_type):
        bp_lib = self.world.get_blueprint_library()
        if sumo_type == "motorcycle":
            candidates = list(bp_lib.filter("vehicle.*motorcycle*")) + list(bp_lib.filter("vehicle.yamaha.*"))
        elif sumo_type == "bus":
            candidates = list(bp_lib.filter("vehicle.*bus*"))
        elif sumo_type == "truck":
            candidates = list(bp_lib.filter("vehicle.carlamotors.carlacola"))
        else:
            candidates = list(bp_lib.filter("vehicle.*"))
        if not candidates:
            candidates = list(bp_lib.filter("vehicle.*"))
        return random.choice(candidates) if candidates else None

    def _transform(self, vehicle_id):
        x, y = self.traci.vehicle.getPosition(vehicle_id)
        angle = self.traci.vehicle.getAngle(vehicle_id)
        waypoint = self.world.get_map().get_waypoint(
            carla.Location(x=float(x), y=float(y)), project_to_road=True
        )
        z = waypoint.transform.location.z + 0.35 if waypoint else 0.35
        return carla.Transform(
            carla.Location(x=float(x), y=float(y), z=z),
            carla.Rotation(yaw=90.0 - float(angle)),
        )

    def spawn_jaywalkers(self, ego_vehicle, count=20):
        walker_bps = list(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        if not walker_bps:
            return

        ego_tf = ego_vehicle.get_transform()
        yaw = math.radians(ego_tf.rotation.yaw)
        forward = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
        lateral = carla.Vector3D(-math.sin(yaw), math.cos(yaw), 0.0)
        targets = []
        spawned = 0
        for _ in range(count):
            candidate = ego_tf.location + forward * random.uniform(10.0, 42.0) + lateral * random.uniform(-7.0, 7.0)
            waypoint = self.world.get_map().get_waypoint(candidate, project_to_road=True)
            if waypoint is None:
                navigation_location = self.world.get_random_location_from_navigation()
                if navigation_location is None:
                    continue
                candidate = navigation_location
            else:
                candidate = waypoint.transform.location
            location = carla.Location(x=candidate.x, y=candidate.y, z=candidate.z + 0.15)
            walker = self.world.try_spawn_actor(
                random.choice(walker_bps), carla.Transform(location, carla.Rotation(yaw=random.uniform(0.0, 360.0)))
            )
            if walker is None:
                continue
            controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller is None:
                walker.destroy()
                continue
            target = location + lateral * random.choice((-15.0, 15.0))
            self.pedestrian_actors.extend((controller, walker))
            targets.append((controller, target))
            spawned += 1

        for controller, target in targets:
            controller.start()
            controller.go_to_location(target)
            controller.set_max_speed(random.uniform(1.2, 1.8))
        print(f"[SUMO] Spawned {spawned} jaywalkers in CARLA.")

    def step(self):
        if not self.started:
            return
        try:
            self.traci.simulationStep()
            self.step_count += 1
            active_ids = set(self.traci.vehicle.getIDList())
            for vehicle_id in active_ids:
                transform = self._transform(vehicle_id)
                actor = self.actors.get(vehicle_id)
                if actor is None or not actor.is_alive:
                    blueprint = self._blueprint_for(self.traci.vehicle.getTypeID(vehicle_id))
                    if blueprint is None:
                        continue
                    actor = self.world.try_spawn_actor(
                        blueprint, transform
                    )
                    if actor is None:
                        continue
                    actor.set_simulate_physics(False)
                    self.actors[vehicle_id] = actor
                actor.set_transform(transform)

            for vehicle_id in set(self.actors) - active_ids:
                actor = self.actors.pop(vehicle_id)
                if actor.is_alive:
                    actor.destroy()
            if self.step_count == 1 or self.step_count % 100 == 0:
                print(f"[SUMO] vehicles={len(active_ids)} mirrored={len(self.actors)}")
        except Exception as exc:
            print(f"[!] SUMO step failed: {exc}")
            self.close()

    def close(self):
        for actor in reversed(self.pedestrian_actors):
            if actor.is_alive:
                actor.destroy()
        self.pedestrian_actors.clear()
        for actor in self.actors.values():
            if actor.is_alive:
                actor.destroy()
        self.actors.clear()
        if self.traci is not None:
            try:
                self.traci.close(False)
            except Exception:
                pass
        self.started = False
        self.sumo = None
        self.traci = None
        if self.work_dir is not None:
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = None
