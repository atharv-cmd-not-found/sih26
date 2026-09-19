import os
import sys
import time
import math
import random
import signal
import atexit
import warnings
import carla

warnings.filterwarnings("ignore")

GLOBAL_ACTORS = []
IS_RUNNING = True


def emergency_cleanup():
    global GLOBAL_ACTORS
    print("\n[+] Cleaning up traffic generator actors...")
    if GLOBAL_ACTORS:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
        batch = [carla.command.DestroyActor(a) for a in GLOBAL_ACTORS if a is not None]
        try:
            client.apply_batch_sync(batch, False)
            print(f"[✓] Destroyed {len(batch)} scenario actors.")
        except Exception as e:
            print(f"[!] Cleanup warning: {e}")
    print("[+] Traffic daemon terminated cleanly.")


def signal_handler(signum, frame):
    global IS_RUNNING
    print("\n[!] Stop signal received. Shutting down traffic generator...")
    IS_RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(emergency_cleanup)


def spawn_ambient_indian_traffic(world, traffic_manager, num_vehicles=24):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    actors = []

    two_wheelers = (
        list(bp_lib.filter("vehicle.yamaha.*"))
        + list(bp_lib.filter("vehicle.vespa.*"))
        + list(bp_lib.filter("vehicle.kawasaki.*"))
    )
    compacts = list(bp_lib.filter("vehicle.audi.a2")) + list(bp_lib.filter("vehicle.nissan.micra"))
    general = list(bp_lib.filter("vehicle.*"))

    for sp in spawn_points[:num_vehicles]:
        roll = random.random()
        bp = (
            random.choice(two_wheelers)
            if (roll < 0.60 and two_wheelers)
            else random.choice(compacts)
            if (roll < 0.85 and compacts)
            else random.choice(general)
        )

        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

        npc = world.try_spawn_actor(bp, sp)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.random_left_lanechange_percentage(npc, 35.0)
            traffic_manager.random_right_lanechange_percentage(npc, 35.0)
            traffic_manager.distance_to_leading_vehicle(npc, 1.2)
            traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-15.0, 25.0))
            actors.append(npc)

    print(f"[✓] Spawned {len(actors)} ambient Indian traffic vehicles.")
    return actors


def spawn_left_lane_traffic(world, traffic_manager, ego_vehicle):
    """Spawns vehicles directly in the left adjacent lane relative to the ego vehicle."""
    bp_lib = world.get_blueprint_library()
    actors = []

    two_wheelers = list(bp_lib.filter("vehicle.yamaha.*")) + list(bp_lib.filter("vehicle.vespa.*"))
    compacts = list(bp_lib.filter("vehicle.audi.a2")) + list(bp_lib.filter("vehicle.mini.cooperst"))
    general = list(bp_lib.filter("vehicle.*"))

    ego_tf = ego_vehicle.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)
    fwd = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    left = carla.Vector3D(math.sin(yaw_rad), -math.cos(yaw_rad), 0.0)

    # 1. Left-lane motorbike overtaking
    bike_bp = random.choice(two_wheelers) if two_wheelers else random.choice(general)
    loc_bike = ego_tf.location + (fwd * 14.0) + (left * 3.6)
    bike = world.try_spawn_actor(bike_bp, carla.Transform(loc_bike, ego_tf.rotation))
    if bike is not None:
        bike.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(bike, -20.0)
        actors.append(bike)

    # 2. Left-lane adjacent car cruising
    car_bp = random.choice(compacts) if compacts else random.choice(general)
    loc_car = ego_tf.location + (fwd * 28.0) + (left * 4.2)
    car = world.try_spawn_actor(car_bp, carla.Transform(loc_car, ego_tf.rotation))
    if car is not None:
        car.set_autopilot(True, traffic_manager.get_port())
        actors.append(car)

    print(f"[✓] Injected {len(actors)} vehicles into the adjacent left lane.")
    return actors


def spawn_pedestrian_swarms(world, ego_vehicle, num_pedestrians=20):
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    controller_bp = bp_lib.find("controller.ai.walker")
    actors = []

    ego_tf = ego_vehicle.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)
    fwd_vec = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    right_vec = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)

    for _ in range(num_pedestrians):
        w_bp = random.choice(walker_bps)
        dist = random.uniform(8.0, 45.0)
        lat = random.uniform(-6.0, 6.0)
        loc = ego_tf.location + (fwd_vec * dist) + (right_vec * lat)

        walker = world.try_spawn_actor(
            w_bp, carla.Transform(loc, carla.Rotation(yaw=random.uniform(0, 360)))
        )
        if walker is not None:
            ctrl = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                ctrl.go_to_location(dest)
                ctrl.set_max_speed(random.uniform(1.0, 1.6))
            actors.extend([ctrl, walker])

    print(f"[✓] Spawned {num_pedestrians} active dynamic pedestrians.")
    return actors


def clear_intersection_pedestrians(world, actors):
    try:
        map_ref = world.get_map()
        for a in actors:
            if a is not None and a.is_alive and isinstance(a, carla.Walker):
                wp = map_ref.get_waypoint(a.get_location(), project_to_road=True)
                if wp and wp.is_junction:
                    a.destroy()
    except Exception:
        pass


def main():
    global GLOBAL_ACTORS, IS_RUNNING

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(15.0)
    world = client.get_world()

    # Reuse the Traffic Manager initialized by the master script on port 8000
    traffic_manager = client.get_trafficmanager(8000)

    print("[+] Waiting for ego vehicle from carla_test_bridge.py...")
    ego_vehicle = None
    while IS_RUNNING and ego_vehicle is None:
        actors = world.get_actors().filter("vehicle.tesla.model3")
        if actors:
            ego_vehicle = actors[0]
            print(f"[✓] Ego vehicle discovered: {ego_vehicle.id}")
            break
        time.sleep(0.5)

    if not IS_RUNNING or ego_vehicle is None:
        return

    # Spawn scenarios
    ambient_veh = spawn_ambient_indian_traffic(world, traffic_manager, num_vehicles=24)
    GLOBAL_ACTORS.extend(ambient_veh)

    left_veh = spawn_left_lane_traffic(world, traffic_manager, ego_vehicle)
    GLOBAL_ACTORS.extend(left_veh)

    pedestrians = spawn_pedestrian_swarms(world, ego_vehicle, num_pedestrians=20)
    GLOBAL_ACTORS.extend(pedestrians)

    print("[+] Traffic daemon running. (Press Ctrl+C to stop)")
    counter = 0
    while IS_RUNNING:
        time.sleep(0.5)
        counter += 1
        if counter % 4 == 0:
            clear_intersection_pedestrians(world, GLOBAL_ACTORS)


if __name__ == "__main__":
    main()