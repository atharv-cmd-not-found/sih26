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
        try:
            client = carla.Client("127.0.0.1", 2000)
            client.set_timeout(10.0)
            batch = [carla.command.DestroyActor(a) for a in GLOBAL_ACTORS if a is not None and a.is_alive]
            if batch:
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


def spawn_ambient_indian_traffic(world, traffic_manager, num_vehicles=28):
    """
    Two-phase vehicle spawner:
    Phase 1: Spawns vehicles and assigns TM autopilot.
    Phase 2: Synchronizes with master tick before configuring behavioral parameters safely.
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicles = []

    two_wheelers = (
        list(bp_lib.filter("vehicle.yamaha.*"))
        + list(bp_lib.filter("vehicle.vespa.*"))
        + list(bp_lib.filter("vehicle.kawasaki.*"))
        + list(bp_lib.filter("vehicle.harley-davidson.*"))
    )
    compacts = (
        list(bp_lib.filter("vehicle.audi.a2"))
        + list(bp_lib.filter("vehicle.nissan.micra"))
        + list(bp_lib.filter("vehicle.mini.cooperst"))
    )
    general = list(bp_lib.filter("vehicle.*"))

    v_count = 0
    for sp in spawn_points:
        if v_count >= num_vehicles:
            break
        roll = random.random()
        if roll < 0.65 and two_wheelers:
            bp = random.choice(two_wheelers)
        elif roll < 0.85 and compacts:
            bp = random.choice(compacts)
        else:
            bp = random.choice(general)

        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

        npc = world.try_spawn_actor(bp, sp)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(npc)
            v_count += 1

    print(f"[+] Spawned {len(vehicles)} vehicles. Awaiting server synchronization tick...")

    # Wait for master client to tick the simulation so TM registers new actors
    for _ in range(2):
        try:
            world.wait_for_tick(timeout=2.0)
        except Exception:
            time.sleep(0.05)

    # Phase 2: Configure Traffic Manager parameters safely
    configured = 0
    for npc in vehicles:
        if npc is not None and npc.is_alive:
            try:
                traffic_manager.distance_to_leading_vehicle(npc, 0.8)
                traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-20.0, 20.0))
                traffic_manager.ignore_vehicles_percentage(npc, 25.0)
                traffic_manager.ignore_walkers_percentage(npc, 15.0)
                traffic_manager.auto_lane_change(npc, True)
                traffic_manager.random_left_lanechange_percentage(npc, 40.0)
                traffic_manager.random_right_lanechange_percentage(npc, 40.0)
                configured += 1
            except Exception:
                # Shields against RPC lane-change errors on 2-wheelers without lateral controllers
                configured += 1

    print(f"[✓] Successfully configured {configured}/{len(vehicles)} vehicles in Traffic Manager.")
    return vehicles


def spawn_active_forward_crossers(world, ego_vehicle, num_pedestrians=20):
    """Spawns dynamic crossing pedestrians synchronized with the simulation clock."""
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    controller_bp = bp_lib.find("controller.ai.walker")
    actors = []
    walkers_and_targets = []

    ego_tf = ego_vehicle.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)
    fwd_vec = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    right_vec = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)

    for _ in range(num_pedestrians):
        w_bp = random.choice(walker_bps)
        dist_fwd = random.uniform(8.0, 45.0)
        dist_lat = random.uniform(-7.0, 7.0)
        loc = ego_tf.location + (fwd_vec * dist_fwd) + (right_vec * dist_lat)

        walker = world.try_spawn_actor(
            w_bp, carla.Transform(loc, carla.Rotation(yaw=random.uniform(0, 360)))
        )
        if walker is not None:
            ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if ctrl is not None:
                actors.extend([ctrl, walker])
                cross_target = loc + (right_vec * random.choice([-15.0, 15.0]))
                walkers_and_targets.append((ctrl, cross_target))

    # Wait for master tick before activating walker controllers
    for _ in range(2):
        try:
            world.wait_for_tick(timeout=2.0)
        except Exception:
            time.sleep(0.05)

    for ctrl, cross_target in walkers_and_targets:
        try:
            ctrl.start()
            ctrl.go_to_location(cross_target)
            ctrl.set_max_speed(random.uniform(1.2, 1.8))
        except Exception:
            pass

    print(f"[✓] Spawned {len(walkers_and_targets)} active crossing pedestrians.")
    return actors


def clear_intersection_pedestrians(world, actors):
    """Removes pedestrians trapped inside junctions to prevent gridlock."""
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
    client.set_timeout(30.0)
    world = client.get_world()

    # Connect to master Traffic Manager on port 8000
    traffic_manager = client.get_trafficmanager(8000)

    print("[+] Traffic Daemon Active. Awaiting ego vehicle from carla_test_bridge.py...")
    ego_vehicle = None
    while IS_RUNNING and ego_vehicle is None:
        actors = world.get_actors().filter("vehicle.tesla.model3")
        if actors:
            ego_vehicle = actors[0]
            print(f"[✓] Ego vehicle acquired: ID {ego_vehicle.id}")
            break
        time.sleep(0.5)

    if not IS_RUNNING or ego_vehicle is None:
        return

    # 1. Spawn Ambient Indian Traffic Fleet
    ambient_traffic = spawn_ambient_indian_traffic(world, traffic_manager, num_vehicles=28)
    GLOBAL_ACTORS.extend(ambient_traffic)

    # 2. Spawn Dynamic Jaywalkers
    crossers = spawn_active_forward_crossers(world, ego_vehicle, num_pedestrians=20)
    GLOBAL_ACTORS.extend(crossers)

    print("[+] Scenario generator running. Passive daemon active. (Press Ctrl+C to exit)")
    loop_count = 0
    while IS_RUNNING:
        time.sleep(0.5)  # Passive sleep: never invokes world.tick()
        loop_count += 1

        # Clear junction deadlocks periodically
        if loop_count % 4 == 0:
            clear_intersection_pedestrians(world, GLOBAL_ACTORS)

        # Replenish crossers as pedestrians complete paths
        if loop_count % 20 == 0 and ego_vehicle.is_alive:
            new_crossers = spawn_active_forward_crossers(world, ego_vehicle, num_pedestrians=10)
            GLOBAL_ACTORS.extend(new_crossers)


if __name__ == "__main__":
    main()