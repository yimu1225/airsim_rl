import argparse
import json


def build_distance_sensor_settings(
    count=36,
    prefix="DistanceSensor",
    start_index=0,
    min_distance=0.05,
    max_distance=40.0,
    z=0.0,
    draw_debug_points=False,
):
    sensors = {}
    for i in range(count):
        sensor_index = start_index + i
        yaw_deg = i * (360.0 / count)
        sensors[f"{prefix}{sensor_index}"] = {
            "SensorType": 5,
            "Enabled": True,
            "X": 0.0,
            "Y": 0.0,
            "Z": float(z),
            "Roll": 0.0,
            "Pitch": 0.0,
            "Yaw": yaw_deg,
            "MinDistance": float(min_distance),
            "MaxDistance": float(max_distance),
            "DrawDebugPoints": bool(draw_debug_points),
        }
    return {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "Vehicles": {
            "SimpleFlight": {
                "VehicleType": "SimpleFlight",
                "Sensors": sensors,
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate an AirSim settings.json with evenly spaced horizontal distance sensors."
    )
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--prefix", type=str, default="DistanceSensor")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--min_distance", type=float, default=0.05)
    parser.add_argument("--max_distance", type=float, default=40.0)
    parser.add_argument("--z", type=float, default=0.0, help="Sensor Z offset in the vehicle frame.")
    parser.add_argument("--draw_debug_points", action="store_true", help="Show distance sensor rays in UE4.")
    args = parser.parse_args()

    settings = build_distance_sensor_settings(
        count=args.count,
        prefix=args.prefix,
        start_index=args.start_index,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        z=args.z,
        draw_debug_points=args.draw_debug_points,
    )
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
