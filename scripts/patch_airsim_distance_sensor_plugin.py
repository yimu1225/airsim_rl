#!/usr/bin/env python3
"""
Patch an AirSim UE4 plugin so distance sensors can be queried over RPC and
optionally drawn in UE4 with DrawDebugPoints.

Run this on every machine that has an AirSim UE4 project:

  python scripts/patch_airsim_distance_sensor_plugin.py --airsim-plugin /path/to/Plugins/AirSim

The script is intentionally idempotent. It creates .bak_distance_sensor_patch
backups before changing each file.
"""

import argparse
import re
import shutil
from pathlib import Path


BACKUP_SUFFIX = ".bak_distance_sensor_patch"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text_if_changed(path: Path, original: str, updated: str) -> bool:
    if original == updated:
        return False
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing AirSim plugin file: {path}")
    return path


def add_include(text: str, include_line: str, after_include: str) -> str:
    if include_line in text:
        return text
    if after_include not in text:
        raise RuntimeError(f"Could not find include anchor: {after_include}")
    return text.replace(after_include, after_include + "\n" + include_line, 1)


def insert_before(text: str, marker: str, block: str) -> str:
    if block.strip() in text:
        return text
    if marker not in text:
        raise RuntimeError(f"Could not find insertion marker: {marker[:80]}")
    return text.replace(marker, block.rstrip() + "\n\n" + marker, 1)


def insert_after(text: str, marker: str, block: str) -> str:
    if block.strip() in text:
        return text
    if marker not in text:
        raise RuntimeError(f"Could not find insertion marker: {marker[:80]}")
    return text.replace(marker, marker + "\n\n" + block.rstrip(), 1)


def replace_function_body(text: str, signature: str, new_body: str) -> str:
    start = text.find(signature)
    if start < 0:
        raise RuntimeError(f"Could not find function signature: {signature}")
    open_brace = text.find("{", start)
    if open_brace < 0:
        raise RuntimeError(f"Could not find function body for: {signature}")

    depth = 0
    close_brace = None
    for i in range(open_brace, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                close_brace = i
                break
    if close_brace is None:
        raise RuntimeError(f"Could not match braces for: {signature}")

    return text[:open_brace + 1] + "\n" + new_body.rstrip() + "\n" + text[close_brace:]


def patch_vehicle_api(root: Path) -> bool:
    path = require_file(root, "Source/AirLib/include/api/VehicleApiBase.hpp")
    original = read_text(path)
    text = add_include(
        original,
        '#include "sensors/distance/DistanceBase.hpp"',
        '#include "sensors/SensorCollection.hpp"',
    )
    method = r'''
    // Distance sensor APIs
    virtual DistanceBase::Output getDistanceSensorData(const std::string& distance_sensor_name) const
    {
        const DistanceBase* distance_sensor = nullptr;

        uint count_distance_sensors = getSensors().size(SensorBase::SensorType::Distance);
        for (uint i = 0; i < count_distance_sensors; i++)
        {
            const DistanceBase* current_distance_sensor =
                static_cast<const DistanceBase*>(getSensors().getByType(SensorBase::SensorType::Distance, i));
            if (current_distance_sensor != nullptr &&
                (current_distance_sensor->getName() == distance_sensor_name || distance_sensor_name == ""))
            {
                distance_sensor = current_distance_sensor;
                break;
            }
        }
        if (distance_sensor == nullptr)
            throw VehicleControllerException(Utils::stringf(
                "No distance sensor with name %s exist on vehicle", distance_sensor_name.c_str()));

        return distance_sensor->getOutput();
    }
'''
    text = insert_before(text, "    /************* battery info is propagated similar to collision info ********/", method)
    return write_text_if_changed(path, original, text)


def patch_rpc_adapters(root: Path) -> bool:
    path = require_file(root, "Source/AirLib/include/api/RpcLibAdapatorsBase.hpp")
    original = read_text(path)
    text = add_include(
        original,
        '#include "sensors/distance/DistanceBase.hpp"',
        '#include "api/WorldSimApiBase.hpp"',
    )
    struct = r'''
    struct DistanceSensorData {
        msr::airlib::real_T distance = 0;
        msr::airlib::real_T min_distance = 0;
        msr::airlib::real_T max_distance = 0;
        Pose relative_pose;

        MSGPACK_DEFINE_MAP(distance, min_distance, max_distance, relative_pose);

        DistanceSensorData()
        {}

        DistanceSensorData(const msr::airlib::DistanceBase::Output& s)
        {
            distance = s.distance;
            min_distance = s.min_distance;
            max_distance = s.max_distance;
            relative_pose = s.relative_pose;
        }

        msr::airlib::DistanceBase::Output to() const
        {
            msr::airlib::DistanceBase::Output d;
            d.distance = distance;
            d.min_distance = min_distance;
            d.max_distance = max_distance;
            d.relative_pose = relative_pose.to();
            return d;
        }
    };
'''
    marker = "\n};\n\n}} //namespace"
    text = insert_before(text, marker, struct)
    return write_text_if_changed(path, original, text)


def patch_rpc_client_hpp(root: Path) -> bool:
    path = require_file(root, "Source/AirLib/include/api/RpcLibClientBase.hpp")
    original = read_text(path)
    text = add_include(
        original,
        '#include "sensors/distance/DistanceBase.hpp"',
        '#include "api/WorldSimApiBase.hpp"',
    )
    decl = '    msr::airlib::DistanceBase::Output getDistanceSensorData(const std::string& distance_sensor_name = "", const std::string& vehicle_name = "") const;'
    text = insert_after(
        text,
        '    msr::airlib::LidarData getLidarData(const std::string& lidar_name = "", const std::string& vehicle_name = "") const;',
        decl,
    )
    return write_text_if_changed(path, original, text)


def patch_rpc_client_cpp(root: Path) -> bool:
    path = require_file(root, "Source/AirLib/src/api/RpcLibClientBase.cpp")
    original = read_text(path)
    impl = r'''
msr::airlib::DistanceBase::Output RpcLibClientBase::getDistanceSensorData(const std::string& distance_sensor_name, const std::string& vehicle_name) const
{
    return pimpl_->client.call("getDistanceSensorData", distance_sensor_name, vehicle_name).as<RpcLibAdapatorsBase::DistanceSensorData>().to();
}
'''
    marker = r'''msr::airlib::LidarData RpcLibClientBase::getLidarData(const std::string& lidar_name, const std::string& vehicle_name) const
{
    return pimpl_->client.call("getLidarData", lidar_name, vehicle_name).as<RpcLibAdapatorsBase::LidarData>().to();
}'''
    text = insert_after(original, marker, impl)
    return write_text_if_changed(path, original, text)


def patch_rpc_server_cpp(root: Path) -> bool:
    path = require_file(root, "Source/AirLib/src/api/RpcLibServerBase.cpp")
    original = read_text(path)
    bind = r'''
    pimpl_->server.bind("getDistanceSensorData", [&](const std::string& distance_sensor_name, const std::string& vehicle_name) -> RpcLibAdapatorsBase::DistanceSensorData {
        const auto& distance_sensor_data = getVehicleApi(vehicle_name)->getDistanceSensorData(distance_sensor_name);
        return RpcLibAdapatorsBase::DistanceSensorData(distance_sensor_data);
    });
'''
    marker = r'''    pimpl_->server.bind("getLidarData", [&](const std::string& lidar_name, const std::string& vehicle_name) -> RpcLibAdapatorsBase::LidarData {
        const auto& lidar_data = getVehicleApi(vehicle_name)->getLidarData(lidar_name);
        return RpcLibAdapatorsBase::LidarData(lidar_data);
    });'''
    text = insert_after(original, marker, bind)
    return write_text_if_changed(path, original, text)


def patch_airsim_settings(root: Path) -> bool:
    path = require_file(root, "Source/AirLib/include/common/AirSimSettings.hpp")
    original = read_text(path)
    text = original
    distance_struct = r'''struct DistanceSetting : SensorSetting {
        real_T min_distance = 20.0f / 100;   // meters
        real_T max_distance = 4000.0f / 100; // meters
        Vector3r position = VectorMath::nanVector();
        Rotation rotation = Rotation::nanRotation();
        bool draw_debug_points = false;
    };'''
    text = re.sub(
        r"struct DistanceSetting\s*:\s*SensorSetting\s*\{.*?\};",
        distance_struct,
        text,
        count=1,
        flags=re.DOTALL,
    )
    body = r'''        distance_setting.min_distance = settings_json.getFloat("MinDistance", distance_setting.min_distance);
        distance_setting.max_distance = settings_json.getFloat("MaxDistance", distance_setting.max_distance);
        distance_setting.draw_debug_points = settings_json.getBool("DrawDebugPoints", distance_setting.draw_debug_points);
        distance_setting.position = createVectorSetting(settings_json, distance_setting.position);
        distance_setting.rotation = createRotationSetting(settings_json, distance_setting.rotation);'''
    text = replace_function_body(text, "static void initializeDistanceSetting", body)
    return write_text_if_changed(path, original, text)


def patch_distance_params(root: Path) -> bool:
    path = require_file(root, "Source/AirLib/include/sensors/distance/DistanceSimpleParams.hpp")
    original = read_text(path)
    text = original
    if "bool draw_debug_points = false;" not in text:
        text = text.replace("    Pose relative_pose;\n", "    Pose relative_pose;\n    bool draw_debug_points = false;\n", 1)
    if "draw_debug_points = settings.draw_debug_points;" not in text:
        text = text.replace(
            "        max_distance = settings.max_distance;\n",
            "        max_distance = settings.max_distance;\n\n"
            "        relative_pose.position = settings.position;\n"
            "        if (std::isnan(relative_pose.position.x()))\n"
            "            relative_pose.position.x() = 0;\n"
            "        if (std::isnan(relative_pose.position.y()))\n"
            "            relative_pose.position.y() = 0;\n"
            "        if (std::isnan(relative_pose.position.z()))\n"
            "            relative_pose.position.z() = 0;\n\n"
            "        float pitch = !std::isnan(settings.rotation.pitch) ? settings.rotation.pitch : 0;\n"
            "        float roll = !std::isnan(settings.rotation.roll) ? settings.rotation.roll : 0;\n"
            "        float yaw = !std::isnan(settings.rotation.yaw) ? settings.rotation.yaw : 0;\n"
            "        relative_pose.orientation = VectorMath::toQuaternion(\n"
            "            Utils::degreesToRadians(pitch),\n"
            "            Utils::degreesToRadians(roll),\n"
            "            Utils::degreesToRadians(yaw));\n\n"
            "        draw_debug_points = settings.draw_debug_points;\n",
            1,
        )
    return write_text_if_changed(path, original, text)


def patch_unreal_distance_sensor(root: Path) -> bool:
    path = require_file(root, "Source/UnrealSensors/UnrealDistanceSensor.cpp")
    original = read_text(path)
    updated = r'''// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "UnrealDistanceSensor.h"
#include "AirBlueprintLib.h"
#include "common/Common.hpp"
#include "Async/Async.h"
#include "DrawDebugHelpers.h"
#include "NedTransform.h"

UnrealDistanceSensor::UnrealDistanceSensor(const AirSimSettings::DistanceSetting& setting,
    AActor* actor, const NedTransform* ned_transform)
    : DistanceSimple(setting), actor_(actor), ned_transform_(ned_transform)
{
}

msr::airlib::real_T UnrealDistanceSensor::getRayLength(const msr::airlib::Pose& pose)
{
    Vector3r start = pose.position;
    Vector3r end = start + VectorMath::rotateVector(VectorMath::front(), pose.orientation, true) * getParams().max_distance;

    FHitResult dist_hit = FHitResult(ForceInit);
    bool is_hit = UAirBlueprintLib::GetObstacle(actor_, ned_transform_->fromLocalNed(start), ned_transform_->fromLocalNed(end), dist_hit);
    float distance = is_hit ? dist_hit.Distance / 100.0f : getParams().max_distance;

    if (getParams().draw_debug_points)
    {
        const FVector start_ue = ned_transform_->fromLocalNed(start);
        const FVector end_ue = is_hit ? dist_hit.ImpactPoint : ned_transform_->fromLocalNed(end);
        AActor* actor = actor_;
        auto draw_ray = [actor, start_ue, end_ue, is_hit]() {
            if (!actor || !actor->GetWorld())
                return;

            DrawDebugLine(
                actor->GetWorld(),
                start_ue,
                end_ue,
                is_hit ? FColor::Red : FColor::Green,
                false,
                0.05f,
                0,
                2.0f
            );
            if (is_hit) {
                DrawDebugPoint(
                    actor->GetWorld(),
                    end_ue,
                    8.0f,
                    FColor::Yellow,
                    false,
                    0.05f
                );
            }
        };

        if (UAirBlueprintLib::IsInGameThread())
            draw_ray();
        else
            AsyncTask(ENamedThreads::GameThread, draw_ray);
    }

    return distance;
}
'''
    return write_text_if_changed(path, original, updated)


def normalize_plugin_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "AirSim" and (path / "Source").exists():
        return path
    candidate = path / "Plugins" / "AirSim"
    if (candidate / "Source").exists():
        return candidate.resolve()
    candidate = path / "Unreal" / "Plugins" / "AirSim"
    if (candidate / "Source").exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find Plugins/AirSim/Source under {path}. "
        "Pass the AirSim plugin directory or the UE project root."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch AirSim UE4 plugin for distance sensor RPC and UE4 debug rays.")
    parser.add_argument("--airsim-plugin", required=True, help="Path to Plugins/AirSim, or to a UE project containing Plugins/AirSim.")
    args = parser.parse_args()

    root = normalize_plugin_path(Path(args.airsim_plugin))
    patchers = [
        patch_vehicle_api,
        patch_rpc_adapters,
        patch_rpc_client_hpp,
        patch_rpc_client_cpp,
        patch_rpc_server_cpp,
        patch_airsim_settings,
        patch_distance_params,
        patch_unreal_distance_sensor,
    ]

    changed = []
    for patcher in patchers:
        if patcher(root):
            changed.append(patcher.__name__)

    if changed:
        print(f"Patched {root}")
        print("Changed steps:")
        for name in changed:
            print(f"  - {name}")
        print(f"Backups use suffix: {BACKUP_SUFFIX}")
    else:
        print(f"No changes needed; {root} already appears patched.")


if __name__ == "__main__":
    main()
