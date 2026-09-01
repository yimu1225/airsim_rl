#!/usr/bin/env python3
"""Patch the ST UE4 project with persistent in-engine trajectory RPCs.

The patch extends the already project-specific ``simSwitchToTopDownCamera``
hook with three sibling RPCs:

* ``simClearTrajectory``
* ``simAppendTrajectoryPoint``
* ``simFinalizeTrajectory``

Trajectory points are read from the live UE4 vehicle Actor, so the rendered
line does not depend on Python-to-Unreal coordinate conversion.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PATCH_MARKERS = {
    "Source/ST/STTopDownCamera.h": "AppendTrajectoryPoint",
    "Source/ST/STTopDownCamera.cpp": "ASTTopDownCamera::AppendTrajectoryPoint",
    "Plugins/AirSim/Source/AirLib/include/api/WorldSimApiBase.hpp": "appendTrajectoryPoint",
    "Plugins/AirSim/Source/WorldSimApi.h": "appendTrajectoryPoint() override",
    "Plugins/AirSim/Source/WorldSimApi.cpp": "WorldSimApi::appendTrajectoryPoint",
    "Plugins/AirSim/Source/AirLib/src/api/RpcLibServerBase.cpp": "simAppendTrajectoryPoint",
}

GOAL_PATCH_MARKERS = {
    "Source/ST/STTopDownCamera.h": "SetTrajectoryGoalMarker",
    "Source/ST/STTopDownCamera.cpp": "ASTTopDownCamera::SetTrajectoryGoalMarker",
    "Plugins/AirSim/Source/AirLib/include/api/WorldSimApiBase.hpp": "setTrajectoryGoal",
    "Plugins/AirSim/Source/WorldSimApi.h": "setTrajectoryGoal",
    "Plugins/AirSim/Source/WorldSimApi.cpp": "WorldSimApi::setTrajectoryGoal",
    "Plugins/AirSim/Source/AirLib/src/api/RpcLibServerBase.cpp": "simSetTrajectoryGoal",
}

SOLID_GOAL_PATCH_MARKERS = {
    "Source/ST/STTopDownCamera.h": "UpdateTrajectoryGoalMarker",
    "Source/ST/STTopDownCamera.cpp": "ASTTopDownCamera::UpdateTrajectoryGoalMarker",
}


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one patch anchor in {path}, found {count}: {old[:80]!r}"
        )
    return text.replace(old, new, 1)


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + ".bak_trajectory_renderer")
    if not backup.exists():
        shutil.copy2(path, backup)


def _upgrade_persistent_depth_priority(root: Path) -> bool:
    """Move persistent trajectory primitives out of the one-frame foreground batch."""

    path = root / "Source/ST/STTopDownCamera.cpp"
    text = path.read_text(encoding="utf-8")
    old_line = """\t\ttrue,
\t\t-1.0f,
\t\t1,
\t\tFMath::Max(0.1f, TrajectoryThickness));"""
    new_line = old_line.replace("\n\t\t1,", "\n\t\t0,")
    old_start = (
        "FColor::Green, true, -1.0f, 1, 4.0f);"
    )
    old_end = "FColor::Blue, true, -1.0f, 1, 4.0f);"
    if old_line not in text and old_start not in text and old_end not in text:
        return False

    text = text.replace(old_line, new_line, 1)
    text = text.replace(old_start, "FColor::Green, true, -1.0f, 0, 4.0f);", 1)
    text = text.replace(old_end, "FColor::Blue, true, -1.0f, 0, 4.0f);", 1)
    path.write_text(text, encoding="utf-8", newline="")
    return True


def _upgrade_goal_marker(root: Path) -> bool:
    """Add a distinct red marker for the configured local-NED navigation goal."""

    changed = False

    path = root / "Source/ST/STTopDownCamera.h"
    text = path.read_text(encoding="utf-8")
    if GOAL_PATCH_MARKERS["Source/ST/STTopDownCamera.h"] not in text:
        text = _replace_once(
            text,
            """\tfloat TrajectoryMarkerRadiusMeters;

\t/** Ignore vehicle movements smaller than this spacing. */""",
            """\tfloat TrajectoryMarkerRadiusMeters;

\t/** Radius of the configured navigation-goal marker. */
\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ST|Trajectory", meta = (ClampMin = "0.01"))
\tfloat TrajectoryGoalMarkerRadiusMeters;

\t/** Colour of the configured navigation-goal marker. */
\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ST|Trajectory")
\tFLinearColor TrajectoryGoalColor;

\t/** Ignore vehicle movements smaller than this spacing. */""",
            path=path,
        )
        text = _replace_once(
            text,
            """\tUFUNCTION(BlueprintCallable, Category = "ST|Trajectory")
\tvoid FinalizeTrajectory();

private:""",
            """\tUFUNCTION(BlueprintCallable, Category = "ST|Trajectory")
\tvoid FinalizeTrajectory();

\t/** Set and draw the configured navigation goal in UE4 world coordinates. */
\tUFUNCTION(BlueprintCallable, Category = "ST|Trajectory")
\tvoid SetTrajectoryGoalMarker(FVector GoalWorldPosition);

private:""",
            path=path,
        )
        text = _replace_once(
            text,
            """\tFVector LastTrajectoryPoint;
\tbool bHasTrajectoryPoint;
\tint32 TrajectoryPointCount;""",
            """\tFVector LastTrajectoryPoint;
\tFVector TrajectoryGoalPoint;
\tbool bHasTrajectoryPoint;
\tbool bHasTrajectoryGoal;
\tint32 TrajectoryPointCount;""",
            path=path,
        )
        path.write_text(text, encoding="utf-8", newline="")
        changed = True

    path = root / "Source/ST/STTopDownCamera.cpp"
    text = path.read_text(encoding="utf-8")
    if GOAL_PATCH_MARKERS["Source/ST/STTopDownCamera.cpp"] not in text:
        text = _replace_once(
            text,
            """\t, TrajectoryMarkerRadiusMeters(0.35f)
\t, TrajectoryMinimumPointSpacingMeters(0.03f)""",
            """\t, TrajectoryMarkerRadiusMeters(0.35f)
\t, TrajectoryGoalMarkerRadiusMeters(0.25f)
\t, TrajectoryGoalColor(FLinearColor::Red)
\t, TrajectoryMinimumPointSpacingMeters(0.03f)""",
            path=path,
        )
        text = _replace_once(
            text,
            """\t, LastTrajectoryPoint(FVector::ZeroVector)
\t, bHasTrajectoryPoint(false)
\t, TrajectoryPointCount(0)""",
            """\t, LastTrajectoryPoint(FVector::ZeroVector)
\t, TrajectoryGoalPoint(FVector::ZeroVector)
\t, bHasTrajectoryPoint(false)
\t, bHasTrajectoryGoal(false)
\t, TrajectoryPointCount(0)""",
            path=path,
        )
        text = _replace_once(
            text,
            """\tLastTrajectoryPoint = FVector::ZeroVector;
\tbHasTrajectoryPoint = false;
\tTrajectoryPointCount = 0;
}

void ASTTopDownCamera::AppendTrajectoryPoint()""",
            """\tLastTrajectoryPoint = FVector::ZeroVector;
\tTrajectoryGoalPoint = FVector::ZeroVector;
\tbHasTrajectoryPoint = false;
\tbHasTrajectoryGoal = false;
\tTrajectoryPointCount = 0;
}

void ASTTopDownCamera::SetTrajectoryGoalMarker(FVector GoalWorldPosition)
{
\tif (GetWorld() == nullptr)
\t{
\t\treturn;
\t}

\tTrajectoryGoalPoint = GoalWorldPosition
\t\t+ FVector(0.0f, 0.0f, FMath::Max(0.0f, TrajectoryVerticalOffsetMeters) * CentimetresPerMetre);
\tbHasTrajectoryGoal = true;
\tconst float GoalMarkerRadius = FMath::Max(0.01f, TrajectoryGoalMarkerRadiusMeters)
\t\t* CentimetresPerMetre;
\tDrawDebugSphere(
\t\tGetWorld(),
\t\tTrajectoryGoalPoint,
\t\tGoalMarkerRadius,
\t\t24,
\t\tTrajectoryGoalColor.ToFColor(true),
\t\ttrue,
\t\t-1.0f,
\t\t0,
\t\t5.0f);
}

void ASTTopDownCamera::AppendTrajectoryPoint()""",
            path=path,
        )
        text = _replace_once(
            text,
            """\tconst float MarkerRadius = FMath::Max(0.01f, TrajectoryMarkerRadiusMeters)
\t\t* CentimetresPerMetre;
\tDrawDebugSphere(GetWorld(), TrajectoryStartPoint""",
            """\tconst float MarkerRadius = FMath::Max(0.01f, TrajectoryMarkerRadiusMeters)
\t\t* CentimetresPerMetre;
\tif (bHasTrajectoryGoal)
\t{
\t\tconst float GoalMarkerRadius = FMath::Max(0.01f, TrajectoryGoalMarkerRadiusMeters)
\t\t\t* CentimetresPerMetre;
\t\tDrawDebugSphere(
\t\t\tGetWorld(),
\t\t\tTrajectoryGoalPoint,
\t\t\tGoalMarkerRadius,
\t\t\t24,
\t\t\tTrajectoryGoalColor.ToFColor(true),
\t\t\ttrue,
\t\t\t-1.0f,
\t\t\t0,
\t\t\t5.0f);
\t}
\tDrawDebugSphere(GetWorld(), TrajectoryStartPoint""",
            path=path,
        )
        path.write_text(text, encoding="utf-8", newline="")
        changed = True

    small_replacements = {
        "Plugins/AirSim/Source/AirLib/include/api/WorldSimApiBase.hpp": (
            "    virtual bool appendTrajectoryPoint() { return false; }\n",
            "    virtual bool appendTrajectoryPoint() { return false; }\n"
            "    virtual bool setTrajectoryGoal(const Vector3r& goal_ned) { return false; }\n",
        ),
        "Plugins/AirSim/Source/WorldSimApi.h": (
            "    virtual bool appendTrajectoryPoint() override;\n",
            "    virtual bool appendTrajectoryPoint() override;\n"
            "    virtual bool setTrajectoryGoal(const Vector3r& goal_ned) override;\n",
        ),
        "Plugins/AirSim/Source/AirLib/src/api/RpcLibServerBase.cpp": (
            """    pimpl_->server.bind("simAppendTrajectoryPoint", [&]() -> bool {
        return getWorldSimApi()->appendTrajectoryPoint();
    });
""",
            """    pimpl_->server.bind("simAppendTrajectoryPoint", [&]() -> bool {
        return getWorldSimApi()->appendTrajectoryPoint();
    });
    pimpl_->server.bind("simSetTrajectoryGoal", [&](double x, double y, double z) -> bool {
        return getWorldSimApi()->setTrajectoryGoal(Vector3r(
            static_cast<real_T>(x),
            static_cast<real_T>(y),
            static_cast<real_T>(z)));
    });
""",
        ),
    }
    for relative, (old, new) in small_replacements.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        if GOAL_PATCH_MARKERS[relative] in text:
            continue
        path.write_text(
            _replace_once(text, old, new, path=path),
            encoding="utf-8",
            newline="",
        )
        changed = True

    path = root / "Plugins/AirSim/Source/WorldSimApi.cpp"
    text = path.read_text(encoding="utf-8")
    if GOAL_PATCH_MARKERS["Plugins/AirSim/Source/WorldSimApi.cpp"] not in text:
        text = _replace_once(
            text,
            '#include "EngineUtils.h"\n',
            '#include "EngineUtils.h"\n#include "PawnSimApi.h"\n',
            path=path,
        )
        text = _replace_once(
            text,
            """        return false;
    }
}

bool WorldSimApi::switchToTopDownCamera()""",
            """        return false;
    }

    bool setSTTopDownCameraGoal(UWorld* World, const FVector& GoalWorldPosition)
    {
        if (World == nullptr)
        {
            return false;
        }
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            AActor* Actor = *It;
            if (Actor == nullptr || !Actor->GetName().StartsWith(TEXT("STTopDownCamera")))
            {
                continue;
            }
            UFunction* Function = Actor->FindFunction(FName(TEXT("SetTrajectoryGoalMarker")));
            if (Function == nullptr)
            {
                return false;
            }
            struct FGoalMarkerParameters
            {
                FVector GoalWorldPosition;
            };
            FGoalMarkerParameters Parameters{GoalWorldPosition};
            Actor->ProcessEvent(Function, &Parameters);
            return true;
        }
        return false;
    }
}

bool WorldSimApi::switchToTopDownCamera()""",
            path=path,
        )
        text = _replace_once(
            text,
            """bool WorldSimApi::finalizeTrajectory()
{""",
            """bool WorldSimApi::setTrajectoryGoal(const Vector3r& goal_ned)
{
    bool result = false;
    UAirBlueprintLib::RunCommandOnGameThread([this, &goal_ned, &result]() {
        if (simmode_ == nullptr || simmode_->GetWorld() == nullptr)
        {
            return;
        }
        const PawnSimApi* VehicleSimApi = simmode_->getVehicleSimApi();
        if (VehicleSimApi == nullptr)
        {
            return;
        }
        const FVector GoalWorldPosition = VehicleSimApi->getNedTransform().fromLocalNed(goal_ned);
        result = setSTTopDownCameraGoal(simmode_->GetWorld(), GoalWorldPosition);
    }, true);
    return result;
}

bool WorldSimApi::finalizeTrajectory()
{""",
            path=path,
        )
        path.write_text(text, encoding="utf-8", newline="")
        changed = True

    return changed


def _upgrade_solid_goal_marker(root: Path) -> bool:
    """Replace the wireframe goal sphere with one solid red mesh marker."""

    path = root / "Source/ST/STTopDownCamera.h"
    header = path.read_text(encoding="utf-8")
    if SOLID_GOAL_PATCH_MARKERS["Source/ST/STTopDownCamera.h"] not in header:
        header = _replace_once(
            header,
            "class APawn;\n",
            """class APawn;
class UMaterialInstanceDynamic;
class UMaterialInterface;
class UStaticMesh;
class UStaticMeshComponent;
""",
            path=path,
        )
        header = _replace_once(
            header,
            "\tvoid DetectTakeoff();\n",
            "\tvoid DetectTakeoff();\n\tvoid UpdateTrajectoryGoalMarker();\n",
            path=path,
        )
        header = _replace_once(
            header,
            "\tint32 TrajectoryPointCount;\n",
            """\tint32 TrajectoryPointCount;

\tUPROPERTY(VisibleAnywhere, Category = "ST|Trajectory")
\tUStaticMeshComponent* TrajectoryGoalMarkerComponent;

\tUPROPERTY()
\tUStaticMesh* TrajectoryGoalMarkerMesh;

\tUPROPERTY()
\tUMaterialInterface* TrajectoryGoalBaseMaterial;

\tUPROPERTY(Transient)
\tUMaterialInstanceDynamic* TrajectoryGoalMaterial;
""",
            path=path,
        )
        path.write_text(header, encoding="utf-8", newline="")

    path = root / "Source/ST/STTopDownCamera.cpp"
    cpp = path.read_text(encoding="utf-8")
    if SOLID_GOAL_PATCH_MARKERS["Source/ST/STTopDownCamera.cpp"] in cpp:
        return False

    cpp = _replace_once(
        cpp,
        '#include "Camera/CameraComponent.h"\n',
        """#include "Camera/CameraComponent.h"
#include "Components/StaticMeshComponent.h"
#include "ConstructorHelpers.h"
""",
        path=path,
    )
    cpp = _replace_once(
        cpp,
        '#include "Kismet/GameplayStatics.h"\n',
        """#include "Kismet/GameplayStatics.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
""",
        path=path,
    )
    cpp = _replace_once(
        cpp,
        """\t, bHasTrajectoryGoal(false)
\t, TrajectoryPointCount(0)
{
\tPrimaryActorTick.bCanEverTick = true;
""",
        """\t, bHasTrajectoryGoal(false)
\t, TrajectoryPointCount(0)
\t, TrajectoryGoalMarkerComponent(nullptr)
\t, TrajectoryGoalMarkerMesh(nullptr)
\t, TrajectoryGoalBaseMaterial(nullptr)
\t, TrajectoryGoalMaterial(nullptr)
{
\tPrimaryActorTick.bCanEverTick = true;

\tstatic ConstructorHelpers::FObjectFinder<UStaticMesh> GoalMarkerMeshFinder(
\t\tTEXT("/Engine/BasicShapes/Sphere.Sphere"));
\tstatic ConstructorHelpers::FObjectFinder<UMaterialInterface> GoalMaterialFinder(
\t\tTEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
\tTrajectoryGoalMarkerMesh = GoalMarkerMeshFinder.Succeeded() ? GoalMarkerMeshFinder.Object : nullptr;
\tTrajectoryGoalBaseMaterial = GoalMaterialFinder.Succeeded() ? GoalMaterialFinder.Object : nullptr;
\tTrajectoryGoalMarkerComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("TrajectoryGoalMarker"));
\tTrajectoryGoalMarkerComponent->SetupAttachment(GetRootComponent());
\tTrajectoryGoalMarkerComponent->SetAbsolute(true, true, true);
\tTrajectoryGoalMarkerComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
\tTrajectoryGoalMarkerComponent->SetCastShadow(false);
\tTrajectoryGoalMarkerComponent->SetStaticMesh(TrajectoryGoalMarkerMesh);
\tTrajectoryGoalMarkerComponent->SetMaterial(0, TrajectoryGoalBaseMaterial);
\tTrajectoryGoalMarkerComponent->SetHiddenInGame(true);
""",
        path=path,
    )
    cpp = _replace_once(
        cpp,
        """void ASTTopDownCamera::BeginPlay()
{
\tSuper::BeginPlay();
\tRecalculateFraming();
}
""",
        """void ASTTopDownCamera::BeginPlay()
{
\tSuper::BeginPlay();
\tif (TrajectoryGoalMarkerComponent != nullptr && TrajectoryGoalBaseMaterial != nullptr)
\t{
\t\tTrajectoryGoalMaterial = TrajectoryGoalMarkerComponent
\t\t\t->CreateAndSetMaterialInstanceDynamicFromMaterial(0, TrajectoryGoalBaseMaterial);
\t\tif (TrajectoryGoalMaterial != nullptr)
\t\t{
\t\t\tTrajectoryGoalMaterial->SetVectorParameterValue(TEXT("Color"), TrajectoryGoalColor);
\t\t}
\t}
\tRecalculateFraming();
}
""",
        path=path,
    )
    cpp = _replace_once(
        cpp,
        """\tbHasTrajectoryGoal = false;
\tTrajectoryPointCount = 0;
}

void ASTTopDownCamera::SetTrajectoryGoalMarker""",
        """\tbHasTrajectoryGoal = false;
\tTrajectoryPointCount = 0;
\tif (TrajectoryGoalMarkerComponent != nullptr)
\t{
\t\tTrajectoryGoalMarkerComponent->SetHiddenInGame(true);
\t\tTrajectoryGoalMarkerComponent->SetVisibility(false, true);
\t}
}

void ASTTopDownCamera::UpdateTrajectoryGoalMarker()
{
\tif (!bHasTrajectoryGoal || TrajectoryGoalMarkerComponent == nullptr)
\t{
\t\treturn;
\t}

\tconst float GoalMarkerRadiusCentimetres = FMath::Max(0.01f, TrajectoryGoalMarkerRadiusMeters)
\t\t* CentimetresPerMetre;
\tconst float GoalMarkerScale = GoalMarkerRadiusCentimetres / 50.0f;
\tTrajectoryGoalMarkerComponent->SetWorldLocation(TrajectoryGoalPoint);
\tTrajectoryGoalMarkerComponent->SetWorldScale3D(FVector(GoalMarkerScale));
\tTrajectoryGoalMarkerComponent->SetHiddenInGame(false);
\tTrajectoryGoalMarkerComponent->SetVisibility(true, true);
}

void ASTTopDownCamera::SetTrajectoryGoalMarker""",
        path=path,
    )
    wire_goal = """\tconst float GoalMarkerRadius = FMath::Max(0.01f, TrajectoryGoalMarkerRadiusMeters)
\t\t* CentimetresPerMetre;
\tDrawDebugSphere(
\t\tGetWorld(),
\t\tTrajectoryGoalPoint,
\t\tGoalMarkerRadius,
\t\t24,
\t\tTrajectoryGoalColor.ToFColor(true),
\t\ttrue,
\t\t-1.0f,
\t\t0,
\t\t5.0f);"""
    cpp = _replace_once(cpp, wire_goal, "\tUpdateTrajectoryGoalMarker();", path=path)
    final_wire_goal = """\t\tconst float GoalMarkerRadius = FMath::Max(0.01f, TrajectoryGoalMarkerRadiusMeters)
\t\t\t* CentimetresPerMetre;
\t\tDrawDebugSphere(
\t\t\tGetWorld(),
\t\t\tTrajectoryGoalPoint,
\t\t\tGoalMarkerRadius,
\t\t\t24,
\t\t\tTrajectoryGoalColor.ToFColor(true),
\t\t\ttrue,
\t\t\t-1.0f,
\t\t\t0,
\t\t\t5.0f);"""
    cpp = _replace_once(
        cpp,
        final_wire_goal,
        "\t\tUpdateTrajectoryGoalMarker();",
        path=path,
    )
    path.write_text(cpp, encoding="utf-8", newline="")
    return True


def _patch_camera_header(path: Path, text: str) -> str:
    text = _replace_once(
        text,
        "\tfloat MinimumHeightMeters;\n\n\t/** Recalculate the camera position for the current window size. */",
        """\tfloat MinimumHeightMeters;

\t/** Persistent trajectory colour used in the UE4 viewport. */
\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ST|Trajectory")
\tFLinearColor TrajectoryColor;

\t/** Persistent trajectory line thickness. */
\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ST|Trajectory", meta = (ClampMin = "0.1"))
\tfloat TrajectoryThickness;

\t/** Small upward offset that prevents z-fighting with scene geometry. */
\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ST|Trajectory", meta = (ClampMin = "0.0"))
\tfloat TrajectoryVerticalOffsetMeters;

\t/** Radius of the start and terminal-position markers. */
\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ST|Trajectory", meta = (ClampMin = "0.01"))
\tfloat TrajectoryMarkerRadiusMeters;

\t/** Ignore vehicle movements smaller than this spacing. */
\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ST|Trajectory", meta = (ClampMin = "0.0"))
\tfloat TrajectoryMinimumPointSpacingMeters;

\t/** Recalculate the camera position for the current window size. */""",
        path=path,
    )
    text = _replace_once(
        text,
        """\tUFUNCTION(BlueprintCallable, Category = "ST|TopDown")
\tvoid SwitchToTopDownView();

private:""",
        """\tUFUNCTION(BlueprintCallable, Category = "ST|TopDown")
\tvoid SwitchToTopDownView();

\t/** Remove the path left by an earlier rollout. */
\tUFUNCTION(BlueprintCallable, Category = "ST|Trajectory")
\tvoid ClearTrajectory();

\t/** Append the current vehicle Actor position to the persistent path. */
\tUFUNCTION(BlueprintCallable, Category = "ST|Trajectory")
\tvoid AppendTrajectoryPoint();

\t/** Keep the successful path and draw start/end markers. */
\tUFUNCTION(BlueprintCallable, Category = "ST|Trajectory")
\tvoid FinalizeTrajectory();

private:""",
        path=path,
    )
    return _replace_once(
        text,
        """\tbool bHasTakeoffReference;
\tbool bViewWasSwitched;
};""",
        """\tbool bHasTakeoffReference;
\tbool bViewWasSwitched;
\tFVector TrajectoryStartPoint;
\tFVector LastTrajectoryPoint;
\tbool bHasTrajectoryPoint;
\tint32 TrajectoryPointCount;
};""",
        path=path,
    )


def _patch_camera_cpp(path: Path, text: str) -> str:
    text = _replace_once(
        text,
        '#include "Camera/CameraComponent.h"\n',
        '#include "Camera/CameraComponent.h"\n#include "DrawDebugHelpers.h"\n',
        path=path,
    )
    text = _replace_once(
        text,
        """\t, CameraFOVDegrees(90.0f)
\t, MinimumHeightMeters(15.0f)
\t, LastCameraLocation(FVector::ZeroVector)""",
        """\t, CameraFOVDegrees(90.0f)
\t, MinimumHeightMeters(15.0f)
\t, TrajectoryColor(FLinearColor(1.0f, 0.15f, 0.02f, 1.0f))
\t, TrajectoryThickness(8.0f)
\t, TrajectoryVerticalOffsetMeters(0.08f)
\t, TrajectoryMarkerRadiusMeters(0.35f)
\t, TrajectoryMinimumPointSpacingMeters(0.03f)
\t, LastCameraLocation(FVector::ZeroVector)""",
        path=path,
    )
    text = _replace_once(
        text,
        """\t, bHasTakeoffReference(false)
\t, bViewWasSwitched(false)
{""",
        """\t, bHasTakeoffReference(false)
\t, bViewWasSwitched(false)
\t, TrajectoryStartPoint(FVector::ZeroVector)
\t, LastTrajectoryPoint(FVector::ZeroVector)
\t, bHasTrajectoryPoint(false)
\t, TrajectoryPointCount(0)
{""",
        path=path,
    )
    return _replace_once(
        text,
        """void ASTTopDownCamera::SwitchToTopDownView()
{
\tAPlayerController* PlayerController = UGameplayStatics::GetPlayerController(this, 0);
\tif (PlayerController == nullptr)
\t{
\t\treturn;
\t}

\tPlayerController->SetViewTarget(this);
\tbViewWasSwitched = true;
\tUE_LOG(LogTemp, Warning, TEXT("ST top-down camera is now the active viewport camera"));
}
""",
        """void ASTTopDownCamera::SwitchToTopDownView()
{
\tAPlayerController* PlayerController = UGameplayStatics::GetPlayerController(this, 0);
\tif (PlayerController == nullptr)
\t{
\t\treturn;
\t}

\tPlayerController->SetViewTarget(this);
\tbViewWasSwitched = true;
\tUE_LOG(LogTemp, Warning, TEXT("ST top-down camera is now the active viewport camera"));
}

void ASTTopDownCamera::ClearTrajectory()
{
\tif (GetWorld() != nullptr)
\t{
\t\tFlushPersistentDebugLines(GetWorld());
\t}
\tTrajectoryStartPoint = FVector::ZeroVector;
\tLastTrajectoryPoint = FVector::ZeroVector;
\tbHasTrajectoryPoint = false;
\tTrajectoryPointCount = 0;
}

void ASTTopDownCamera::AppendTrajectoryPoint()
{
\tAPawn* CurrentVehiclePawn = FindVehiclePawn();
\tif (CurrentVehiclePawn == nullptr || GetWorld() == nullptr)
\t{
\t\treturn;
\t}

\tconst FVector CurrentPoint = CurrentVehiclePawn->GetActorLocation()
\t\t+ FVector(0.0f, 0.0f, FMath::Max(0.0f, TrajectoryVerticalOffsetMeters) * CentimetresPerMetre);
\tif (!bHasTrajectoryPoint)
\t{
\t\tTrajectoryStartPoint = CurrentPoint;
\t\tLastTrajectoryPoint = CurrentPoint;
\t\tbHasTrajectoryPoint = true;
\t\tTrajectoryPointCount = 1;
\t\treturn;
\t}

\tconst float MinimumSpacing = FMath::Max(0.0f, TrajectoryMinimumPointSpacingMeters)
\t\t* CentimetresPerMetre;
\tif ((CurrentPoint - LastTrajectoryPoint).SizeSquared() < FMath::Square(MinimumSpacing))
\t{
\t\treturn;
\t}

\tDrawDebugLine(
\t\tGetWorld(),
\t\tLastTrajectoryPoint,
\t\tCurrentPoint,
\t\tTrajectoryColor.ToFColor(true),
\t\ttrue,
\t\t-1.0f,
\t\t0,
\t\tFMath::Max(0.1f, TrajectoryThickness));
\tLastTrajectoryPoint = CurrentPoint;
\t++TrajectoryPointCount;
}

void ASTTopDownCamera::FinalizeTrajectory()
{
\tAppendTrajectoryPoint();
\tif (!bHasTrajectoryPoint || GetWorld() == nullptr)
\t{
\t\treturn;
\t}

\tconst float MarkerRadius = FMath::Max(0.01f, TrajectoryMarkerRadiusMeters)
\t\t* CentimetresPerMetre;
\tDrawDebugSphere(GetWorld(), TrajectoryStartPoint, MarkerRadius, 24, FColor::Green, true, -1.0f, 0, 4.0f);
\tDrawDebugSphere(GetWorld(), LastTrajectoryPoint, MarkerRadius, 24, FColor::Blue, true, -1.0f, 0, 4.0f);
\tUE_LOG(LogTemp, Warning, TEXT("ST: finalized successful trajectory with %d points"), TrajectoryPointCount);
}
""",
        path=path,
    )


def _patch_world_api_base(path: Path, text: str) -> str:
    return _replace_once(
        text,
        """    virtual bool switchToTopDownCamera() { return false; }

    //----------- APIs to control ACharacter in scene ----------/""",
        """    virtual bool switchToTopDownCamera() { return false; }
    virtual bool clearTrajectory() { return false; }
    virtual bool appendTrajectoryPoint() { return false; }
    virtual bool finalizeTrajectory() { return false; }

    //----------- APIs to control ACharacter in scene ----------/""",
        path=path,
    )


def _patch_world_api_header(path: Path, text: str) -> str:
    return _replace_once(
        text,
        """    virtual bool setObjectPose(const std::string& object_name, const Pose& pose, bool teleport) override;
    virtual bool switchToTopDownCamera() override;

    //----------- APIs to control ACharacter in scene ----------/""",
        """    virtual bool setObjectPose(const std::string& object_name, const Pose& pose, bool teleport) override;
    virtual bool switchToTopDownCamera() override;
    virtual bool clearTrajectory() override;
    virtual bool appendTrajectoryPoint() override;
    virtual bool finalizeTrajectory() override;

    //----------- APIs to control ACharacter in scene ----------/""",
        path=path,
    )


def _patch_world_api_cpp(path: Path, text: str) -> str:
    old = """bool WorldSimApi::switchToTopDownCamera()
{
    bool result = false;
    UAirBlueprintLib::RunCommandOnGameThread([this, &result]() {
        if (simmode_ == nullptr || simmode_->GetWorld() == nullptr)
        {
            return;
        }

        for (TActorIterator<AActor> It(simmode_->GetWorld()); It; ++It)
        {
            AActor* Actor = *It;
            if (Actor == nullptr || !Actor->GetName().StartsWith(TEXT("STTopDownCamera")))
            {
                continue;
            }

            UFunction* SwitchFunction = Actor->FindFunction(FName(TEXT("SwitchToTopDownView")));
            if (SwitchFunction != nullptr)
            {
                Actor->ProcessEvent(SwitchFunction, nullptr);
                result = true;
            }
            return;
        }
    }, true);
    return result;
}
"""
    new = """namespace
{
    bool invokeSTTopDownCamera(UWorld* World, const FName& FunctionName)
    {
        if (World == nullptr)
        {
            return false;
        }

        for (TActorIterator<AActor> It(World); It; ++It)
        {
            AActor* Actor = *It;
            if (Actor == nullptr || !Actor->GetName().StartsWith(TEXT("STTopDownCamera")))
            {
                continue;
            }

            UFunction* Function = Actor->FindFunction(FunctionName);
            if (Function == nullptr)
            {
                return false;
            }
            Actor->ProcessEvent(Function, nullptr);
            return true;
        }
        return false;
    }
}

bool WorldSimApi::switchToTopDownCamera()
{
    bool result = false;
    UAirBlueprintLib::RunCommandOnGameThread([this, &result]() {
        result = simmode_ != nullptr
            && invokeSTTopDownCamera(simmode_->GetWorld(), FName(TEXT("SwitchToTopDownView")));
    }, true);
    return result;
}

bool WorldSimApi::clearTrajectory()
{
    bool result = false;
    UAirBlueprintLib::RunCommandOnGameThread([this, &result]() {
        result = simmode_ != nullptr
            && invokeSTTopDownCamera(simmode_->GetWorld(), FName(TEXT("ClearTrajectory")));
    }, true);
    return result;
}

bool WorldSimApi::appendTrajectoryPoint()
{
    bool result = false;
    UAirBlueprintLib::RunCommandOnGameThread([this, &result]() {
        result = simmode_ != nullptr
            && invokeSTTopDownCamera(simmode_->GetWorld(), FName(TEXT("AppendTrajectoryPoint")));
    }, true);
    return result;
}

bool WorldSimApi::finalizeTrajectory()
{
    bool result = false;
    UAirBlueprintLib::RunCommandOnGameThread([this, &result]() {
        result = simmode_ != nullptr
            && invokeSTTopDownCamera(simmode_->GetWorld(), FName(TEXT("FinalizeTrajectory")));
    }, true);
    return result;
}
"""
    return _replace_once(text, old, new, path=path)


def _patch_rpc_server(path: Path, text: str) -> str:
    return _replace_once(
        text,
        """    pimpl_->server.bind("simSwitchToTopDownCamera", [&]() -> bool {
        return getWorldSimApi()->switchToTopDownCamera();
    });

    pimpl_->server.bind("simGetGroundTruthKinematics""" ,
        """    pimpl_->server.bind("simSwitchToTopDownCamera", [&]() -> bool {
        return getWorldSimApi()->switchToTopDownCamera();
    });
    pimpl_->server.bind("simClearTrajectory", [&]() -> bool {
        return getWorldSimApi()->clearTrajectory();
    });
    pimpl_->server.bind("simAppendTrajectoryPoint", [&]() -> bool {
        return getWorldSimApi()->appendTrajectoryPoint();
    });
    pimpl_->server.bind("simFinalizeTrajectory", [&]() -> bool {
        return getWorldSimApi()->finalizeTrajectory();
    });

    pimpl_->server.bind("simGetGroundTruthKinematics""",
        path=path,
    )


PATCHERS = {
    "Source/ST/STTopDownCamera.h": _patch_camera_header,
    "Source/ST/STTopDownCamera.cpp": _patch_camera_cpp,
    "Plugins/AirSim/Source/AirLib/include/api/WorldSimApiBase.hpp": _patch_world_api_base,
    "Plugins/AirSim/Source/WorldSimApi.h": _patch_world_api_header,
    "Plugins/AirSim/Source/WorldSimApi.cpp": _patch_world_api_cpp,
    "Plugins/AirSim/Source/AirLib/src/api/RpcLibServerBase.cpp": _patch_rpc_server,
}


def apply_patch(root: Path, *, create_backups: bool = True) -> None:
    missing = [relative for relative in PATCHERS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "ST project is missing required source files: " + ", ".join(missing)
        )

    if all(PATCH_MARKERS[relative] in (root / relative).read_text(encoding="utf-8") for relative in PATCHERS):
        if _upgrade_persistent_depth_priority(root):
            print(f"Upgraded ST trajectory lines to persistent world depth: {root}")
        if _upgrade_goal_marker(root):
            print(f"Added configured navigation-goal marker: {root}")
        if _upgrade_solid_goal_marker(root):
            print(f"Upgraded configured goal marker to a solid sphere: {root}")
        print(f"ST trajectory renderer is already patched: {root}")
        return

    for relative, patcher in PATCHERS.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        marker = PATCH_MARKERS[relative]
        if marker in text:
            continue
        updated = patcher(path, text)
        if create_backups:
            _backup(path)
        path.write_text(updated, encoding="utf-8", newline="")
        print(f"Patched {path}")

    _upgrade_persistent_depth_priority(root)
    _upgrade_goal_marker(root)
    _upgrade_solid_goal_marker(root)


def check_patch(root: Path) -> None:
    missing = []
    for relative, marker in PATCH_MARKERS.items():
        path = root / relative
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            missing.append(f"{relative}: {marker}")
    if missing:
        raise RuntimeError("ST trajectory patch is incomplete:\n" + "\n".join(missing))
    missing_goal = []
    for relative, marker in GOAL_PATCH_MARKERS.items():
        path = root / relative
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            missing_goal.append(f"{relative}: {marker}")
    if missing_goal:
        raise RuntimeError(
            "ST configured-goal marker patch is incomplete:\n"
            + "\n".join(missing_goal)
        )
    missing_solid_goal = []
    for relative, marker in SOLID_GOAL_PATCH_MARKERS.items():
        path = root / relative
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            missing_solid_goal.append(f"{relative}: {marker}")
    if missing_solid_goal:
        raise RuntimeError(
            "ST solid goal-marker patch is incomplete:\n"
            + "\n".join(missing_solid_goal)
        )
    camera_cpp = (root / "Source/ST/STTopDownCamera.cpp").read_text(encoding="utf-8")
    foreground_patterns = (
        "\t\t-1.0f,\n\t\t1,\n\t\tFMath::Max(0.1f, TrajectoryThickness));",
        "FColor::Green, true, -1.0f, 1, 4.0f);",
        "FColor::Blue, true, -1.0f, 1, 4.0f);",
    )
    if any(pattern in camera_cpp for pattern in foreground_patterns):
        raise RuntimeError(
            "ST trajectory primitives still use one-frame foreground depth priority"
        )
    print(f"ST trajectory renderer patch verified: {root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/d/Projects/ST"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.check:
        check_patch(root)
        return
    apply_patch(root, create_backups=not args.no_backup)
    check_patch(root)


if __name__ == "__main__":
    main()
