param(
  [Parameter(Mandatory=$true)][string]$MeshA,
  [Parameter(Mandatory=$true)][string]$MeshB,
  [Parameter(Mandatory=$true)][string]$Scene,
  [string]$RefA = "",
  [string]$RefB = "",
  [string]$RootDir = "pipeline/data/dual_vlm_gate_runs/example",
  [string]$SampleId = "dual_example",
  [string]$Device = "cuda:0",
  [double]$Threshold = 0.72,
  [int]$MaxRounds = 5
)

$ErrorActionPreference = "Stop"
$cmd = @(
  "python", "pipeline/dual_vlm_gate/dual_gate_loop.py",
  "--root-dir", $RootDir,
  "--sample-id", $SampleId,
  "--mesh-a", $MeshA,
  "--mesh-b", $MeshB,
  "--scene", $Scene,
  "--device", $Device,
  "--threshold", "$Threshold",
  "--max-rounds", "$MaxRounds"
)
if ($RefA) { $cmd += @("--ref-a", $RefA) }
if ($RefB) { $cmd += @("--ref-b", $RefB) }

& $cmd[0] $cmd[1..($cmd.Count - 1)]
