#!/bin/bash
# run_all.sh — 并行启动 4 个模型的 Blender 渲染任务
# 用法:
#   bash run_all.sh                   # 完整渲染（30帧 × 10相机/圈）
#   bash run_all.sh sanity            # 快速验证（5帧 × 3相机/圈）
#   bash run_all.sh sanity 3          # 只渲模型 3（快速验证）

BASE_DIR="<blender-workspace>"
BLENDER="blender"
SCRIPT="$BASE_DIR/scripts/render_all.py"
SCENE_ID=4

MODE="${1:-full}"
ONLY_MODEL="${2:-}"   # 如果指定，只渲这一个模型

if [ "$MODE" = "sanity" ]; then
    NUM_FRAMES=5
    NUM_CAMERAS=3
    echo "=== 快速验证模式：${NUM_FRAMES}帧 × ${NUM_CAMERAS}相机/圈 ==="
else
    NUM_FRAMES=30
    NUM_CAMERAS=10
    echo "=== 完整渲染模式：${NUM_FRAMES}帧 × ${NUM_CAMERAS}相机/圈 ==="
fi

# GPU 分配：model 1→GPU0, model 2→GPU1, model 3→GPU2, model 4→GPU0（等前3个完成后）
declare -A MODEL_GPU
MODEL_GPU[1]=0
MODEL_GPU[2]=1
MODEL_GPU[3]=2
MODEL_GPU[4]=0   # model 4 复用 GPU0（会排队）

run_model() {
    local model_id=$1
    local gpu_idx=${MODEL_GPU[$model_id]}
    local session="render_m${model_id}"

    # 检查 tmux session 是否已存在
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "⚠️  tmux session '$session' 已存在，跳过（如需重跑请先 tmux kill-session -t $session）"
        return
    fi

    echo "🚀 启动 model=${model_id} scene=${SCENE_ID} gpu=${gpu_idx} tmux=${session}"
    tmux new-session -d -s "$session" \
        "$BLENDER -b -P $SCRIPT -- \
            --model-id $model_id \
            --scene-id $SCENE_ID \
            --gpu-index $gpu_idx \
            --num-frames $NUM_FRAMES \
            --num-cameras-per-ring $NUM_CAMERAS \
        ; echo '=== 任务完成 (model=$model_id) ==='; sleep 60"
    echo "   tmux attach -t $session  （查看进度）"
}

if [ -n "$ONLY_MODEL" ]; then
    run_model "$ONLY_MODEL"
else
    # model 1/2/3 立即并行启动
    for mid in 1 2 3; do
        run_model $mid
    done
    # model 4 稍后启动（等 GPU0 完成 model 1 后手动运行，或直接启动让它等）
    echo ""
    echo "ℹ️  model 4 使用 GPU0，与 model 1 共用。"
    echo "   建议等 model 1 完成后再运行:"
    echo "   bash $BASE_DIR/run_all.sh $MODE 4"
fi

echo ""
echo "=== 查看所有渲染进度 ==="
echo "  tmux ls"
for mid in 1 2 3 4; do
    echo "  tmux attach -t render_m${mid}"
done
echo ""
echo "输出目录: $BASE_DIR/output/${SCENE_ID}_{model_id}/"
