import React, { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Bootstrap, Job } from "./types";
import {
  ActionButton,
  SelectField,
  SettingsTabs,
  SettingsTabPanel,
  SwitchField,
  TextField,
} from "./components/ui";
import { formatState, type SettingsCategory, type Translate } from "./ui";
import "./data-workspace.css";

type Stage = "collect" | "curate" | "review";
type RecordRow = Record<string, unknown>;
type PackagePage = {
  package: string;
  metadata: RecordRow;
  metadataSha256: string;
  validation: string;
  rows?: RecordRow[];
  table?: string;
  cursor?: number;
  nextCursor?: number | null;
};
const json = (value: unknown) => JSON.stringify(value, null, 2);
const text = (value: unknown) =>
  typeof value === "string" || typeof value === "number" ? String(value) : "—";
function Detail({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="data-detail">
      <summary>{title}</summary>
      <pre>{json(value)}</pre>
    </details>
  );
}

export function DataWorkspace({
  data,
  t,
  onSettings,
}: {
  data: Bootstrap;
  t: Translate;
  onSettings: (category?: SettingsCategory) => void;
}) {
  const [stage, setStage] = useState<Stage>("collect");
  const [goals, setGoals] = useState({ collect: "", curate: "" });
  const [campaign, setCampaign] = useState("");
  const [campaignRoot, setCampaignRoot] = useState(".dataevolver/campaigns");
  const [idempotencyKey, setIdempotencyKey] = useState("");
  const [maxActions, setMaxActions] = useState("1");
  const [dryRun, setDryRun] = useState(false);
  const [receipt, setReceipt] = useState<Job>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [packagePath, setPackagePath] = useState("");
  const [table, setTable] = useState("manifest.jsonl");
  const [page, setPage] = useState<PackagePage>();
  const [previous, setPrevious] = useState<number[]>([]);
  const [selected, setSelected] = useState(0);
  const [image, setImage] = useState("");
  const [imageError, setImageError] = useState("");
  const receiptRequest = useRef(0);
  const packageRequest = useRef(0);
  const imageRequest = useRef(0);
  const hostScope = JSON.stringify([data.config.execution, data.config.paths]);
  const root =
    data.config.execution.target === "ssh"
      ? data.config.execution.remoteDatasets
      : data.config.paths.datasets;
  const goal = stage === "curate" ? goals.curate : goals.collect;
  const row = page?.rows?.[selected];

  const showReceipt = async (id: string) => {
    const request = ++receiptRequest.current;
    const result = await api<Job>(`jobs/${id}`);
    if (request === receiptRequest.current) setReceipt(result);
  };
  useEffect(() => {
    const id = receipt?.id;
    if (!id || !["queued", "running"].includes(receipt.status)) return;
    const timer = setInterval(() => {
      void showReceipt(id).catch((e) => setError(String(e.message)));
    }, 2000);
    return () => clearInterval(timer);
  }, [receipt?.id, receipt?.status]);
  useEffect(() => {
    packageRequest.current++;
    imageRequest.current++;
    setPage(undefined);
    setPrevious([]);
    setImage("");
    setImageError("");
  }, [hostScope]);
  useEffect(
    () => () => {
      receiptRequest.current++;
      packageRequest.current++;
      imageRequest.current++;
    },
    [],
  );
  useEffect(() => {
    const request = ++imageRequest.current;
    setImage("");
    setImageError("");
    if (!page || typeof row?.image !== "string") return;
    void api<{ dataUrl: string }>("data/read", {
      package: page.package,
      image: row.image,
      scope: hostScope,
    })
      .then((result) => {
        if (request === imageRequest.current) setImage(result.dataUrl);
      })
      .catch((e) => {
        if (request === imageRequest.current) setImageError(String(e.message));
      });
    return () => {
      imageRequest.current++;
    };
  }, [page, selected]);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function dispatch(command: string) {
    const values =
      command === "plan"
        ? { goal }
        : command === "create"
          ? { goal, campaignRoot, idempotencyKey, dryRun }
          : {
              campaign,
              campaignRoot,
              ...(command === "step" ? { maxActions: Number(maxActions) } : {}),
            };
    const result = await api<Job>("harness", { command, values });
    receiptRequest.current++;
    setReceipt({ ...result, output: "" });
  }
  async function readPage(
    cursor = 0,
    history: number[] = [],
    summary = false,
    packageName = packagePath,
  ) {
    const request = ++packageRequest.current;
    const result = await api<PackagePage>("data/read", {
      package: packageName,
      scope: hostScope,
      ...(summary ? {} : { table, cursor }),
    });
    if (request !== packageRequest.current) return;
    setPage(result);
    setSelected(0);
    setPrevious(history);
  }
  const related = data.jobs.filter(
    (job) =>
      (job.values?.goal &&
        Object.values(goals).includes(String(job.values.goal))) ||
      (campaign &&
        job.values?.campaign === campaign &&
        job.values?.campaignRoot === campaignRoot),
  );
  let plan: RecordRow | undefined;
  if (receipt?.command === "plan" && receipt.status === "succeeded") {
    try {
      plan = JSON.parse(receipt.output || "");
    } catch {
      /* The retained log may be truncated; raw receipt remains available. */
    }
  }
  const tabs = [
    {
      id: "collect",
      label: t("Collect", "收集"),
      summary: t("Sources, quotas and acquisition", "来源、配额与采集"),
    },
    {
      id: "curate",
      label: t("Curate", "清洗"),
      summary: t("Quality gates and independent review", "质量门槛与独立审核"),
    },
    {
      id: "review",
      label: t("Review & export", "审核与导出"),
      summary: t("Samples, quarantine and packages", "样本、隔离与交付包"),
    },
  ];
  const objects = Array.isArray(row?.objects)
    ? (row.objects as RecordRow[])
    : [];

  return (
    <section className="data-workspace">
      <header className="data-heading">
        <div>
          <h1>{t("Data workspace", "数据工作区")}</h1>
          <p>
            {t(
              "From explicit sources to traceable image data for your training goal.",
              "围绕训练目标，将明确的数据来源整理为可追溯的图像数据。",
            )}
          </p>
        </div>
        <ActionButton onClick={() => onSettings("execution")}>
          {t("Execution settings", "执行设置")}
        </ActionButton>
      </header>
      <p className="data-host">
        <strong>{t("Selected host", "当前主机")}</strong>{" "}
        {data.config.execution.target === "ssh"
          ? data.config.execution.sshHost
          : t("Local", "本机")}
        <br />
        {t("Dataset directory", "数据集目录")}:{" "}
        <code>{root || t("Configure in Settings", "请在设置中配置")}</code>
      </p>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <SettingsTabs
        value={stage}
        onValueChange={(value) => setStage(value as Stage)}
        ariaLabel={t("Data workflow", "数据工作流")}
        tabs={tabs}
      >
        {(["collect", "curate"] as const).map((part) => (
          <SettingsTabPanel value={part} key={part}>
            <div className="data-stage">
              <h2>
                {part === "collect"
                  ? t("Collect a source bundle", "收集源数据包")
                  : t("Curate layout data", "清洗布局数据")}
              </h2>
              <p>
                {part === "collect"
                  ? t(
                      "Load an existing collection Goal. The preflight receipt shows pinned revisions, field mappings, quotas, mirror and download budgets.",
                      "加载已有的收集 Goal。预检回执展示固定 revision、字段映射、配额、镜像和下载预算。",
                    )
                  : t(
                      "Load an existing curation Goal with its SourceBundle, quality profile and ModelProfile role bindings. These models are independent of Image studio routes.",
                      "加载已有的清洗 Goal，其中包含 SourceBundle、质量策略和 ModelProfile 角色绑定。这些模型独立于图像工坊的接口配置。",
                    )}
              </p>
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  void run(() => dispatch("plan"));
                }}
              >
                <TextField
                  label={t(
                    "Goal file on selected host",
                    "当前主机上的 Goal 文件",
                  )}
                  value={goals[part]}
                  set={(value) => setGoals({ ...goals, [part]: value })}
                  mono
                  placeholder="path/to/goal.json"
                />
                <div className="data-actions">
                  <ActionButton
                    type="submit"
                    variant="primary"
                    disabled={busy || !goals[part]}
                  >
                    {t("Inspect plan", "预检计划")}
                  </ActionButton>
                  <a
                    href="https://github.com/PRIS-CV/DataEvolver/blob/main/docs/IMAGE_DATA_PIPELINES.md"
                    target="_blank"
                    rel="noreferrer"
                  >
                    {t("Goal and configuration guide", "Goal 与配置指南")}
                  </a>
                </div>
              </form>
              <p className="data-note">
                {t(
                  "Source, model and policy configuration stays in the hashed Goal and its referenced inputs. This page does not rewrite them. Set store and model environment variables on the execution host before starting Workbench; inspect missing variables in plan receipts.",
                  "来源、模型和策略保存在带哈希的 Goal 及其引用输入中，页面不改写这些文件。启动 Workbench 前，请在执行主机设置 store 和模型环境变量；缺失变量会显示在预检回执中。",
                )}
              </p>
            </div>
          </SettingsTabPanel>
        ))}
        <SettingsTabPanel value="review">
          <div className="data-stage">
            <h2>{t("Inspect an existing package", "查看已有交付包")}</h2>
            <p>
              {t(
                "Use the relative directory containing bundle.json or dataset.json under the configured dataset root. Packages stay on the selected host.",
                "填写数据集目录下含 bundle.json 或 dataset.json 的相对目录。交付包保留在当前主机上。",
              )}
            </p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void run(() => readPage());
              }}
            >
              <TextField
                label={t("Package directory (relative)", "交付包相对目录")}
                value={packagePath}
                set={setPackagePath}
                mono
                placeholder="runs/example/export"
              />
              <SelectField
                label={t("Record table", "记录表")}
                value={table}
                onChange={(value) => {
                  packageRequest.current++;
                  setTable(value);
                  setPage(undefined);
                  setPrevious([]);
                }}
                options={[
                  {
                    value: "manifest.jsonl",
                    label: t("Selected records", "入选记录"),
                  },
                  {
                    value: "quarantine.jsonl",
                    label: t("Quarantine reasons", "隔离原因"),
                  },
                  {
                    value: "reviews.jsonl",
                    label: t("Review evidence", "审核证据"),
                  },
                ]}
              />
              <div className="data-actions">
                <ActionButton
                  type="submit"
                  variant="primary"
                  disabled={busy || !root || !packagePath}
                >
                  {t("Read samples", "读取样本")}
                </ActionButton>
                <ActionButton
                  disabled={busy || !root || !packagePath}
                  onClick={() => void run(() => readPage(0, [], true))}
                >
                  {t("Package summary", "交付包摘要")}
                </ActionButton>
                <ActionButton
                  onClick={() =>
                    onSettings(
                      data.config.execution.target === "ssh"
                        ? "execution"
                        : "paths",
                    )
                  }
                >
                  {t("Dataset paths", "数据集路径")}
                </ActionButton>
              </div>
            </form>
            {page && (
              <div className="data-package">
                <div className="data-package-heading">
                  <h3>{page.package}</h3>
                  <span className="pill">
                    {text(page.metadata.governance || page.metadata.mode)}
                  </span>
                </div>
                <p className="data-note">
                  {t(
                    "Read-only viewer. Image bytes are checked against the package member hash; full package validity and training suitability are not verified here.",
                    "只读查看器。图像字节与交付包成员哈希核对；此处不验证完整交付包，也不判定训练适用性。",
                  )}
                </p>
                <Detail
                  title={t(
                    "Package metadata and export members",
                    "包元数据与导出成员",
                  )}
                  value={{
                    ...page.metadata,
                    metadata_sha256: page.metadataSha256,
                  }}
                />
                {page.rows && (
                  <>
                    <div className="data-pagination">
                      <ActionButton
                        size="sm"
                        disabled={busy || !previous.length}
                        onClick={() =>
                          void run(() =>
                            readPage(
                              previous.at(-1)!,
                              previous.slice(0, -1),
                              false,
                              page.package,
                            ),
                          )
                        }
                      >
                        {t("Previous", "上一页")}
                      </ActionButton>
                      <span>
                        {t("Up to 20 records per page", "每页最多 20 条记录")} /{" "}
                        {page.rows.length}
                      </span>
                      <ActionButton
                        size="sm"
                        disabled={busy || page.nextCursor == null}
                        onClick={() =>
                          void run(() =>
                            readPage(
                              page.nextCursor!,
                              [...previous, page.cursor || 0],
                              false,
                              page.package,
                            ),
                          )
                        }
                      >
                        {t("Next", "下一页")}
                      </ActionButton>
                    </div>
                    {!page.rows.length && (
                      <p>{t("No records in this table.", "此表暂无记录。")}</p>
                    )}
                    {page.rows.length > 0 && (
                      <div className="data-samples">
                        <div
                          className="data-record-list"
                          aria-label={t("Samples", "样本")}
                        >
                          {page.rows.map((sample, index) => (
                            <button
                              type="button"
                              key={index}
                              className={
                                index === selected
                                  ? "data-record selected"
                                  : "data-record"
                              }
                              aria-pressed={index === selected}
                              onClick={() => setSelected(index)}
                            >
                              <strong>{text(sample.id)}</strong>
                              <small>
                                {text(sample.governance || sample.source_id)}
                              </small>
                            </button>
                          ))}
                        </div>
                        <div className="data-record-detail">
                          {imageError && <p role="alert">{imageError}</p>}
                          {image && (
                            <figure className="data-image">
                              <img
                                src={image}
                                alt={
                                  typeof row?.caption === "string"
                                    ? row.caption
                                    : t(
                                        "Selected dataset sample",
                                        "所选数据样本",
                                      )
                                }
                              />
                              <svg
                                viewBox="0 0 1 1"
                                preserveAspectRatio="none"
                                aria-hidden="true"
                              >
                                {objects.map((object, index) => {
                                  const box = object?.box_xyxy;
                                  if (
                                    !Array.isArray(box) ||
                                    box.length !== 4 ||
                                    !box.every(
                                      (n) =>
                                        typeof n === "number" &&
                                        Number.isFinite(n) &&
                                        n >= 0 &&
                                        n <= 1,
                                    ) ||
                                    box[2] <= box[0] ||
                                    box[3] <= box[1]
                                  )
                                    return null;
                                  return (
                                    <rect
                                      key={index}
                                      x={box[0]}
                                      y={box[1]}
                                      width={box[2] - box[0]}
                                      height={box[3] - box[1]}
                                    />
                                  );
                                })}
                              </svg>
                            </figure>
                          )}
                          <p>{text(row?.caption)}</p>
                          <p className="data-note">
                            {t("Governance", "证据状态")}:{" "}
                            {text(row?.governance)} / layout_ready:{" "}
                            {text(
                              row?.layout_ready === undefined
                                ? undefined
                                : String(row.layout_ready),
                            )}
                          </p>
                          <Detail
                            title={t(
                              "Annotations, gates and evidence",
                              "标注、门槛与证据",
                            )}
                            value={row}
                          />
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
            <p className="data-note">
              {t(
                "This page displays the pipeline's existing export; it does not create archives or promote records. mock_verified and training_candidate do not establish training benefit.",
                "页面展示流水线已有的导出包，不创建压缩包，也不提升记录等级。mock_verified 和 training_candidate 均不能证明训练收益。",
              )}
            </p>
          </div>
        </SettingsTabPanel>
      </SettingsTabs>

      <section
        className="data-campaign"
        aria-label={t("Campaign control", "Campaign 控制")}
      >
        <h2>{t("Campaign control", "Campaign 控制")}</h2>
        <p>
          {t(
            "Create only initializes state. Bounded step and resume execute work on the selected host; pause requests a checkpoint. Preflight success does not authorize or prove data quality.",
            "创建只初始化状态。有界推进和恢复会在当前主机执行任务；暂停请求写入检查点。预检成功不代表数据质量已通过。",
          )}
        </p>
        <div className="data-form-grid">
          <TextField
            label={t("Campaign ID (from Goal)", "Campaign 标识（与 Goal 一致）")}
            value={campaign}
            set={setCampaign}
            mono
          />
          <TextField
            label={t("Campaign root", "Campaign 根目录")}
            value={campaignRoot}
            set={setCampaignRoot}
            mono
          />
          <TextField
            label={t("Create idempotency key", "创建幂等键")}
            value={idempotencyKey}
            set={setIdempotencyKey}
            mono
          />
          <TextField
            label={t("Maximum actions (1–100)", "最大动作数（1–100）")}
            type="number"
            value={maxActions}
            set={setMaxActions}
          />
        </div>
        <SwitchField
          title={t("Fixture Goal (--dry-run)", "Fixture Goal（--dry-run）")}
          description={t(
            "Enable only to create a fixture image Campaign. Server Goals must keep this off. This is the native mode flag, not a preview of server execution.",
            "仅在创建 fixture 图像 Campaign 时启用。server Goal 应关闭。此标志指定原生模式，不用于预演服务器任务。",
          )}
          checked={dryRun}
          onChange={setDryRun}
        />
        <div className="data-actions">
          <ActionButton
            disabled={
              busy ||
              stage === "review" ||
              !goal ||
              !campaignRoot ||
              !idempotencyKey
            }
            onClick={() => void run(() => dispatch("create"))}
          >
            {t("Create from Goal", "从 Goal 创建")}
          </ActionButton>
          {["step", "status", "explain", "replay", "pause", "resume"].map(
            (command) => (
              <ActionButton
                key={command}
                disabled={busy || !campaign || !campaignRoot}
                variant={command === "step" ? "primary" : "secondary"}
                onClick={() => void run(() => dispatch(command))}
              >
                {
                  {
                    step: t("Bounded step", "有界推进"),
                    status: t("Status", "状态"),
                    explain: t("Explain", "解释"),
                    replay: t("Replay", "回放"),
                    pause: t("Pause", "暂停"),
                    resume: t("Resume execution", "恢复执行"),
                  }[command]
                }
              </ActionButton>
            ),
          )}
        </div>
        <div className="data-jobs">
          {related.slice(0, 12).map((job) => (
            <ActionButton
              key={job.id}
              size="sm"
              onClick={() => void run(() => showReceipt(job.id))}
            >
              {job.command} / {formatState(job.status, t)} /{" "}
              {job.id.slice(0, 8)}
            </ActionButton>
          ))}
        </div>
        {receipt && (
          <div className="data-receipt" aria-live="polite">
            <div className="data-package-heading">
              <h3>{receipt.command}</h3>
              <span className="pill">{formatState(receipt.status, t)}</span>
            </div>
            <code>{receipt.id}</code>
            <Detail
              title={t("Job inputs", "任务输入")}
              value={receipt.values}
            />
            {receipt.status === "unknown" && (
              <p role="alert">
                {t(
                  "Effects are uncertain. Inspect Campaign receipts before retrying.",
                  "执行效果未知。请先核对 Campaign 回执，再决定是否重试。",
                )}
              </p>
            )}
            <div className="data-actions">
              <ActionButton
                size="sm"
                disabled={busy}
                onClick={() => void run(() => showReceipt(receipt.id))}
              >
                {t("Refresh receipt", "刷新回执")}
              </ActionButton>
              {receipt.status === "queued" && (
                <ActionButton
                  size="sm"
                  onClick={() =>
                    void run(async () => {
                      await api(`jobs/${receipt.id}/cancel`, {});
                      await showReceipt(receipt.id);
                    })
                  }
                >
                  {t("Cancel queued job", "取消排队任务")}
                </ActionButton>
              )}
            </div>
            {plan && (
              <>
                <Detail
                  title={t(
                    "Plan, sources, profile and budget",
                    "计划、来源、策略与预算",
                  )}
                  value={plan}
                />
                {Array.isArray(plan.blockers) && plan.blockers.length > 0 && (
                  <p role="alert">
                    {t("Preflight blockers", "预检阻塞项")}:{" "}
                    {plan.blockers.map(text).join(", ")}
                  </p>
                )}
              </>
            )}
            {receipt.truncated && (
              <p>
                {t(
                  "Showing the retained log tail only.",
                  "仅显示已保留的日志尾部。",
                )}
              </p>
            )}
            <pre>
              {receipt.output ||
                t("Waiting for receipt output…", "等待回执输出…")}
            </pre>
          </div>
        )}
      </section>
    </section>
  );
}
