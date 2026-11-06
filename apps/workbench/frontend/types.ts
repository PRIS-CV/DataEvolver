import type { RpcEventListener } from "@earendil-works/pi-coding-agent";

export type Agent = { provider: string; model: string; thinking: string };
export type Provider = {
  id: string;
  api: string;
  baseUrl: string;
  apiKeyEnv: string;
  models: {
    id: string;
    reasoning: boolean;
    input: string[];
    contextWindow: number;
    maxTokens: number;
  }[];
};
export type Route = { baseUrl: string; model: string; apiKeyEnv: string };
export type Config = {
  version: number;
  providers: Provider[];
  agents: { main: Agent; child: Agent };
  orchestration: {
    maxOpenChildren: number;
    maxConcurrentJobs: number;
    maxQueuedJobs: number;
    maxContextBytes: number;
  };
  image: { generation: Route; edit: Route; promptEnhancer: Route };
  paths: Record<string, string>;
  execution: Record<string, string>;
  pi: { cli: string; solExtension: string };
  sol: {
    enabled: boolean;
    actionFusion: boolean;
    observationPack: boolean;
    evidencePreservingReducer: boolean;
    onlineContextCompact: boolean;
    cacheWriteReadRatio: number;
    reducerProvider: string;
    reducerModel: string;
  };
};
export type Session = {
  id: string;
  parentId: string | null;
  name: string;
  createdAt: string;
  profile: Agent;
  state: string;
  mode?: "task" | "proposal" | "review";
  workspace?: string;
  sourceWorkspace?: string;
  contextPath?: string | null;
  contextSha256?: string | null;
};
export type Job = {
  id: string;
  command: string;
  status: string;
  target: string;
  createdAt: string;
  output?: string;
  truncated?: boolean;
  values: Record<string, string | number | boolean>;
};
export type Message = {
  role: string;
  content:
    | string
    | {
        type: string;
        text?: string;
        name?: string;
        data?: string;
        mimeType?: string;
      }[];
  timestamp?: number;
  toolName?: string;
  isError?: boolean;
};
export type Bootstrap = {
  config: Config;
  credentials: string[];
  sessions: Session[];
  jobs: Job[];
  inspection: { paths: Record<string, string>; sshHosts: string[] };
};

export type ImageResult = {
  path?: string;
  text?: string;
  receipt: string;
  sha256?: string;
  host?: string;
};
export type WorkbenchEvent =
  | { type: "connected" }
  | {
      type: "pi_event";
      sessionId: string;
      state: string;
      event: Parameters<RpcEventListener>[0];
    }
  | { type: "session_error"; sessionId: string; error: string }
  | { type: "session_created"; session: Session }
  | { type: "session_opened" | "session_stopped"; sessionId: string }
  | { type: "job_queued" | "job_started" | "job_finished"; job: Job }
  | { type: "job_output"; jobId: string; text: string };
