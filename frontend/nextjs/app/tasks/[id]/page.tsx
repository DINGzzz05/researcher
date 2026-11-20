"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import Report from "@/components/Task/Report";
import { getHost } from "@/helpers/getHost";

interface ResearchTask {
  query: string;
  status: string;
  error_code: string | null;
  can_resume: boolean;
  cancel_requested: boolean;
  progress?: { stage?: string; completed?: number; total?: number };
}

interface ResearchResult {
  report: string;
  pdf_path?: string;
  docx_path?: string;
}

interface Activity {
  id: string;
  message: string;
}

function readActivity(key: string): { cursor: string; activity: Activity[] } {
  try {
    return JSON.parse(sessionStorage.getItem(key) || "null") || { cursor: "", activity: [] };
  } catch {
    return { cursor: "", activity: [] };
  }
}

export default function ResearchTaskPage({ params }: { params: { id: string } }) {
  const [task, setTask] = useState<ResearchTask>();
  const [result, setResult] = useState<ResearchResult>();
  const [activity, setActivity] = useState<Activity[]>([]);
  const [backend, setBackend] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [actionError, setActionError] = useState("");
  const [pending, setPending] = useState(false);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    const host = getHost().replace(/\/$/, "");
    setBackend(host);
    const url = `${host}/api/research/${encodeURIComponent(params.id)}`;
    const key = `research-events:${url}`;
    let { cursor, activity: history } = readActivity(key);
    setActivity(history);
    setResult(undefined);
    const controller = new AbortController();
    let stream: EventSource | undefined;
    let finished = false;
    let syncing = false;

    const saveActivity = () => {
      try {
        sessionStorage.setItem(key, JSON.stringify({ cursor, activity: history }));
      } catch {
        // The active stream can continue if browser storage is unavailable.
      }
    };

    const syncStatus = async () => {
      if (syncing || finished) return;
      syncing = true;
      const options = { signal: controller.signal, cache: "no-store" as const };
      try {
        const response = await fetch(url, options);
        if (response.status === 404) {
          finished = true;
          stream?.close();
          setConnectionError("Research task not found.");
          return;
        }
        if (!response.ok) throw new Error("Could not query the research task.");
        const status: ResearchTask = await response.json();
        if (controller.signal.aborted) return;
        setTask(status);
        if (status.status === "succeeded") {
          const reportResponse = await fetch(`${url}/result`, options);
          if (!reportResponse.ok) throw new Error("Could not retrieve the report.");
          const report: ResearchResult = await reportResponse.json();
          if (controller.signal.aborted) return;
          setResult(report);
        }
        finished = status.status !== "queued" && status.status !== "running";
        setConnectionError("");
        if (!finished && stream?.readyState === EventSource.CLOSED) openStream();
      } catch {
        if (!controller.signal.aborted) setConnectionError("Connection interrupted. Retrying…");
      } finally {
        syncing = false;
      }
    };

    function openStream() {
      const after = cursor ? `?after=${encodeURIComponent(cursor)}` : "";
      const source = new EventSource(`${url}/events${after}`);
      stream = source;
      const receiveEvent = (event: MessageEvent) => {
        if (event.lastEventId === cursor) return;
        cursor = event.lastEventId;
        if (event.type !== "report_delta") {
          const data = JSON.parse(event.data);
          const message = data.output ?? data.stage ?? data.content;
          if (typeof message === "string") {
            history = [...history.slice(-79), { id: cursor, message: message.slice(0, 1000) }];
            setActivity(history);
          }
        }
        saveActivity();
      };
      for (const kind of ["log", "stage_changed", "progress", "report_delta"]) {
        source.addEventListener(kind, receiveEvent);
      }
      source.addEventListener("reset", (event: MessageEvent) => {
        cursor = event.lastEventId;
        history = [];
        setActivity(history);
        saveActivity();
        void syncStatus();
      });
      source.addEventListener("task_status", (event: MessageEvent) => {
        const { status } = JSON.parse(event.data);
        if (status !== "queued" && status !== "running") source.close();
        void syncStatus();
      });
      source.onerror = () => { if (finished) source.close(); };
    }

    openStream();
    void syncStatus();
    const poll = window.setInterval(syncStatus, 4000);
    return () => {
      controller.abort();
      stream?.close();
      window.clearInterval(poll);
    };
  }, [params.id, revision]);

  const updateTask = async (action: "resume" | "cancel") => {
    setPending(true);
    setActionError("");
    try {
      const response = await fetch(
        `${backend}/api/research/${encodeURIComponent(params.id)}/${action}`,
        { method: "POST" },
      );
      if (!response.ok) throw new Error();
      setRevision((previous) => previous + 1);
    } catch {
      setActionError(`Could not ${action} this task. Refresh to check its current status.`);
    } finally {
      setPending(false);
    }
  };

  const progress = task?.progress;
  const active = task?.status === "queued" || task?.status === "running";
  return (
    <main className="mx-auto max-w-4xl space-y-6 px-5 py-10 text-slate-100">
      <Link href="/" className="text-sky-300 hover:underline">New research</Link>
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold">{task?.query || "Research task"}</h1>
        <p role="status">
          {task?.cancel_requested && active ? "Cancelling…" : task?.status || "Loading…"}
          {progress?.stage && ` · ${progress.stage.replace(/_/g, " ")}`}
          {task?.error_code && ` · ${task.error_code.replace(/_/g, " ")}`}
        </p>
        {Boolean(progress?.total) && (
          <p>{progress?.completed ?? 0} / {progress?.total} research sections completed</p>
        )}
        {connectionError && <p role="alert">{connectionError}</p>}
      </header>

      {active && !task.cancel_requested && (
        <button className="rounded border px-3 py-1" disabled={pending} onClick={() => updateTask("cancel")}>
          Cancel task
        </button>
      )}
      {task?.can_resume && (
        <button className="rounded border px-3 py-1" disabled={pending} onClick={() => updateTask("resume")}>
          Resume from checkpoint
        </button>
      )}
      {actionError && <p role="alert">{actionError}</p>}

      {result && (
        <section className="space-y-4">
          <div className="flex gap-4 text-sky-300">
            {result.pdf_path && <a href={`${backend}/${result.pdf_path}`}>Download PDF</a>}
            {result.docx_path && <a href={`${backend}/${result.docx_path}`}>Download DOCX</a>}
          </div>
          <Report report={result.report} />
        </section>
      )}
      {activity.length > 0 && (
        <details open={active} className="rounded border border-slate-700 p-4">
          <summary className="cursor-pointer font-medium">Research activity</summary>
          <ol className="mt-3 max-h-96 space-y-2 overflow-auto break-words text-sm text-slate-300">
            {activity.map(({ id, message }) => <li key={id}>{message}</li>)}
          </ol>
        </details>
      )}
    </main>
  );
}
