"use client";

import { useState, useRef } from "react";

const AGENT_LABELS = {
  recall: "🧠 Вспоминаю прошлые диалоги...",
  supervisor: "🧭 Супервизор решает, что делать дальше...",
  retriever: "📄 Ищу в документах...",
  web: "🌐 Ищу в интернете...",
  data_sql: "🗄️ Запрашиваю базу данных...",
  code: "🧮 Считаю...",
  draft_answer: "✍️ Формирую черновик ответа...",
  critic: "🔍 Критик проверяет ответ...",
  save: "💾 Сохраняю в память...",
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [steps, setSteps] = useState([]);
  const [finalAnswer, setFinalAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef(null);

  async function handleAsk() {
    if (!question.trim() || loading) return;

    setSteps([]);
    setFinalAnswer(null);
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const rawEvent of events) {
          if (!rawEvent.trim()) continue;

          const lines = rawEvent.split("\n");
          let eventType = "message";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) eventType = line.slice(7);
            if (line.startsWith("data: ")) dataStr = line.slice(6);
          }

          if (!dataStr) continue;
          const data = JSON.parse(dataStr);

          if (eventType === "step") {
            setSteps((prev) => [...prev, data.node]);
          } else if (eventType === "final") {
            setFinalAnswer(data);
          } else if (eventType === "error") {
            setFinalAnswer({ answer: `Ошибка: ${data.message}`, critic_ok: false });
          }
        }
      }
    } catch (err) {
      setFinalAnswer({ answer: `Ошибка соединения: ${err.message}`, critic_ok: false });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold mb-1">🌌 Multi-Agent AI Analyst</h1>
        <p className="text-slate-400 mb-6 text-sm">
          Спросите про звёзды, чёрные дыры или созвездия
        </p>

        <div className="flex gap-2 mb-6">
          <input
            className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 outline-none focus:border-blue-500"
            placeholder="Какая самая массивная чёрная дыра в базе?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          />
          <button
            onClick={handleAsk}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 rounded-lg px-5 py-2 font-medium transition"
          >
            {loading ? "..." : "Спросить"}
          </button>
        </div>

        {steps.length > 0 && (
          <div className="mb-6 space-y-2">
            {steps.map((node, i) => (
              <div
                key={i}
                className="flex items-center gap-2 text-sm text-slate-300 bg-slate-900/50 rounded-lg px-4 py-2 border border-slate-800 animate-pulse"
                style={{ animationIterationCount: i === steps.length - 1 && loading ? "infinite" : 1 }}
              >
                <span>{AGENT_LABELS[node] || `⚙️ ${node}`}</span>
              </div>
            ))}
          </div>
        )}

        {finalAnswer && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <p className="text-lg mb-3">{finalAnswer.answer}</p>
            <div className="flex gap-3 text-xs text-slate-500">
              <span>
                {finalAnswer.critic_ok ? "✅ Одобрено критиком" : "⚠️ Не одобрено"}
              </span>
              {finalAnswer.revisions !== undefined && (
                <span>🔄 Ревизий: {finalAnswer.revisions}</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}