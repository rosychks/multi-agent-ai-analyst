"use client";

import { useState, useRef } from "react";

const AGENT_LABELS: Record<string, string> = {
  recall: "Восстанавливаю контекст диалога",
  supervisor: "Определяю стратегию ответа",
  retriever: "Анализирую базу знаний",
  web: "Собираю данные из сети",
  data_sql: "Выполняю запрос к базе данных",
  code: "Провожу вычисления",
  draft_answer: "Формулирую ответ",
  critic: "Проверяю точность ответа",
  save: "Обновляю память системы",
};

// Signature mark: two nodes joined by a signal line, with a spark
// travelling between them. Doubles as the logo (slow idle pulse)
// and the loading indicator (fast pulse while a request is in flight).
function SynapseMark({ size = 28, pulse = false }: { size?: number; pulse?: boolean }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <line x1="9" y1="30" x2="31" y2="10" stroke="#17A398" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="9" cy="30" r="6" fill="#FF6F59" />
      <circle cx="31" cy="10" r="6" fill="#FFC145" />
      <circle r="3" fill="#16262A">
        <animateMotion path="M9,30 L31,10" dur={pulse ? "0.85s" : "2.6s"} repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [steps, setSteps] = useState<any[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  async function handleAsk() {
    if (!question.trim() || loading) return;

    setSteps([]);
    setFinalAnswer(null);
    setShowEvidence(false);
    setShowHistory(false);
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

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
    } catch (err: any) {
      setFinalAnswer({ answer: `Ошибка соединения: ${err.message}`, critic_ok: false });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="min-h-screen text-[#16262A]"
      style={{
        background: "linear-gradient(160deg, #FDECE8 0%, #F4F7EF 45%, #E7F5F1 100%)",
      }}
    >
      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=JetBrains+Mono:wght@500&display=swap");
      `}</style>

      <div className="mx-auto max-w-2xl px-6 pb-24 pt-16">
        {/* Header / signature */}
        <div className="mb-1 flex items-center gap-3">
          <SynapseMark size={34} pulse={loading} />
          <h1
            className="text-2xl font-bold tracking-tight"
            style={{ fontFamily: "'Sora', sans-serif" }}
          >
            Synapse
          </h1>
        </div>
        <p className="mb-8 text-sm text-[#5B6D6A]">
          Ваш ИИ-аналитик для быстрых и точных ответов
        </p>

        {/* Input row */}
        <div className="mb-10 flex gap-3">
          <div className="relative flex-1">
            <input
              className="w-full rounded-full border-2 border-[#16262A]/10 bg-white px-5 py-3 pr-11 text-[#16262A] outline-none transition placeholder:text-[#9AACA9] focus:border-[#17A398]"
              placeholder="Введите запрос..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            />
            {question.length > 0 && (
              <button
                type="button"
                onClick={() => setQuestion("")}
                aria-label="Очистить запрос"
                className="absolute right-4 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded-full text-[#9AACA9] transition hover:bg-[#EFF3F2] hover:text-[#5B6D6A]"
              >
                ×
              </button>
            )}
          </div>
          <button
            onClick={handleAsk}
            disabled={loading}
            className="flex items-center gap-2 rounded-full bg-[#FF6F59] px-6 py-3 font-medium text-white transition hover:bg-[#FF5A42] disabled:bg-[#EFDAD5] disabled:text-[#BFA9A2]"
          >
            {loading ? <SynapseMark size={20} pulse /> : "Спросить"}
          </button>
        </div>

        {/* Agent steps — a vertical signal chain, in execution order */}
        {steps.length > 0 && (
          <div className="relative mb-10">
            <div className="absolute bottom-3 left-[19px] top-3 w-[2px] bg-[#DCE7E5]" />
            <div>
              {steps.map((node, i) => {
                const isCurrent = loading && i === steps.length - 1;
                const label = AGENT_LABELS[node as keyof typeof AGENT_LABELS] || node;
                return (
                  <div key={i} className="relative flex items-center gap-4 py-2">
                    <div
                      className={`relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-full border-2 bg-white ${
                        isCurrent ? "border-[#FFC145]" : "border-[#17A398]"
                      }`}
                    >
                      <span
                        className={`h-2.5 w-2.5 rounded-full ${
                          isCurrent ? "animate-pulse bg-[#FFC145]" : "bg-[#17A398]"
                        }`}
                      />
                    </div>
                    <span
                      className={`text-sm ${
                        isCurrent ? "font-medium text-[#16262A]" : "text-[#5B6D6A]"
                      }`}
                    >
                      {label}
                      {isCurrent ? "…" : ""}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Final answer */}
        {finalAnswer && (
          <div className="relative rounded-2xl border-2 border-[#16262A]/10 bg-white p-6 shadow-[6px_6px_0_0_#FFC145]">
            <div className="absolute -top-3 right-5 flex gap-2">
              {typeof finalAnswer.confidence === "number" && (
                <span
                  className={`inline-flex items-center rounded-full border-2 bg-white px-3 py-1 text-xs font-medium ${
                    finalAnswer.confidence >= 75
                      ? "border-[#17A398] text-[#17A398]"
                      : finalAnswer.confidence >= 40
                      ? "border-[#FFC145] text-[#8A6A00]"
                      : "border-[#FF6F59] text-[#FF6F59]"
                  }`}
                >
                  Уверенность: {finalAnswer.confidence}%
                </span>
              )}
              <span
                className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium text-white ${
                  finalAnswer.critic_ok ? "bg-[#17A398]" : "bg-[#FF6F59]"
                }`}
              >
                {finalAnswer.critic_ok ? "Одобрено критиком" : "Не одобрено"}
              </span>
            </div>

            <p className="text-lg leading-relaxed text-[#16262A]">{finalAnswer.answer}</p>

            <div className="mt-5 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setShowEvidence((v) => !v)}
                className="rounded-full border-2 border-[#16262A]/10 px-3 py-1.5 text-xs font-medium text-[#5B6D6A] transition hover:border-[#17A398] hover:text-[#17A398]"
              >
                {showEvidence ? "Скрыть источники" : "Как я это проверил"}
              </button>
              {finalAnswer.draft_history?.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowHistory((v) => !v)}
                  className="rounded-full border-2 border-[#16262A]/10 px-3 py-1.5 text-xs font-medium text-[#5B6D6A] transition hover:border-[#FFC145] hover:text-[#8A6A00]"
                >
                  {showHistory
                    ? "Скрыть процесс проверки"
                    : `Процесс проверки (${finalAnswer.draft_history.length})`}
                </button>
              )}
            </div>

            {showEvidence && (
              <div className="mt-4 space-y-3 rounded-xl bg-[#F5F7F6] p-4 text-sm">
                {finalAnswer.evidence?.sql_result && (
                  <div>
                    <p className="mb-1 font-medium text-[#16262A]">База данных</p>
                    <pre className="whitespace-pre-wrap break-words text-xs text-[#5B6D6A]">
                      {finalAnswer.evidence.sql_result}
                    </pre>
                  </div>
                )}
                {finalAnswer.evidence?.web_result && (
                  <div>
                    <p className="mb-1 font-medium text-[#16262A]">Веб-поиск</p>
                    <p className="text-xs text-[#5B6D6A]">{finalAnswer.evidence.web_result}</p>
                  </div>
                )}
                {finalAnswer.evidence?.documents?.length > 0 && (
                  <div>
                    <p className="mb-1 font-medium text-[#16262A]">Документы</p>
                    {finalAnswer.evidence.documents.map((doc: string, i: number) => (
                      <p key={i} className="mb-1 text-xs text-[#5B6D6A]">
                        {doc}
                      </p>
                    ))}
                  </div>
                )}
                {finalAnswer.evidence?.code_result && (
                  <div>
                    <p className="mb-1 font-medium text-[#16262A]">Расчёты</p>
                    <pre className="whitespace-pre-wrap break-words text-xs text-[#5B6D6A]">
                      {finalAnswer.evidence.code_result}
                    </pre>
                  </div>
                )}
                {!finalAnswer.evidence?.sql_result &&
                  !finalAnswer.evidence?.web_result &&
                  !finalAnswer.evidence?.code_result &&
                  !(finalAnswer.evidence?.documents?.length > 0) && (
                    <p className="text-xs text-[#5B6D6A]">
                      Данные из базы/сети не запрашивались — ответ основан на общих знаниях модели.
                    </p>
                  )}
              </div>
            )}

            {showHistory && (
              <div className="mt-4 space-y-2 rounded-xl bg-[#F5F7F6] p-4">
                {finalAnswer.draft_history.map((d: any, i: number) => (
                  <div key={i} className="flex items-start gap-3 text-sm">
                    <span
                      className={`mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white ${
                        d.approved ? "bg-[#17A398]" : d.approved === false ? "bg-[#FF6F59]" : "bg-[#DCE7E5]"
                      }`}
                    >
                      {i + 1}
                    </span>
                    <div>
                      <p className="text-[#16262A]">
                        {d.approved ? "Одобрено" : d.approved === false ? "Отклонено" : "В процессе"}
                        {typeof d.confidence === "number" ? ` · уверенность ${d.confidence}%` : ""}
                      </p>
                      {d.reason && <p className="text-xs text-[#5B6D6A]">{d.reason}</p>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {finalAnswer.revisions !== undefined && (
              <p
                className="mt-4 text-xs text-[#5B6D6A]"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                ревизий: {finalAnswer.revisions}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}