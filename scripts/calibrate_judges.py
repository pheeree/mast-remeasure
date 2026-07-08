#!/usr/bin/env python3
"""MAST judge 캘리브레이션 — human 19편에 최신 judge를 얹어 원 o1(κ=0.77)과 나란히 잰다.

파이프라인 충실도가 원칙: 프롬프트는 agentdash의 빌더를 그대로 쓰고(shim 주입),
정의·few-shot은 원 repo의 definitions.txt/examples.txt(런타임 취득), 응답은 자유 텍스트,
파싱은 agentdash의 정규식 파서를 그대로 임포트한다. 구조화 출력 등 우리 편의는 넣지 않는다
— 재는 것은 "같은 파이프라인에서 judge만 바꿨을 때"이므로.

사용:
  python scripts/calibrate_judges.py --smoke            # 트레이스 1편 × gemini만
  python scripts/calibrate_judges.py                    # 전체 19편 × 3 judge (raw 있으면 재사용)
  python scripts/calibrate_judges.py --judges opus      # 특정 judge만
  python scripts/calibrate_judges.py --report-only      # 저장된 raw로 지표만 재계산

키: ~/.config/knowledge-mind/.env 의 ANTHROPIC_API_KEY / GEMINI_API_KEY (또는 환경변수).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import os

from agentdash.annotator import annotator as MastAnnotator

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = ROOT / "results" / "calibration" / "raw"
REPORT_DIR = ROOT / "results" / "calibration"

MODES = ["1.1", "1.2", "1.3", "1.4", "1.5",
         "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
         "3.1", "3.2", "3.3"]

JUDGES = {
    # Claude 쌍은 정의만 남김 — 사용자 결정(2026-07-08)으로 파일럿은 Gemini 계열만 사용
    "opus": ("anthropic", "claude-opus-4-8"),
    "sonnet": ("anthropic", "claude-sonnet-5"),
    "gemini": ("gemini", "gemini-2.5-flash"),
    "gemini-pro": ("gemini", "gemini-2.5-pro"),
}

# 3단계 진행 임계 (사전 등록, 2026-07-08 — Pro 결과를 보기 전에 확정):
# 어떤 Gemini judge든 human 다수결 대비 κ ≥ 0.6 (Landis-Koch 'substantial' 경계)을
# 넘어야 그 judge로 신규 트레이스 재측정(3단계)을 진행한다. 미달이면 judge 이전
# 실패가 파일럿의 결과다.
PHASE3_KAPPA_GATE = 0.6

# 원 논문 Table 2 — o1 (few shot), 사람 전문가 대비. 우리 표의 참조 행.
O1_PAPER = {"accuracy": 0.94, "recall": 0.77, "precision": 0.833, "f1": 0.80, "kappa": 0.77}


class PromptShim:
    """agentdash annotator의 _create_evaluation_prompt가 요구하는 self 대체물."""

    def __init__(self, definitions: str, examples: str):
        self.definitions = definitions
        self.examples = examples


def load_materials() -> tuple[list[dict], PromptShim]:
    human = json.loads((DATA / "mast" / "MAD_human_labelled_dataset.json").read_text())
    definitions = (DATA / "upstream" / "definitions.txt").read_text()
    examples = (DATA / "upstream" / "examples.txt").read_text()
    return human, PromptShim(definitions, examples)


def ground_truth(record: dict) -> dict[str, int]:
    """모드별 3인 다수결. 'failure mode' 필드의 ID 접두(1.1 등)가 조인 키."""
    gt: dict[str, int] = {}
    for ann in record["annotations"]:
        mode_id = ann["failure mode"].split(" ", 1)[0].strip()
        votes = [ann["annotator_1"], ann["annotator_2"], ann["annotator_3"]]
        gt[mode_id] = 1 if sum(bool(v) for v in votes) >= 2 else 0
    missing = [m for m in MODES if m not in gt]
    if missing:
        raise ValueError(f"라벨 누락 모드: {missing}")
    return gt


def record_key(i: int, record: dict) -> str:
    return f"{i:02d}_{record['mas_name']}_{str(record['trace_id'])[:40]}".replace("/", "_")


def create_prompt_reordered(shim: PromptShim, trace: str) -> str:
    """재구성 변형 — 정의·few-shot을 트레이스 *앞*으로.

    원 프롬프트와의 차이는 딱 둘: ① 블록 순서(정의→예시→트레이스), ② 그 순서를 지칭하던
    메타 문장 한 개. 나머지 문장은 agentdash 원문 그대로다. Flash가 원 순서(정의가 긴
    트레이스 *뒤*)에서 κ=0.056을 낸 것이 위치 효과인지 가리는 변형 (2026-07-08).
    """
    return (
        "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
        "In the traces, analyze the system behaviour."
        "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
        "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
        "Also tell me whether the task is successfully completed or not, as a binary yes or no."
        "First, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better. After the examples, I will provide you with the trace."
        "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
        "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
        "*** begin of things you should answer *** @@"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
        "B. Whether the task is successfully completed or not: <yes or no>"
        "C. Whether you encounter any of the failure modes or inefficiencies:"
        "1.1 Disobey Task Specification: <yes or no>"
        "1.2 Disobey Role Specification: <yes or no>"
        "1.3 Step Repetition: <yes or no>"
        "1.4 Loss of Conversation History: <yes or no>"
        "1.5 Unaware of Termination Conditions: <yes or no>"
        "2.1 Conversation Reset: <yes or no>"
        "2.2 Fail to Ask for Clarification: <yes or no>"
        "2.3 Task Derailment: <yes or no>"
        "2.4 Information Withholding: <yes or no>"
        "2.5 Ignored Other Agent's Input: <yes or no>"
        "2.6 Action-Reasoning Mismatch: <yes or no>"
        "3.1 Premature Termination: <yes or no>"
        "3.2 No or Incorrect Verification: <yes or no>"
        "3.3 Weak Verification: <yes or no>"
        "@@*** end of your answer ***"
        "An example answer is: \n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no \n"
        "C. \n"
        "1.1 no \n"
        "1.2 no \n"
        "1.3 no \n"
        "1.4 no \n"
        "1.5 no \n"
        "1.6 yes \n"
        "2.1 no \n"
        "2.2 no \n"
        "2.3 yes \n"
        "2.4 no \n"
        "2.5 no \n"
        "2.6 yes \n"
        "2.7 no \n"
        "3.1 no \n"
        "3.2 yes \n"
        "3.3 no \n"
        "Here are the explanations (definitions) of the failure modes and inefficiencies: \n"
        f"{shim.definitions} \n"
        "Here are some examples of the failure modes and inefficiencies: \n"
        f"{shim.examples}"
        "Here is the trace: \n"
        f"{trace}"
    )


def build_prompt(shim: PromptShim, trace: str, variant: str) -> str:
    if variant == "original":
        return MastAnnotator._create_evaluation_prompt(shim, trace)
    return create_prompt_reordered(shim, trace)


def call_anthropic(model: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {response.stop_details}")
    return "".join(b.text for b in response.content if b.type == "text")


def call_gemini(model: str, prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:  # 429/503 — 무료 쿼터 레이트리밋이 실측된 인프라
            if attempt == 3 or not any(c in str(e) for c in ("429", "503", "RESOURCE_EXHAUSTED")):
                raise
            wait = 30 * (attempt + 1)
            print(f"    Gemini {e.__class__.__name__} — {wait}s 대기 후 재시도", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def annotate(judge: str, records: list[dict], shim: PromptShim, variant: str) -> None:
    provider, model = JUDGES[judge]
    label = judge if variant == "original" else f"{judge}+{variant}"
    out_dir = RAW_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        key = record_key(i, record)
        out = out_dir / f"{key}.json"
        if out.exists():
            print(f"  [{i + 1}/{len(records)}] {key} — 저장본 재사용", flush=True)
            continue
        prompt = build_prompt(shim, record["trace"], variant)
        t0 = time.time()
        print(f"  [{i + 1}/{len(records)}] {key} — 프롬프트 {len(prompt):,}자 호출...", flush=True)
        text = call_anthropic(model, prompt) if provider == "anthropic" else call_gemini(model, prompt)
        out.write_text(json.dumps({
            "judge": judge, "model": model, "key": key, "variant": variant,
            "prompt_chars": len(prompt), "elapsed_s": round(time.time() - t0, 1),
            "response": text,
        }, ensure_ascii=False, indent=1))
        print(f"      완료 ({time.time() - t0:.0f}s)", flush=True)


def parse_response_strict(response: str) -> dict[str, int]:
    """줄 단위 엄격 파서 — 모드 ID로 시작하는 줄의 yes/no만 취한다.

    원 agentdash 파서(v0.1.0)는 패턴 `C\\..*?{mode}.*?(yes|no)`에서 모드 ID의 점을
    이스케이프하지 않고 DOTALL 비탐욕으로 최초 유사 매치에 걸려, 정상 형식 응답에서도
    다른 줄의 답을 오귀속한다(human 06번 트레이스에서 2.5·3.2 yes→no 오귀속 실측,
    2026-07-08). 이 파서는 그 인공물을 걷어낸 대조군 — 두 파서를 병기해 κ 차이를
    파서 노이즈와 판정 불일치로 분해한다.
    """
    import re

    parsed = {m: 0 for m in MODES}
    for line in response.splitlines():
        m = re.match(r"\s*(?:C\.)?(\d\.\d)\b[^a-zA-Z]*[^:]*?:?\s*.*?\b(yes|no)\b",
                     line, re.IGNORECASE)
        if m and m.group(1) in parsed:
            parsed[m.group(1)] = 1 if m.group(2).lower() == "yes" else 0
    return parsed


PARSERS = {
    "agentdash": lambda text: MastAnnotator._parse_response(None, text),
    "strict": parse_response_strict,
}


def collect_predictions(label: str, records: list[dict], parser: str) -> tuple[list[int], list[int], dict]:
    """저장된 raw(라벨 디렉토리) → 지정 파서 → (y_true, y_pred) 평탄화 + 모드별 집계."""
    y_true: list[int] = []
    y_pred: list[int] = []
    per_mode = {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for m in MODES}
    for i, record in enumerate(records):
        raw_path = RAW_DIR / label / f"{record_key(i, record)}.json"
        if not raw_path.exists():
            continue
        response = json.loads(raw_path.read_text())["response"]
        pred = PARSERS[parser](response)
        gt = ground_truth(record)
        for m in MODES:
            y_true.append(gt[m])
            y_pred.append(pred[m])
            cell = ("tp" if gt[m] and pred[m] else "fn" if gt[m] else
                    "fp" if pred[m] else "tn")
            per_mode[m][cell] += 1
    return y_true, y_pred, per_mode


def metrics_row(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {
        "n_cells": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 3),
    }


def report(records: list[dict]) -> None:
    from sklearn.metrics import cohen_kappa_score

    labels = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir()) if RAW_DIR.exists() else []
    rows: dict[str, dict] = {}
    preds: dict[str, list[int]] = {}
    mode_tables: dict[str, dict] = {}
    for label in labels:
        for parser in PARSERS:
            y_true, y_pred, per_mode = collect_predictions(label, records, parser)
            if not y_true:
                continue
            key = f"{label}/{parser}"
            rows[key] = metrics_row(y_true, y_pred)
            preds[key] = y_pred
            mode_tables[key] = per_mode

    lines = ["# Judge 캘리브레이션 — human 19편 (다수결 GT)", "",
             "원 파서(agentdash v0.1.0)는 모드 ID 점 미이스케이프 + DOTALL 비탐욕으로",
             "오귀속이 실측돼(스크립트 docstring), strict(줄 단위) 파서를 병기한다.", ""]
    lines += ["| judge/parser | model | cells | acc | prec | recall | F1 | κ |",
              "|---|---|---|---|---|---|---|---|"]
    lines.append(f"| o1 (논문 Table 2) | o1 few-shot | — | {O1_PAPER['accuracy']} | "
                 f"{O1_PAPER['precision']} | {O1_PAPER['recall']} | {O1_PAPER['f1']} | {O1_PAPER['kappa']} |")
    for key, row in rows.items():
        base = key.split("/")[0].split("+")[0]
        lines.append(f"| {key} | {JUDGES[base][1]} | {row['n_cells']} | {row['accuracy']} | "
                     f"{row['precision']} | {row['recall']} | {row['f1']} | {row['kappa']} |")

    for label in labels:
        a, s = f"{label}/agentdash", f"{label}/strict"
        if a in preds and s in preds:
            diff = sum(1 for x, y in zip(preds[a], preds[s]) if x != y)
            lines.append(f"\n- {label}: 두 파서 불일치 {diff}/{len(preds[a])} 셀 "
                         f"(파서 인공물의 크기)")

    done = [j for j in rows
            if j.endswith("/strict") and len(preds[j]) == len(records) * len(MODES)]
    if len(done) >= 2:
        lines += ["", "## judge 간 일치 (Cohen κ, strict 파서·완주 judge끼리)", ""]
        for a in done:
            for b in done:
                if a < b:
                    k = cohen_kappa_score(preds[a], preds[b])
                    lines.append(f"- {a} ↔ {b}: κ = {k:.3f}")

    lines += ["", "## 모드별 (judge / TP·FP·FN, GT 양성 수)", ""]
    gt_pos = {m: 0 for m in MODES}
    for record in records:
        gt = ground_truth(record)
        for m in MODES:
            gt_pos[m] += gt[m]
    header = "| mode | GT+ | " + " | ".join(rows) + " |"
    lines += [header, "|" + "---|" * (2 + len(rows))]
    for m in MODES:
        cells = [f"{mode_tables[j][m]['tp']}·{mode_tables[j][m]['fp']}·{mode_tables[j][m]['fn']}"
                 for j in rows]
        lines.append(f"| {m} | {gt_pos[m]} | " + " | ".join(cells) + " |")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(
        {"judges": rows, "o1_paper": O1_PAPER}, indent=1))
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judges", default="gemini,gemini-pro")
    parser.add_argument("--prompt-variant", default="original",
                        choices=["original", "reordered"])
    parser.add_argument("--smoke", action="store_true", help="트레이스 1편만")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path.home() / ".config" / "knowledge-mind" / ".env")
    records, shim = load_materials()
    print(f"human 트레이스 {len(records)}편, 정의 {len(shim.definitions):,}자, "
          f"few-shot {len(shim.examples):,}자")

    if not args.report_only:
        subset = records[:1] if args.smoke else records
        for judge in [j.strip() for j in args.judges.split(",") if j.strip()]:
            print(f"\n== judge: {judge} ({JUDGES[judge][1]}, {args.prompt_variant}) — {len(subset)}편")
            annotate(judge, subset, shim, args.prompt_variant)

    report(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
