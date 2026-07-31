"""
PromptAlign: Автоматическое итеративное выравнивание большой языковой модели 
под экспертную разметку без дообучения.

Сопутствующий код к научной статье:
Агафонов А.А., Хусаинов А.Р., Прокопьев Н.А. 
"PromptAlign: автоматическое итеративное выравнивание большой языковой модели 
под экспертную разметку без дообучения" // Электронные библиотеки, 2026.

Авторы: 
  А.А. Агафонов (a.a.agathonov@gmail.com)
  А.Р. Хусаинов (ahat2182@gmail.com)
  Н.А. Прокопьев (nikolai.prokopyev@gmail.com)
  
Лицензия: MIT License
Репозиторий: [ВСТАВИТЬ ССЫЛКУ НА GITHUB REPO]
"""

import json
import time
import re
import os
import hashlib
import glob
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from openai import OpenAI

# =============================================================================
# 1. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# =============================================================================

LABEL_NAMES = {0: "Информативный", 1: "Деструктивный", 2: "Конструктивный"}
LABEL_REVERSE = {"информативный": 0, "деструктивный": 1, "конструктивный": 2}

# Ограничения API и обработки
TPM_LIMIT = 12000
RPM_LIMIT = 28
MAX_TEXT_CHARS = 4000
MAX_PROMPT_LENGTH = 1500
MAX_COMPILER_TOKENS = 900

# Параметры конвейера (согласно Таблице 2 статьи)
MAX_ITERATIONS = 8
MAX_BPI_THRESHOLD = 0.15
MAX_CLASS_DOMINANCE = 0.65
MIN_CLASS_SHARE = 0.10
MAX_ERROR_RULES = 4
MAX_ERRORS_PER_TYPE = 2
MAX_ERRORS_TOTAL = 15
MAX_ACCUMULATED_RULES = 20
REBUILD_EVERY = 2  # Периодичность полной перекомпоновки

# Параметры статистической значимости и контроля баланса
ENABLE_REBUILD = True
N_BOOTSTRAP = 1000
MIN_P_BETTER = 0.75
MIN_POINT_DELTA = 0.015
MAX_MIN_F1_DROP = 0.05
BOOTSTRAP_SEED = 42
ENABLE_FEATURE_EXPANSION = False  # Отключено для предотвращения шума на малых данных

# Пути к файлам (относительные, для кроссплатформенности)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
STATE_FILE = os.path.join(CHECKPOINT_DIR, "pipeline_state.json")
HISTORY_FILE = os.path.join(BASE_DIR, "results", "prompt_history.json")


# =============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def remove_think_tags(text: str) -> str:
    """Удаляет теги рассуждений модели для очистки вывода."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"⟨think⟩.*?⟨/think⟩", "", text, flags=re.DOTALL)
    return text


def truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Обрезает текст до последнего полного предложения или слова."""
    if len(text) <= max_chars:
        return text
    cut = max(text.rfind(" ", 0, max_chars), text.rfind(".", 0, max_chars))
    return text[:cut if cut > 0 else max_chars] + "…"


# =============================================================================
# 3. УПРАВЛЕНИЕ СОСТОЯНИЕМ (CHECKPOINTING)
# =============================================================================

def save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"  [State] Сохранено: итерация={state['next_iteration']}, "
          f"best_val_score={state['best_val_score']:.4f}")


def load_state() -> Optional[Dict[str, Any]]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def clear_state() -> None:
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("  [State] Файл состояния удалён")


# =============================================================================
# 4. ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_dataset(file_map: Dict[int, List[str]], separator: str = "*********") -> pd.DataFrame:
    """Загружает тексты из файлов, разделённых маркером."""
    records = []
    for label, paths in file_map.items():
        for path in paths:
            full_path = os.path.join(BASE_DIR, "data", path)
            if not os.path.exists(full_path):
                print(f"  [Warning] Файл не найден: {full_path}")
                continue
            with open(full_path, "r", encoding="utf-8") as f:
                cur = []
                for line in f:
                    if separator in line:
                        txt = "\n".join(cur).strip()
                        if txt:
                            records.append({"text": txt, "label": label})
                        cur = []
                    else:
                        cur.append(line.rstrip())
                txt = "\n".join(cur).strip()
                if txt:
                    records.append({"text": txt, "label": label})
    
    df = pd.DataFrame(records)
    print(f"Загружено {len(df)} текстов. Распределение: {dict(Counter(df['label']))}")
    return df


# =============================================================================
# 5. КЛИЕНТ API С КОНТРОЛЕМ ЛИМИТОВ (Rate Limiting)
# =============================================================================

class RateLimitedLLMClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.tpm_used = 0
        self.tpm_reset = time.time()
        self.rpm_count = 0
        self.rpm_reset = time.time()

    def _refresh_windows(self):
        now = time.time()
        if now - self.tpm_reset >= 60:
            self.tpm_used = 0
            self.tpm_reset = now
        if now - self.rpm_reset >= 60:
            self.rpm_count = 0
            self.rpm_reset = now

    def _wait_if_needed(self, est_tokens: int):
        self._refresh_windows()
        if self.rpm_count >= RPM_LIMIT:
            w = 60 - (time.time() - self.rpm_reset) + 0.5
            print(f"[{_get_timestamp()}] Лимит RPM. Ожидание {w:.1f}с")
            time.sleep(max(w, 0))
            self._refresh_windows()
        if self.tpm_used + est_tokens > TPM_LIMIT:
            w = 60 - (time.time() - self.tpm_reset) + 0.5
            print(f"[{_get_timestamp()}] Лимит TPM. Ожидание {w:.1f}с")
            time.sleep(max(w, 0))
            self._refresh_windows()

    def call(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0, max_retries: int = 5) -> str:
        est = int(len(prompt) / 3.5) + max_tokens + 20
        self._wait_if_needed(est)
        
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                )
                usage = completion.usage
                real_tokens = usage.total_tokens if usage else est
                self.tpm_used += real_tokens
                self.rpm_count += 1
                
                raw = completion.choices[0].message.content or ""
                print(f"[{_get_timestamp()}] OK {time.time()-t0:.1f}с | tokens={real_tokens}")
                return raw
            except Exception as e:
                if "429" in str(e):
                    w = 5 * (2 ** attempt)
                    print(f"[{_get_timestamp()}] Ошибка 429 — повтор через {w}с")
                    time.sleep(w)
                else:
                    print(f"[{_get_timestamp()}] Ошибка API: {e}")
                    return ""
        print("[!] Превышено число попыток запроса к API")
        return ""


# =============================================================================
# 6. ОБРАБОТКА И КЛАССИФИКАЦИЯ
# =============================================================================

def parse_label(response: str) -> int:
    cleaned = remove_think_tags(response).lower().strip()
    for name, idx in LABEL_REVERSE.items():
        if name in cleaned:
            return idx
    for pref, idx in [("инф", 0), ("дестр", 1), ("констр", 2)]:
        if cleaned.startswith(pref):
            return idx
    return -1


def load_checkpoint(path: str) -> List[Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("predictions", [])
    return []


def save_checkpoint(path: str, preds: List[Any], extra_meta: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"predictions": preds, "ts": time.time()}
    if extra_meta:
        data.update(extra_meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def classify_batch(texts: List[str], prompt_fn: callable, client: RateLimitedLLMClient, 
                   checkpoint_path: str, delay: float = 0.3, max_tokens: int = 1500) -> List[int]:
    preds = load_checkpoint(checkpoint_path)
    preds = [None if p == -1 else p for p in preds]

    total = len(texts)
    for i in range(total):
        if i < len(preds) and preds[i] is not None:
            continue
        if i % 25 == 0:
            print(f"  [{_get_timestamp()}] Обработка: {i}/{total}")
            
        raw = client.call(prompt_fn(texts[i]), max_tokens=max_tokens, temperature=0.0)
        label = parse_label(raw)
        
        if i >= len(preds):
            preds.extend([None] * (i + 1 - len(preds)))
        preds[i] = label
        save_checkpoint(checkpoint_path, preds)
        time.sleep(delay)
        
    return [p if p is not None else -1 for p in preds[:total]]


# =============================================================================
# 7. БАЗОВЫЕ ПРОМПТЫ И САНИТАРНАЯ ПРОВЕРКА
# =============================================================================

BASE_PROMPT_TEMPLATE = """/no_think
Определи тип поведения автора текста. Есть три класса:

• Деструктивный — прямые оскорбления, угрозы, унижения, призывы к разрушению, паника, агрессия, а также выражение безысходности, эмоционального страдания или беспомощности без попыток решения.
• Информативный — аналитика, факты, статистика, цифры, описание событий или проектов; НЕТ личной позиции автора и призывов к действию.
• Конструктивный — предложения решений, эмпатия, поддержка, призывы к созиданию, спокойный тон. Сюда же относится критика с аргументацией и предложением улучшений.

Ответь ОДНИМ СЛОВОМ (Информативный / Деструктивный / Конструктивный).

Текст: {text}"""


def make_classify_fn(template: str) -> callable:
    return lambda text: template.format(text=truncate(text))


def sanity_check_prompt(preds: List[int], label: str = "") -> bool:
    valid = [p for p in preds if p != -1]
    total = len(preds)
    if not valid:
        print(f"  [Санити-чек{label}] ПРОВАЛ: все предсказания -1")
        return False
        
    failed = total - len(valid)
    if failed / total > 0.2:
        print(f"  [Санити-чек{label}] ПРОВАЛ: {failed}/{total} ответов не распознаны")
        return False
        
    counts = Counter(valid)
    max_frac = max(counts.values()) / total
    dominant = max(counts, key=counts.get)
    min_class_frac = min(counts.get(c, 0) for c in [0, 1, 2]) / total
    
    print(f"  [Санити-чек{label}] Распределение: "
          f"{ {LABEL_NAMES[k]: v for k, v in sorted(counts.items())} }")
          
    if max_frac > MAX_CLASS_DOMINANCE:
        print(f"  [Санити-чек{label}] ПРОВАЛ: доминирование одного класса ({max_frac:.1%})")
        return False
    if min_class_frac < MIN_CLASS_SHARE:
        print(f"  [Санити-чек{label}] ПРОВАЛ: минорный класс слишком мал ({min_class_frac:.1%})")
        return False
    return True


# =============================================================================
# 8. СТАТИСТИЧЕСКАЯ ОЦЕНКА И МЕТРИКИ
# =============================================================================

def bootstrap_macro_f1_diff(y_true: List, y_pred_old: List, y_pred_new: List, 
                            n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    arr_true = np.array(y_true)
    arr_old = np.array(y_pred_old)
    arr_new = np.array(y_pred_new)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        f1_old = f1_score(arr_true[idx], arr_old[idx], average='macro', labels=[0, 1, 2], zero_division=0)
        f1_new = f1_score(arr_true[idx], arr_new[idx], average='macro', labels=[0, 1, 2], zero_division=0)
        diffs[b] = f1_new - f1_old

    p_better = float(np.mean(diffs > 0))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    mean_diff = float(np.mean(diffs))
    return p_better, float(ci_low), float(ci_high), mean_diff


def compute_metrics(y_true: List, y_pred: List, prefix: str = "") -> Dict[str, Any]:
    mask = np.array(y_pred) != -1
    yt, yp = np.array(y_true)[mask], np.array(y_pred)[mask]
    if len(yt) == 0:
        return {}
    pc = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0).tolist()
    return {
        f"{prefix}macro_f1": float(np.mean(pc)),
        f"{prefix}f1_info": pc[0],
        f"{prefix}f1_destr": pc[1],
        f"{prefix}f1_constr": pc[2],
        f"{prefix}min_f1": float(min(pc)),
        f"{prefix}kappa": float(cohen_kappa_score(yt, yp)),
        f"{prefix}bal_acc": float(balanced_accuracy_score(yt, yp)),
        f"{prefix}confusion": confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(),
        f"{prefix}n_valid": int(mask.sum()),
        f"{prefix}n_failed": int((~mask).sum()),
    }


def compute_bpi(metrics_before: Dict, metrics_after: Dict, prefix: str = "") -> float:
    """Вычисляет индекс дисбаланса классов (Blanket Pulling Index)."""
    keys = [f"{prefix}f1_info", f"{prefix}f1_destr", f"{prefix}f1_constr"]
    delta = [metrics_after.get(k, 0) - metrics_before.get(k, 0) for k in keys]
    return float(max(delta) - min(delta))


def compute_cav(metrics: Dict, prefix: str = "") -> float:
    """Вычисляет дисперсию классовых метрик (Class Accuracy Variance)."""
    vals = [metrics.get(f"{prefix}f1_info", 0), metrics.get(f"{prefix}f1_destr", 0), metrics.get(f"{prefix}f1_constr", 0)]
    return float(np.var(vals))


# =============================================================================
# 9. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ И СУРРОГАТНЫЕ МОДЕЛИ
# =============================================================================

def rule_based_features(text: str) -> Dict[str, float]:
    """Извлекает детерминированные лингвистические признаки (Приложение Б.1)."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = text.lower().split()
    n_words = max(len(words), 1)
    return {
        "rb_exclamation": text.count("!") / max(len(sentences), 1),
        "rb_question": text.count("?") / max(len(sentences), 1),
        "rb_caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "rb_negative_words": len(re.findall(r"\b(не|нельзя|невозможно|запрещено|никогда|никто|ничего)\b", text.lower())) / n_words,
        "rb_first_person": len(re.findall(r"\b(я|мне|мой|моё|меня|мною)\b", text.lower())) / n_words,
        "rb_we_form": len(re.findall(r"\b(мы|нам|наш|нашего|нашей|нас)\b", text.lower())) / n_words,
        "rb_imperative": len(re.findall(r"\b(давайте|нужно|необходимо|следует|должны|обязаны)\b", text.lower())) / n_words,
        "rb_avg_sent_len": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        "rb_text_len_log": np.log1p(len(text)),
        "rb_ellipsis": text.count("…") + text.count("..."),
        "rb_quotes": text.count("«") + text.count('"'),
        "rb_numbers": len(re.findall(r"\b\d+[\d,.]*\b", text)) / n_words,
        "rb_url": int(bool(re.search(r"https?://|www\.", text))),
    }


BASE_FEATURE_EXTRACTOR_PROMPT = """Ты — лингвистический анализатор. Извлеки признаки текста в формате JSON.
Если признак отсутствует — false или 0. Отвечай ТОЛЬКО валидным JSON без пояснений и без markdown.
Признаки: has_aggression, has_indirect_aggression, has_empathy, has_solution, has_facts,
has_author_position, has_call_to_action (bool), speech_act (утверждение/вопрос/призыв/критика/оценка/совет),
irony_probability, emotional_intensity (0..1){extra_fields}.
Текст: {text}"""


def extract_llm_features(text: str, client: RateLimitedLLMClient, extra_features: List[Dict] = None, n_votes: int = 2) -> Dict:
    """Извлекает семантические признаки с помощью LLM с усреднением результатов (n_votes)."""
    extra = ""
    if extra_features:
        extra = ", " + ", ".join(f"{f['name']} (bool)" for f in extra_features)
    prompt = BASE_FEATURE_EXTRACTOR_PROMPT.replace("{extra_fields}", extra).format(text=truncate(text, 2000))
    
    results = []
    for _ in range(n_votes):
        raw = client.call(prompt, max_tokens=800, temperature=0.1)
        raw = remove_think_tags(raw)
        raw_clean = re.sub(r"```json|```", "", raw).strip()
        try:
            results.append(json.loads(raw_clean))
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
            if m:
                try:
                    results.append(json.loads(m.group()))
                except Exception:
                    pass
                    
    if not results:
        return {}
        
    agg = {}
    bool_keys = ["has_aggression", "has_indirect_aggression", "has_empathy", "has_solution", 
                 "has_facts", "has_author_position", "has_call_to_action"]
    if extra_features:
        bool_keys += [f["name"] for f in extra_features]
    float_keys = ["irony_probability", "emotional_intensity"]
    
    for k in bool_keys:
        vals = [r.get(k, False) for r in results]
        agg[k] = int(sum(vals) > len(vals) / 2)
    for k in float_keys:
        vals = [float(r.get(k, 0)) for r in results]
        agg[k] = float(np.mean(vals))
        
    acts = [r.get("speech_act", "утверждение") for r in results]
    acts = [a[0] if isinstance(a, list) else str(a) for a in acts]
    agg["speech_act"] = Counter(acts).most_common(1)[0][0]
    return agg


def build_feature_matrix(texts: List[str], labels_true: List, labels_llm: List, 
                         client: RateLimitedLLMClient, cache_path: str, 
                         errors_only: bool = False, extra_features: List[Dict] = None) -> pd.DataFrame:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
            
    rows = []
    for i, (text, lt, ll) in enumerate(zip(texts, labels_true, labels_llm)):
        is_error = (lt != ll and ll != -1)
        if errors_only and not is_error:
            continue
            
        key = hashlib.md5(text[:200].encode()).hexdigest()
        if key in cache:
            llm_feats = cache[key]
        else:
            llm_feats = extract_llm_features(text, client, extra_features, n_votes=2)
            cache[key] = llm_feats
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
                
        row = {**rule_based_features(text), **llm_feats, "true_label": lt, "llm_label": ll, "is_error": int(is_error)}
        rows.append(row)
        if i % 20 == 0:
            print(f"  [{_get_timestamp()}] Извлечение признаков: {i+1}/{len(texts)}")
            
    return pd.DataFrame(rows)


CATEGORICAL_FEATURES = ["speech_act"]

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    drop = ["true_label", "llm_label", "is_error"] + CATEGORICAL_FEATURES
    X = df.drop(columns=[c for c in drop if c in df.columns])
    for cat in CATEGORICAL_FEATURES:
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat], prefix=cat)
            X = pd.concat([X, dummies], axis=1)
    return X.fillna(0).astype(float)


def fit_logreg(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    return LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                              class_weight="balanced", random_state=42, max_iter=500).fit(X, y)


def extract_delta_weights(clf_exp: LogisticRegression, clf_llm: LogisticRegression, 
                          feat_names: List[str], top_n: int = 8) -> List[Dict]:
    w_exp = np.mean(np.abs(clf_exp.coef_), axis=0)
    w_llm = np.mean(np.abs(clf_llm.coef_), axis=0)
    delta = w_exp - w_llm
    idx = np.argsort(delta)[::-1][:top_n]
    return [{"feature": feat_names[i], "delta": float(delta[i]), 
             "w_expert": float(w_exp[i]), "w_llm": float(w_llm[i])} for i in idx]


# =============================================================================
# 10. МЕТА-АНАЛИЗ ОШИБОК И КОМПИЛЯЦИЯ ПРОМПТА
# =============================================================================

def stratified_errors(errors_df: pd.DataFrame, max_per_type: int = MAX_ERRORS_PER_TYPE) -> pd.DataFrame:
    parts = []
    for wrong in [0, 1, 2]:
        for correct in [0, 1, 2]:
            if wrong == correct: continue
            subset = errors_df[(errors_df["llm_label"] == wrong) & (errors_df["true_label"] == correct)].head(max_per_type)
            if len(subset):
                parts.append(subset)
    if not parts:
        return errors_df.head(max_per_type * 6)
    return pd.concat(parts).reset_index(drop=True)


META_ANALYSIS_PROMPT = """/no_think
Ты — эксперт по анализу ошибок классификатора.
Проанализируй текст, неверный ответ LLM и правильную метку эксперта.
Найди системную причину ошибки, которая может повторяться в других текстах.
Сформулируй правило, которое поможет избежать такой ошибки в будущем.
Отвечай ТОЛЬКО валидным JSON (без markdown и пояснений):
{{"reasons": ["системная причина"], "rule": "Если [обобщённый признак], то класс '{correct}', даже при наличии [другой признак]."}}

Текст: {text}
Ответ LLM (неверный): {wrong}
Правильная метка: {correct}"""


def _parse_json_safe(raw: str) -> Optional[Dict]:
    raw = remove_think_tags(raw)
    raw = re.sub(r"```json|```", "", raw).strip()
    for pattern in [r"\{.*\}", r'\{[^{}]*"rule"[^{}]*\}']:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                continue
    rule_match = re.search(r'"rule"\s*:\s*"([^"]+)"', raw)
    if rule_match:
        return {"rule": rule_match.group(1)}
    return None


def analyze_errors_llm(errors_df: pd.DataFrame, client: RateLimitedLLMClient, 
                       cache_path: str, max_errors: int = MAX_ERRORS_TOTAL) -> List[str]:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) > 0:
            return cached
        else:
            os.remove(cache_path)
            
    rules = []
    sample = errors_df.head(max_errors)
    print(f"  [Мета-анализ] Анализ {len(sample)} ошибок...")
    
    for _, row in sample.iterrows():
        prompt = META_ANALYSIS_PROMPT.format(
            text=truncate(row.get("text", ""), 2000),
            wrong=LABEL_NAMES.get(int(row["llm_label"]), "?"),
            correct=LABEL_NAMES.get(int(row["true_label"]), "?")
        )
        parsed = None
        for attempt in range(3):
            raw = client.call(prompt, max_tokens=250, temperature=0.0)
            parsed = _parse_json_safe(raw)
            if parsed and "rule" in parsed:
                break
            time.sleep(1)
        if parsed and "rule" in parsed:
            rules.append(parsed["rule"])
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)
    return rules


# Словарь семантической интерпретации признаков (согласно методологии статьи)
FEATURE_SEMANTICS = {
    "has_facts": "Наличие фактов, цифр и статистики — главный признак Информативного класса.",
    "has_solution": "Конструктивный класс обязательно содержит явное предложение решения или помощи.",
    "has_empathy": "Эмпатия и поддержка — маркеры Конструктивного класса, даже при критике.",
    "has_aggression": "Прямая агрессия обычно указывает на Деструктивный класс, если нет конструктивной цели.",
    "has_call_to_action": "Призыв к действию может быть и Конструктивным (созидание), и Деструктивным (разрушение).",
    "rb_negative_words": "Отрицательные слова (не, нельзя) сами по себе не делают текст Деструктивным.",
    "rb_exclamation": "Восклицания могут быть и в Конструктивных призывах, не всегда агрессия.",
    "has_author_position": "Чёткая авторская позиция с аргументами — скорее Информативный или Конструктивный класс.",
    "irony_probability": "Ирония и сарказм часто маскируют Деструктивный класс, но могут быть в Конструктивной критике.",
    "speech_act_критика": "Критика без предложения решения — скорее Деструктивный; с решением — Конструктивный.",
    "has_indirect_aggression": "Скрытая агрессия (намёки, пассивная агрессия) — признак Деструктивного поведения.",
}


def compile_prompt(current_prompt: str, feature_weights: List[Dict], error_rules: List[str], 
                   client: RateLimitedLLMClient, cache_path: str, iteration: int) -> str:
    """Инкрементальная компиляция промпта с добавлением новых правил."""
    h = hashlib.md5((json.dumps(error_rules) + str(iteration)).encode()).hexdigest()[:8]
    versioned_cache = cache_path.replace(".txt", f"_{h}.txt")
    
    if os.path.exists(versioned_cache):
        with open(versioned_cache, "r", encoding="utf-8") as f:
            return _postprocess_prompt(f.read().strip())
    if not error_rules:
        return current_prompt

    feature_lines = []
    for fw in feature_weights[:5]:
        if fw['delta'] > 0.05:
            desc = FEATURE_SEMANTICS.get(fw['feature'], f"Признак '{fw['feature']}' важен для экспертов.")
            feature_lines.append(f"  • {desc}")
    feature_lines_str = "\n".join(feature_lines) if feature_lines else "Дополнительных указаний по признакам нет."
    rules_text = "\n".join(f"  • {r}" for r in error_rules[:MAX_ERROR_RULES])

    COMPILER_PROMPT = f"""/no_think
Ты — генератор промптов для LLM-классификатора на русском языке.
Улучши ДАННЫЙ промпт классификации, добавив к нему предоставленные правила.
Сохрани исходную логику и структуру.

Текущий промпт:
{current_prompt}

Правила из анализа ошибок (добавь напрямую, не более 4):
{rules_text}

Признаки, важные для экспертов (подсказка):
{feature_lines_str}

СТРОГИЕ требования:
- Не удаляй существующую логику.
- Не добавляй правил, которых нет в списке.
- Заверши промпт фразой: "Ответь ОДНИМ СЛОВОМ (Информативный / Деструктивный / Конструктивный)."
- Удели особое внимание Конструктивному классу: его часто путают с Информативным. Для Конструктивного обязательно наличие явного предложения решения или поддержки.
- Длина итогового промпта не более {MAX_PROMPT_LENGTH} символов.
- Оставь подстановку "{{text}}" для текста.
Напиши только текст промпта, без пояснений."""

    raw = client.call(COMPILER_PROMPT, max_tokens=MAX_COMPILER_TOKENS, temperature=0.2)
    new_prompt = _postprocess_prompt(remove_think_tags(raw).strip())

    if len(new_prompt) > MAX_PROMPT_LENGTH:
        shrink_prompt = f"""/no_think
Сократи следующий промпт до длины не более {MAX_PROMPT_LENGTH-100} символов, сохранив все ключевые правила, подстановку "{{text}}" и завершающую фразу "Ответь ОДНИМ СЛОВОМ (Информативный / Деструктивный / Конструктивный)."
Промпт:
{new_prompt}
Сокращённый промпт:"""
        raw2 = client.call(shrink_prompt, max_tokens=MAX_COMPILER_TOKENS, temperature=0.0)
        new_prompt2 = _postprocess_prompt(remove_think_tags(raw2).strip())
        if len(new_prompt2) <= MAX_PROMPT_LENGTH:
            new_prompt = new_prompt2

    with open(versioned_cache, "w", encoding="utf-8") as f:
        f.write(new_prompt)
    return new_prompt


def rebuild_prompt_from_scratch(base_template: str, all_rules: List[str], 
                                feature_weights: List[Dict], client: RateLimitedLLMClient, cache_path: str) -> str:
    """Полная перекомпоновка промпта для устранения противоречий и избыточности."""
    h = hashlib.md5((json.dumps(all_rules) + "rebuild").encode()).hexdigest()[:8]
    versioned_cache = cache_path.replace(".txt", f"_rebuild_{h}.txt")
    
    if os.path.exists(versioned_cache):
        with open(versioned_cache, "r", encoding="utf-8") as f:
            return _postprocess_prompt(f.read().strip())

    feature_lines = []
    for fw in feature_weights[:5]:
        if fw['delta'] > 0.05:
            desc = FEATURE_SEMANTICS.get(fw['feature'], f"Признак '{fw['feature']}' важен для экспертов.")
            feature_lines.append(f"  • {desc}")
    feature_lines_str = "\n".join(feature_lines) if feature_lines else "Дополнительных указаний по признакам нет."
    rules_text = "\n".join(f"  • {r}" for r in all_rules) if all_rules else "Правил пока нет."

    REBUILD_PROMPT = f"""/no_think
Ты — генератор промптов для LLM-классификатора на русском языке.
Создай новый, оптимальный промпт классификации текстов на основе базового шаблона и ВСЕХ накопленных правил.
Обобщи правила, убери дубликаты, оставь только самые полезные формулировки.

Базовый шаблон:
{base_template}

Все накопленные правила (обобщи их):
{rules_text}

Важные признаки от экспертов:
{feature_lines_str}

Требования:
- Сохрани три класса и их определения.
- Добавь обобщённые правила (чёткие, без повторов).
- Обязательно подчеркни: Конструктивный класс часто путают с Информативным. Критика + предложение решения = Конструктивный. Просто факты = Информативный.
- Заверши фразой: "Ответь ОДНИМ СЛОВОМ (Информативный / Деструктивный / Конструктивный)."
- Длина не более {MAX_PROMPT_LENGTH} символов.
- Оставь подстановку "{{text}}" для текста.
Напиши только текст промпта, без пояснений."""

    raw = client.call(REBUILD_PROMPT, max_tokens=MAX_COMPILER_TOKENS, temperature=0.2)
    new_prompt = _postprocess_prompt(remove_think_tags(raw).strip())

    if len(new_prompt) > MAX_PROMPT_LENGTH:
        shrink_prompt = f"/no_think\nСократи промпт до {MAX_PROMPT_LENGTH-100} символов, сохранив все правила, подстановку '{{text}}' и завершающую фразу."
        raw2 = client.call(shrink_prompt, max_tokens=MAX_COMPILER_TOKENS, temperature=0.0)
        new_prompt2 = _postprocess_prompt(remove_think_tags(raw2).strip())
        if len(new_prompt2) <= MAX_PROMPT_LENGTH:
            new_prompt = new_prompt2

    with open(versioned_cache, "w", encoding="utf-8") as f:
        f.write(new_prompt)
    return new_prompt


def _postprocess_prompt(prompt: str) -> str:
    required = "Ответь ОДНИМ СЛОВОМ (Информативный / Деструктивный / Конструктивный)."
    if required.lower() not in prompt.lower():
        prompt = prompt.rstrip() + "\n\n" + required
    if "{text}" not in prompt:
        prompt = prompt.rstrip() + "\n\nТекст: {text}"
    prompt = re.sub(r'(/no_think\s*)+', '/no_think\n', prompt, flags=re.IGNORECASE).strip()
    if not prompt.lower().startswith("/no_think"):
        prompt = "/no_think\n" + prompt
    return prompt


# =============================================================================
# 11. ОСНОВНОЙ ЦИКЛ КОНВЕЙЕРА
# =============================================================================

def run_prompt_align(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, 
                     client: RateLimitedLLMClient, output_dir: str = "results") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    X_train, X_val, X_test = df_train["text"].tolist(), df_val["text"].tolist(), df_test["text"].tolist()

    state = load_state()
    if state and state.get("done"):
        print("  [Resume] Полный цикл завершён ранее. Удалите состояние для перезапуска.")
        return {}

    history = []
    extra_features = []
    all_accumulated_rules = []

    if state is None:
        print("\n[Этап 1] Базовая классификация (Zero-Shot)")
        base_fn = make_classify_fn(BASE_PROMPT_TEMPLATE)
        
        y_train_base = classify_batch(X_train, base_fn, client, os.path.join(CHECKPOINT_DIR, "train_base.json"))
        y_val_base = classify_batch(X_val, base_fn, client, os.path.join(CHECKPOINT_DIR, "val_base.json"))

        if not sanity_check_prompt(y_val_base, " val_base"):
            raise ValueError("Базовая модель не прошла санитарную проверку.")

        metrics_val_base = compute_metrics(df_val["label"], y_val_base, prefix="val_")
        print(f"Базовый val macro-F1: {metrics_val_base['val_macro_f1']:.4f}")
        
        best_prompt = BASE_PROMPT_TEMPLATE
        best_val_score = metrics_val_base["val_macro_f1"]
        y_val_best = list(y_val_base)
        train_preds = list(y_train_base)
        
        history = [{"iteration": 0, "prompt": best_prompt, "metrics": metrics_val_base, "bpi": 0.0}]
        save_state({
            "best_prompt": best_prompt, "best_val_score": best_val_score, "y_val_best": y_val_best,
            "train_preds": train_preds, "next_iteration": 1, "history": history,
            "extra_features": extra_features, "all_accumulated_rules": all_accumulated_rules, "done": False,
        })
    else:
        best_prompt = state["best_prompt"]
        best_val_score = state["best_val_score"]
        y_val_best = state.get("y_val_best")
        train_preds = state["train_preds"]
        start_iteration = state["next_iteration"]
        history = state.get("history", [])
        extra_features = state.get("extra_features", [])
        all_accumulated_rules = state.get("all_accumulated_rules", [])

    for iteration in range(start_iteration, MAX_ITERATIONS + 1):
        print(f"\n[Этап 2] Итерация конвейера {iteration}")

        feat_cache = os.path.join(CHECKPOINT_DIR, f"features_iter{iteration}.json")
        feat_df = build_feature_matrix(X_train, df_train["label"].tolist(), train_preds, client, 
                                       feat_cache, errors_only=False, extra_features=extra_features)
        if len(feat_df) < 10:
            print("Слишком мало данных для построения суррогатной модели"); break

        X_feat = prepare_X(feat_df)
        valid = X_feat["llm_label"] != -1
        Xv, ye, yl = X_feat[valid], feat_df["true_label"][valid].astype(int), feat_df["llm_label"][valid].astype(int)

        clf_exp = fit_logreg(Xv, ye)
        clf_llm = fit_logreg(Xv, yl)
        delta_weights = extract_delta_weights(clf_exp, clf_llm, list(X_feat.columns), top_n=8)

        errors_df = feat_df[feat_df["is_error"] == 1].copy()
        errors_df["text"] = [X_train[i] for i in errors_df.index]
        errors_df = stratified_errors(errors_df, MAX_ERRORS_PER_TYPE)

        error_rules_cache = os.path.join(CHECKPOINT_DIR, f"error_rules_iter{iteration}.json")
        error_rules = analyze_errors_llm(errors_df, client, error_rules_cache, max_errors=MAX_ERRORS_TOTAL)

        for rule in error_rules:
            if rule not in all_accumulated_rules:
                all_accumulated_rules.append(rule)
        if len(all_accumulated_rules) > MAX_ACCUMULATED_RULES:
            all_accumulated_rules = all_accumulated_rules[-MAX_ACCUMULATED_RULES:]

        do_rebuild = (ENABLE_REBUILD and iteration > 1 and iteration % REBUILD_EVERY == 0 and len(all_accumulated_rules) > 5)
        if do_rebuild:
            print("  -> Полная перекомпоновка промпта (обобщение всех правил)")
            rebuild_cache = os.path.join(CHECKPOINT_DIR, f"rebuild_prompt_iter{iteration}.txt")
            new_prompt = rebuild_prompt_from_scratch(BASE_PROMPT_TEMPLATE, all_accumulated_rules, delta_weights, client, rebuild_cache)
        else:
            compiled_cache = os.path.join(CHECKPOINT_DIR, f"compiled_prompt_iter{iteration}.txt")
            new_prompt = compile_prompt(best_prompt, delta_weights, error_rules, client, compiled_cache, iteration)

        new_classify_fn = make_classify_fn(new_prompt)
        val_preds_new = classify_batch(X_val, new_classify_fn, client, os.path.join(CHECKPOINT_DIR, f"val_iter{iteration}.json"))

        if not sanity_check_prompt(val_preds_new, f" iter{iteration}"):
            continue

        metrics_new = compute_metrics(df_val["label"], val_preds_new, prefix="val_")
        metrics_old = compute_metrics(df_val["label"], y_val_best, prefix="val_")
        
        if any(metrics_new.get(k, 1) == 0.0 for k in ["val_f1_info", "val_f1_destr", "val_f1_constr"]):
            print("  -> Класс схлопнулся, промпт отброшен")
            continue

        point_delta = metrics_new["val_macro_f1"] - metrics_old["val_macro_f1"]
        p_better, ci_low, ci_high, mean_diff = bootstrap_macro_f1_diff(df_val["label"].tolist(), y_val_best, val_preds_new)
        min_f1_old, min_f1_new = metrics_old["val_min_f1"], metrics_new["val_min_f1"]

        accept = (p_better >= MIN_P_BETTER) or (point_delta > MIN_POINT_DELTA and min_f1_new >= min_f1_old - MAX_MIN_F1_DROP)

        if accept:
            per_class_delta = [metrics_new.get("val_f1_info", 0) - metrics_old.get("val_f1_info", 0),
                               metrics_new.get("val_f1_destr", 0) - metrics_old.get("val_f1_destr", 0),
                               metrics_new.get("val_f1_constr", 0) - metrics_old.get("val_f1_constr", 0)]
            worst_class_delta = min(per_class_delta)
            bpi = compute_bpi(metrics_old, metrics_new, "val_")
            
            if worst_class_delta < -0.03 and bpi > MAX_BPI_THRESHOLD:
                print(f"  -> Промпт отклонён: деградация миноритарного класса ({worst_class_delta:+.4f}) при BPI={bpi:.4f}")
                accept = False

        if not accept:
            print(f"  -> Промпт отклонён по тесту значимости (P={p_better:.1%}, Δ={point_delta:+.4f})")
            history.append({"iteration": iteration, "prompt": new_prompt, "metrics": metrics_new, "note": "rejected"})
            continue

        print(f"  -> Промпт принят (значимое улучшение, Δ={point_delta:+.4f})")
        history.append({"iteration": iteration, "prompt": new_prompt, "metrics": metrics_new, "bpi": compute_bpi(metrics_old, metrics_new, "val_"), "note": "accepted"})
        
        best_prompt = new_prompt
        best_val_score = metrics_new["val_macro_f1"]
        y_val_best = list(val_preds_new)
        
        save_state({
            "best_prompt": best_prompt, "best_val_score": best_val_score, "y_val_best": y_val_best,
            "train_preds": train_preds, "next_iteration": iteration + 1, "history": history,
            "extra_features": extra_features, "all_accumulated_rules": all_accumulated_rules, "done": False,
        })

    print("\n[Этап 3] Финальное тестирование на отложенной выборке")
    final_fn = make_classify_fn(best_prompt)
    test_preds = classify_batch(X_test, final_fn, client, os.path.join(CHECKPOINT_DIR, "test_final.json"))
    metrics_test = compute_metrics(df_test["label"], test_preds, prefix="test_")
    
    test_base_preds = classify_batch(X_test, make_classify_fn(BASE_PROMPT_TEMPLATE), client, os.path.join(CHECKPOINT_DIR, "test_base.json"))
    base_test = compute_metrics(df_test["label"], test_base_preds, prefix="test_")
    
    p_better_test, ci_low_test, ci_high_test, mean_diff_test = bootstrap_macro_f1_diff(df_test["label"].tolist(), test_base_preds, test_preds)
    
    summary = {
        "best_prompt": best_prompt, "metrics_test": metrics_test, "metrics_base": base_test,
        "final_bpi": compute_bpi(base_test, metrics_test, "test_"),
        "test_significance": {"p_better": p_better_test, "ci_low": ci_low_test, "ci_high": ci_high_test, "mean_diff": mean_diff_test}
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "best_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(best_prompt)
        
    save_state({**state, "done": True} if state else {"done": True})
    
    print("\n" + "="*70)
    print(f"{'Метод':<35} {'macro-F1':>8} {'min-F1':>8} {'BPI':>6} {'κ':>6}")
    print("-"*70)
    print(f"{'Baseline (zero-shot)':<35} {base_test.get('test_macro_f1',0):8.3f} {base_test.get('test_min_f1',0):8.3f} {'—':>6} {base_test.get('test_kappa',0):6.3f}")
    print(f"{'PromptAlign (предложенный)':<35} {metrics_test.get('test_macro_f1',0):8.3f} {metrics_test.get('test_min_f1',0):8.3f} {summary['final_bpi']:6.3f} {metrics_test.get('test_kappa',0):6.3f}")
    print("="*70)
    
    return summary


# =============================================================================
# 12. ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    # Инициализация API ключа (поддержка Colab и локального окружения)
    try:
        from google.colab import userdata
        API_KEY = userdata.get('DEEPSEEK_API_KEY')
    except Exception:
        API_KEY = os.environ.get("DEEPSEEK_API_KEY") or input("Введите DeepSeek API key: ").strip()

    client = RateLimitedLLMClient(api_key=API_KEY, model="deepseek-chat")

    # Загрузка данных (ожидается структура папок: ./data/0_1.txt, ./data/1_1.txt и т.д.)
    FILE_MAP = {
        0: ["0_1.txt", "0_2.txt"], 
        1: ["1_1.txt", "1_2.txt"], 
        2: ["2_1.txt", "2_2.txt"]
    }
    
    print("Загрузка датасета...")
    df = load_dataset(FILE_MAP)
    
    print("Разделение на выборки (70% train, 30% val, 20% test от общего объёма)...")
    df_trainval, df_test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    df_train, df_val = train_test_split(df_trainval, test_size=0.30, stratify=df_trainval["label"], random_state=42)

    print("\nЗапуск конвейера PromptAlign...")
    summary = run_prompt_align(df_train, df_val, df_test, client)
    print("\nВычислительный конвейер успешно завершён.")