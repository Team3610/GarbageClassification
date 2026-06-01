import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowRight,
  BadgeCheck,
  Camera,
  Info,
  Leaf,
  Loader2,
  Recycle,
  ScanSearch,
  Upload,
} from 'lucide-react';

import { HierarchicalClassifier } from './inference/classifier.js';
import { MODEL_PRESETS } from './inference/config.js';

const categories = [
  {
    name: 'Clothes',
    label: '의류',
    description: '의류, 헌 옷, 천이나 직물류',
    tone: 'bg-indigo-100 text-indigo-900',
  },
  {
    name: 'Glass',
    label: '유리',
    description: '유리병, 깨진 유리 등 유리류',
    tone: 'bg-sky-100 text-sky-900',
  },
  {
    name: 'Plastic',
    label: '플라스틱',
    description: '페트병, 플라스틱 용기, 비닐 및 포장재',
    tone: 'bg-cyan-100 text-cyan-900',
  },
  {
    name: 'Shoes',
    label: '신발',
    description: '운동화, 구두, 슬리퍼 등 신발류',
    tone: 'bg-stone-200 text-stone-900',
  },
  {
    name: 'Cardboard',
    label: '골판지',
    description: '택배 상자, 골판지처럼 두꺼운 종이류',
    tone: 'bg-orange-100 text-orange-900',
  },
  {
    name: 'Paper',
    label: '종이',
    description: '일반 종이, 신문지, 영수증, 노트 등 종이류',
    tone: 'bg-lime-100 text-lime-900',
  },
  {
    name: 'Metal',
    label: '금속',
    description: '알루미늄 캔, 고철, 철사 등 금속류',
    tone: 'bg-zinc-200 text-zinc-900',
  },
  {
    name: 'Battery',
    label: '배터리',
    description: '폐건전지, 보조배터리 등 배터리류',
    tone: 'bg-amber-100 text-amber-900',
  },
  {
    name: 'Biological',
    label: '유기성 폐기물',
    description: '음식물 쓰레기, 나뭇잎 등 유기성 폐기물',
    tone: 'bg-emerald-100 text-emerald-900',
  },
  {
    name: 'Trash',
    label: '일반 쓰레기',
    description: '위 항목에 해당하지 않는 일반 쓰레기',
    tone: 'bg-rose-100 text-rose-900',
  },
];

const categoryByName = Object.fromEntries(categories.map((c) => [c.name, c]));

const steps = [
  '이미지 업로드',
  '브라우저에서 전처리',
  '모델 추론',
  '분리수거 결과 확인',
];

const PRESET_KEYS = Object.keys(MODEL_PRESETS);

function App() {
  const [presetKey, setPresetKey] = useState('fp32Ensemble');
  const [modelStatus, setModelStatus] = useState({ state: 'idle', loaded: 0, total: 0 });
  const [modelError, setModelError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const classifierRef = useRef(null);
  const lastFileRef = useRef(null);

  const preset = MODEL_PRESETS[presetKey];

  useEffect(() => {
    let cancelled = false;
    setModelStatus({ state: 'loading', loaded: 0, total: 1 + preset.stage2.length });
    setModelError(null);
    classifierRef.current = null;

    const classifier = new HierarchicalClassifier(preset);
    classifier
      .load((progress) => {
        if (!cancelled) {
          setModelStatus({ state: 'loading', loaded: progress.loaded, total: progress.total });
        }
      })
      .then(() => {
        if (cancelled) return;
        classifierRef.current = classifier;
        setModelStatus({ state: 'ready', loaded: 1 + preset.stage2.length, total: 1 + preset.stage2.length });
        if (lastFileRef.current) {
          runPrediction(lastFileRef.current, classifier);
        }
      })
      .catch((error) => {
        if (cancelled) return;
        console.error(error);
        setModelError(error.message ?? '모델 로드에 실패했습니다.');
        setModelStatus({ state: 'error', loaded: 0, total: 1 + preset.stage2.length });
      });

    return () => {
      cancelled = true;
    };
  }, [presetKey]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function runPrediction(file, classifier) {
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const result = await classifier.predict(file);
      setPrediction(result);
    } catch (error) {
      console.error(error);
      setPredictionError(error.message ?? '추론에 실패했습니다.');
      setPrediction(null);
    } finally {
      setIsPredicting(false);
    }
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction(null);
    setPredictionError(null);
    lastFileRef.current = file;

    if (classifierRef.current) {
      runPrediction(file, classifierRef.current);
    }
  }

  const headerStatus = useMemo(() => {
    if (modelStatus.state === 'loading') {
      return `모델 로드 중 (${modelStatus.loaded}/${modelStatus.total})`;
    }
    if (modelStatus.state === 'error') return '모델 로드 실패';
    if (modelStatus.state === 'ready') return '모델 준비 완료';
    return '대기';
  }, [modelStatus]);

  const resultCategory = prediction?.isGarbage ? categoryByName[prediction.label] : null;
  const resultHeadline = !prediction
    ? '분석 대기 중'
    : prediction.isGarbage
      ? '분류 완료'
      : '쓰레기가 아닙니다';

  return (
    <div className="min-h-screen overflow-hidden bg-[#f7f7f2] text-[#18211f]">
      <div className="fixed inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(120deg,rgba(247,247,242,0.98)_0%,rgba(247,247,242,0.94)_48%,rgba(226,238,232,0.82)_100%)]" />
        <div className="absolute inset-x-0 top-0 h-px bg-white/80" />
      </div>

      <header className="relative z-10 flex items-center justify-between px-5 py-4 sm:px-8 lg:px-12">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#18211f] text-white">
            <Recycle size={20} strokeWidth={2.3} />
          </div>
          <div>
            <p className="text-sm font-semibold text-[#53615c]">GarbageClassification</p>
            <h1 className="text-lg font-bold tracking-normal">분리수거 이미지 분류</h1>
          </div>
        </div>

        <div className="hidden items-center gap-3 md:flex">
          <label className="flex items-center gap-2 rounded-full border border-[#cfd8d2] bg-white/75 px-3 py-2 text-xs font-semibold text-[#2d3935] shadow-sm backdrop-blur">
            <span className="text-[#687671]">모델</span>
            <select
              value={presetKey}
              onChange={(event) => setPresetKey(event.target.value)}
              className="bg-transparent text-xs font-semibold text-[#18211f] outline-none"
            >
              {PRESET_KEYS.map((key) => (
                <option key={key} value={key}>
                  {MODEL_PRESETS[key].label}
                </option>
              ))}
            </select>
          </label>
          <span className="inline-flex items-center gap-2 rounded-full border border-[#cfd8d2] bg-white/75 px-3 py-2 text-xs font-semibold text-[#2d3935] shadow-sm backdrop-blur">
            {modelStatus.state === 'loading' ? <Loader2 size={14} className="animate-spin" /> : <Info size={14} />}
            {headerStatus}
          </span>
        </div>
      </header>

      <main className="relative z-10 grid min-h-[calc(100vh-76px)] grid-cols-1 gap-8 px-5 pb-8 sm:px-8 lg:grid-cols-[minmax(0,1.08fr)_minmax(340px,0.92fr)] lg:px-12">
        <section className="flex min-h-[760px] flex-col justify-center gap-6 py-4 lg:min-h-0">
          <div className="max-w-3xl">
            <p className="mb-4 inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 text-sm font-semibold text-[#48645c] shadow-sm ring-1 ring-[#dce4df]">
              <Leaf size={16} />
              Client-side waste sorting assistant
            </p>
            <h2 className="max-w-3xl text-4xl font-black leading-tight tracking-normal text-[#101816] sm:text-5xl lg:text-6xl">
              사진 한 장으로 분리수거 방향을 빠르게 확인하세요.
            </h2>
            <p className="mt-5 max-w-2xl text-base leading-7 text-[#53615c] sm:text-lg">
              브라우저 내 ONNX Runtime Web으로 직접 추론합니다. 이미지를 업로드하면 Stage1 쓰레기 판별 후 Stage2 앙상블이 10개 분류 중 가장 가능성 높은 항목을 알려줍니다.
            </p>
          </div>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.08fr)_minmax(280px,0.92fr)]">
            <section className="rounded-[28px] border border-white/80 bg-white/82 p-4 shadow-[0_24px_70px_rgba(30,45,40,0.13)] backdrop-blur-xl sm:p-5">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <h3 className="text-xl font-bold">이미지 업로드</h3>
                  <p className="mt-1 text-sm text-[#687671]">jpg, png 이미지를 선택하면 즉시 분류가 시작됩니다.</p>
                </div>
                <button className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[#e7f4ed] text-[#176b45]">
                  <Upload size={19} />
                </button>
              </div>

              <label className="group flex min-h-[260px] cursor-pointer flex-col items-center justify-center gap-5 overflow-hidden rounded-[24px] border-2 border-dashed border-[#abc4b7] bg-[#f8fbf8] px-5 text-center transition hover:border-[#20885d] hover:bg-[#f2faf5]">
                <input
                  className="sr-only"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="업로드된 이미지"
                    className="max-h-[300px] w-auto rounded-2xl object-contain"
                  />
                ) : (
                  <>
                    <span className="flex h-16 w-16 items-center justify-center rounded-full bg-[#20352f] text-white shadow-lg shadow-emerald-900/15 transition group-hover:scale-105">
                      <Camera size={28} />
                    </span>
                    <span>
                      <strong className="block text-lg">이미지를 선택하거나 드래그하세요</strong>
                      <span className="mt-2 block text-sm leading-6 text-[#65746e]">
                        {modelStatus.state === 'ready'
                          ? '준비 완료 — 이미지를 올리면 바로 분류됩니다.'
                          : modelStatus.state === 'loading'
                            ? `모델 로드 중 (${modelStatus.loaded}/${modelStatus.total})`
                            : modelStatus.state === 'error'
                              ? '모델 로드에 실패했습니다. 새로고침 후 다시 시도하세요.'
                              : '잠시 후 다시 시도하세요.'}
                      </span>
                    </span>
                  </>
                )}
              </label>
              {modelError ? (
                <p className="mt-3 text-sm text-rose-600">{modelError}</p>
              ) : null}
              {predictionError ? (
                <p className="mt-3 text-sm text-rose-600">{predictionError}</p>
              ) : null}
            </section>

            <section className="rounded-[28px] border border-[#d9e2dc] bg-[#1d2a27] p-5 text-white shadow-[0_24px_70px_rgba(30,45,40,0.13)]">
              <div className="mb-6 flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold text-[#9ed7bd]">분류 결과</p>
                  <h3 className="mt-1 text-2xl font-bold">{isPredicting ? '분석 중...' : resultHeadline}</h3>
                </div>
                {isPredicting ? (
                  <Loader2 className="animate-spin text-[#9ed7bd]" size={28} />
                ) : (
                  <ScanSearch className="text-[#9ed7bd]" size={28} />
                )}
              </div>

              <div className="space-y-4">
                <div className="rounded-2xl bg-white/10 p-4 ring-1 ring-white/10">
                  <p className="text-sm text-[#c7d7d0]">예상 카테고리</p>
                  <p className="mt-2 text-3xl font-black">
                    {prediction
                      ? prediction.isGarbage
                        ? (resultCategory?.label ?? prediction.label)
                        : '비대상'
                      : '-'}
                  </p>
                  {prediction ? (
                    <p className="mt-1 text-xs text-white/70">
                      {prediction.isGarbage ? `${prediction.label} · ` : ''}
                      신뢰도 {(prediction.confidence * 100).toFixed(1)}%
                      {prediction.ensembleUsed ? ` · 앙상블 ${prediction.stage2ModelCount}개` : ''}
                    </p>
                  ) : null}
                </div>
                <div className="rounded-2xl bg-white/10 p-4 ring-1 ring-white/10">
                  <p className="text-sm text-[#c7d7d0]">배출 안내</p>
                  <p className="mt-2 text-sm leading-6 text-white/86">
                    {prediction
                      ? prediction.isGarbage
                        ? (resultCategory?.description ?? '재질에 맞춰 분리수거 해주세요.')
                        : '쓰레기로 인식되지 않았습니다. 다른 이미지로 시도해보세요.'
                      : '이미지를 업로드하면 재질별 분류 결과와 간단한 배출 팁이 표시됩니다.'}
                  </p>
                  {prediction ? (
                    <p className="mt-3 text-xs text-white/60">
                      Stage1 {prediction.stage1LatencyMs.toFixed(0)}ms · Stage2 {prediction.stage2LatencyMs.toFixed(0)}ms
                    </p>
                  ) : null}
                </div>
              </div>
            </section>
          </div>

          <section className="rounded-[28px] border border-[#d9e2dc] bg-white/74 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 flex items-center gap-2">
              <BadgeCheck className="text-[#1d7d58]" size={20} />
              <h3 className="text-lg font-bold">작동 흐름</h3>
            </div>
            <div className="grid gap-3 sm:grid-cols-4">
              {steps.map((step, index) => (
                <div key={step} className="flex items-center justify-between rounded-2xl bg-[#f0f5f1] px-4 py-3 text-sm font-semibold text-[#33413d]">
                  <span>{step}</span>
                  {index < steps.length - 1 ? <ArrowRight size={16} className="hidden text-[#7d8d86] sm:block" /> : null}
                </div>
              ))}
            </div>
          </section>
        </section>

        <aside className="flex min-h-[520px] flex-col justify-center gap-5 pb-4 lg:min-h-0">
          <div className="rounded-[28px] border border-[#d9e2dc] bg-white/74 p-5 shadow-sm backdrop-blur">
            <div className="mb-5 flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-[#60716b]">분류 가능 항목</p>
                <h3 className="mt-1 text-2xl font-black">10개 분류 클래스</h3>
              </div>
              <Recycle className="text-[#1f7a57]" size={28} />
            </div>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              {categories.map((category) => (
                <div key={category.name} className="rounded-2xl bg-[#f6faf7] p-3 ring-1 ring-[#dfe8e2]">
                  <div className="flex items-center justify-between gap-2">
                    <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${category.tone}`}>
                      {category.label}
                    </span>
                    <span className="text-xs font-semibold text-[#82918b]">{category.name}</span>
                  </div>
                  <p className="mt-2 text-sm leading-5 text-[#4f5f59]">{category.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[32px] border border-[#d9e2dc] bg-[#172522] p-5 text-white shadow-[0_24px_70px_rgba(30,45,40,0.16)]">
            <div className="mb-5">
              <p className="text-sm font-semibold text-[#9ed7bd]">향후 Spline 적용 위치</p>
              <h3 className="mt-1 text-2xl font-black">분류 결과 3D 뷰어</h3>
              <p className="mt-3 text-sm leading-6 text-white/72">
                Spline은 배경 장식이 아니라, 업로드 이미지가 분류된 뒤 결과물을 회전하며 확인하는 전용 뷰어에 연결할 예정입니다.
              </p>
            </div>

            <div className="flex aspect-[4/3] items-center justify-center rounded-[24px] border border-white/10 bg-[linear-gradient(135deg,rgba(255,255,255,0.12),rgba(255,255,255,0.04))] p-6">
              <div className="text-center">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-white/12 text-[#a8e0c6]">
                  <ScanSearch size={28} />
                </div>
                <p className="mt-4 text-base font-bold">3D preview reserved</p>
                <p className="mt-2 text-sm leading-6 text-white/64">분석 결과가 생기기 전에는 Spline scene을 로드하지 않습니다.</p>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
