type Locale = 'en' | 'zh' | string;

const LEVEL_LABELS: Record<string, string> = {
  DEBUG: '调试',
  INFO: '信息',
  WARNING: '警告',
  ERROR: '错误',
  CRITICAL: '严重错误',
};

const PHASE_LABELS_ZH: Record<string, string> = {
  scoring: '照片评分',
  bursts: '连拍与重复检测',
  tagging: '自动标签',
  vec: '语义搜索索引',
  done: '完成',
};

type TranslationRule = {
  pattern: RegExp;
  replace: (...matches: string[]) => string;
};

function scanStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    completed: '已完成',
    failed: '失败',
    interrupted: '已中断',
  };
  return labels[status] ?? status;
}

const RULES_ZH: TranslationRule[] = [
  {
    pattern: /^Multi-pass processing:\s+(.+)$/,
    replace: (_, progress) => `多阶段处理：${progress}`,
  },
  {
    pattern: /^Config validation passed: all (\d+) categories have valid weight totals$/,
    replace: (_, count) => `配置校验通过：${count} 个分类的权重总和有效`,
  },
  {
    pattern: /^Config version: ([\w.-]+)$/,
    replace: (_, version) => `配置版本：${version}`,
  },
  {
    pattern: /^Using cpu$/i,
    replace: () => '正在使用 CPU',
  },
  {
    pattern: /^Using default SimpleTokenizer\.$/,
    replace: () => '正在使用默认 SimpleTokenizer。',
  },
  {
    pattern: /^Warning: You are sending unauthenticated requests to the HF Hub\. Please set a HF_TOKEN to enable higher rate limits and faster downloads\.$/,
    replace: () => '警告：正在向 Hugging Face Hub 发送未认证请求。请设置 HF_TOKEN，以获得更高请求限额和更快下载速度。',
  },
  {
    pattern: /^Using (.+)$/,
    replace: (_, device) => `正在使用 ${device}`,
  },
  {
    pattern: /^Auto-detecting VRAM profile: No GPU detected, ([\d.]+)GB RAM - legacy profile \(TOPIQ \+ SAMP-Net on CPU\)$/,
    replace: (_, ram) => `自动检测运行配置：未检测到 GPU，内存 ${ram}GB，使用 CPU 兼容模式（TOPIQ + SAMP-Net）`,
  },
  {
    pattern: /^Tip: run 'python facet\.py --doctor' for GPU setup diagnostics$/,
    replace: () => "提示：如需诊断 GPU 设置，可运行 'python facet.py --doctor'",
  },
  {
    pattern: /^Multi-pass mode: skipping eager GPU model loading \(profile: ([^)]+)\)$/,
    replace: (_, profile) => `多阶段模式：暂不预先加载 GPU 模型（配置：${profile}）`,
  },
  {
    pattern: /^Could not create photos_vec: no such module: vec0$/,
    replace: () => '未能创建语义搜索向量表：缺少 vec0 模块。普通扫描和评分不受影响。',
  },
  {
    pattern: /^Scan database file: (.+) \(resolved to (.+), (\d+) photos, ([^)]+)\)$/,
    replace: (_, rawPath, resolvedPath, count, size) =>
      `扫描数据库：${rawPath}（实际路径：${resolvedPath}，已有 ${count} 张照片，${size}）`,
  },
  {
    pattern: /^Scan database file: (.+) \((\d+) photos, ([^)]+)\)$/,
    replace: (_, dbPath, count, size) =>
      `扫描数据库：${dbPath}（已有 ${count} 张照片，${size}）`,
  },
  {
    pattern: /^Found (\d+) total, processing (\d+) new files\.$/,
    replace: (_, total, count) => `发现 ${total} 个文件，本次需要处理 ${count} 个新文件。`,
  },
  {
    pattern: /^No new files to process\.$/,
    replace: () => '没有新文件需要处理。',
  },
  {
    pattern: /^All tasks complete\.$/,
    replace: () => '所有任务已完成。',
  },
  {
    pattern: /^All models unloaded$/,
    replace: () => '所有模型已卸载',
  },
  {
    pattern: /^Tagged (\d+) photos with missing tags\.$/,
    replace: (_, count) => `已为 ${count} 张缺少标签的照片补充标签。`,
  },
  {
    pattern: /^No new tags assigned \(all photos already tagged, or none cleared the similarity threshold\)\.$/,
    replace: () => '没有新增标签：所有照片已有标签，或相似度未达到阈值。',
  },
  {
    pattern: /^Auto-populate of photos_vec failed \(non-fatal\)$/,
    replace: () => '自动填充语义搜索向量表失败（不影响扫描结果）。',
  },
  {
    pattern: /^Scan summary$/,
    replace: () => '扫描摘要',
  },
  {
    pattern: /^Scored:\s+(\d+)$/,
    replace: (_, count) => `已评分：${count}`,
  },
  {
    pattern: /^Bursts \(non-lead, hidden\):\s+(\d+)$/,
    replace: (_, count) => `连拍非主图（已隐藏）：${count}`,
  },
  {
    pattern: /^Duplicates \(non-lead, hidden\):\s+(\d+)$/,
    replace: (_, count) => `重复照片非主图（已隐藏）：${count}`,
  },
  {
    pattern: /^Blinks \(hidden\):\s+(\d+)$/,
    replace: (_, count) => `闭眼照片（已隐藏）：${count}`,
  },
  {
    pattern: /^RAW paired w\/ JPEG \(skipped\):\s+(\d+)$/,
    replace: (_, count) => `已有对应 JPEG 的 RAW（已跳过）：${count}`,
  },
  {
    pattern: /^Processing (\d+) photos \(initial chunk size: (\d+), auto-tuning: (.+)\)\.\.\.$/,
    replace: (_, count, chunkSize, autoTuning) =>
      `正在处理 ${count} 张照片（初始分块大小：${chunkSize}，自动调节：${autoTuning}）...`,
  },
  {
    pattern: /^Processing burst groups \(rapid<=([\d.]+)s, similarity>=(\d+)%, window=([\d.]+)min\)\.\.\.$/,
    replace: (_, rapid, similarity, window) =>
      `正在处理连拍分组（快速连拍 <= ${rapid} 秒，相似度 >= ${similarity}%，时间窗口 ${window} 分钟）...`,
  },
  {
    pattern: /^Assigned (\d+) burst groups$/,
    replace: (_, count) => `已分配 ${count} 个连拍分组`,
  },
  {
    pattern: /^Batch Processing Complete$/,
    replace: () => '批量处理完成',
  },
  {
    pattern: /^Total: (\d+) images in (.+) \(([\d.]+) img\/s\)$/,
    replace: (_, count, duration, speed) => `总计：${count} 张图片，用时 ${duration}（${speed} 张/秒）`,
  },
  {
    pattern: /^Memory peak: ([\d.]+) GB \| GPU peak: ([\d.]+) GB$/,
    replace: (_, memory, gpu) => `内存峰值：${memory} GB | GPU 峰值：${gpu} GB`,
  },
  {
    pattern: /^RAM cache: (\d+)\/(\d+) hits \((\d+)%\)$/,
    replace: (_, hits, total, percent) => `内存缓存：${hits}/${total} 次命中（${percent}%）`,
  },
  {
    pattern: /^Multi-pass processing complete:$/,
    replace: () => '多阶段处理完成：',
  },
  {
    pattern: /^Images:\s+(\d+)$/,
    replace: (_, count) => `图片数：${count}`,
  },
  {
    pattern: /^Chunks:\s+(\d+)$/,
    replace: (_, count) => `分块数：${count}`,
  },
  {
    pattern: /^Passes:\s+(\d+)$/,
    replace: (_, count) => `执行阶段数：${count}`,
  },
  {
    pattern: /^Total time:\s+([\d.]+)s$/,
    replace: (_, seconds) => `总耗时：${seconds} 秒`,
  },
  {
    pattern: /^Throughput:\s+([\d.]+) img\/s$/,
    replace: (_, speed) => `处理速度：${speed} 张/秒`,
  },
  {
    pattern: /^Time breakdown:$/,
    replace: () => '耗时明细：',
  },
  {
    pattern: /^I\/O:\s+([\d.]+)s \((\d+)%\)$/,
    replace: (_, seconds, percent) => `文件读取：${seconds} 秒（${percent}%）`,
  },
  {
    pattern: /^Model load:\s+([\d.]+)s \((\d+)%\)$/,
    replace: (_, seconds, percent) => `模型加载：${seconds} 秒（${percent}%）`,
  },
  {
    pattern: /^Inference:\s+([\d.]+)s \((\d+)%\)$/,
    replace: (_, seconds, percent) => `模型推理：${seconds} 秒（${percent}%）`,
  },
  {
    pattern: /^Model unload:\s+([\d.]+)s \((\d+)%\)$/,
    replace: (_, seconds, percent) => `模型卸载：${seconds} 秒（${percent}%）`,
  },
  {
    pattern: /^Scan run #(\d+) finished: (.+)$/,
    replace: (_, id, status) => `扫描任务 #${id} 已结束：${scanStatusLabel(status)}`,
  },
  {
    pattern: /^Parsing model identifier\. Schema: (.+), Identifier: (.+)$/,
    replace: (_, schema, identifier) => `正在解析模型标识：Schema=${schema}，Identifier=${identifier}`,
  },
  {
    pattern: /^Loaded built-in (.+) model config\.$/,
    replace: (_, model) => `已加载内置 ${model} 模型配置。`,
  },
  {
    pattern: /^HTTP Request: (\w+) (.+) "(HTTP\/[\d.]+ \d+ .+)"$/,
    replace: (_, method, url, status) => `HTTP 请求：${method} ${url}（${status}）`,
  },
  {
    pattern: /^Instantiating model architecture: (.+)$/,
    replace: (_, architecture) => `正在创建模型架构：${architecture}`,
  },
  {
    pattern: /^Loading full pretrained weights from: (.+)$/,
    replace: (_, path) => `正在加载完整预训练权重：${path}`,
  },
  {
    pattern: /^Final image preprocessing configuration set: (.+)$/,
    replace: (_, config) => `最终图像预处理配置：${translatePreprocessConfig(config)}`,
  },
  {
    pattern: /^Model (.+) creation process complete\.$/,
    replace: (_, model) => `模型 ${model} 创建完成。`,
  },
  {
    pattern: /^Parsing tokenizer identifier\. Schema: (.+), Identifier: (.+)$/,
    replace: (_, schema, identifier) => `正在解析分词器标识：Schema=${schema}，Identifier=${identifier}`,
  },
  {
    pattern: /^Attempting to load config from built-in: (.+)$/,
    replace: (_, config) => `正在尝试从内置配置加载：${config}`,
  },
  {
    pattern: /^Traceback \(most recent call last\):$/,
    replace: () => '异常追踪（最近一次调用）：',
  },
  {
    pattern: /^File "([^"]+)", line (\d+), in (.+)$/,
    replace: (_, path, line, fn) => `文件 "${path}"，第 ${line} 行，位于 ${fn}`,
  },
  {
    pattern: /^sqlite3\.OperationalError: no such module: vec0$/,
    replace: () => 'SQLite 操作错误：缺少 vec0 模块',
  },
  {
    pattern: /^SQLite OperationalError: no such module: vec0$/,
    replace: () => 'SQLite 操作错误：缺少 vec0 模块',
  },
  {
    pattern: /^Path does not exist: (.+)$/,
    replace: (_, path) => `路径不存在：${path}`,
  },
  {
    pattern: /^Unsupported file type: (.+)$/,
    replace: (_, path) => `不支持的文件类型：${path}`,
  },
];

function isChinese(locale: Locale): boolean {
  return locale.toLowerCase().startsWith('zh');
}

function translatePreprocessConfig(config: string): string {
  return config
    .replace(/'size'/g, "'尺寸'")
    .replace(/'mode'/g, "'模式'")
    .replace(/'mean'/g, "'均值'")
    .replace(/'std'/g, "'标准差'")
    .replace(/'interpolation'/g, "'插值'")
    .replace(/'resize_mode'/g, "'缩放模式'")
    .replace(/'fill_color'/g, "'填充颜色'")
    .replace(/'shortest'/g, "'短边优先'")
    .replace(/'bicubic'/g, "'双三次'");
}

function translateMessageToChinese(message: string): string {
  const leadingSpace = message.match(/^\s*/)?.[0] ?? '';
  const trimmed = message.trim();

  if (!trimmed || /^=+$/.test(trimmed) || /^-+$/.test(trimmed)) return message;

  for (const rule of RULES_ZH) {
    const match = trimmed.match(rule.pattern);
    if (match) return `${leadingSpace}${rule.replace(...match)}`;
  }

  return message;
}

export function scanPhaseLabel(phase: string | null | undefined, locale: Locale): string {
  if (!phase) return '-';
  return isChinese(locale) ? PHASE_LABELS_ZH[phase] ?? phase : phase;
}

export function translateScanLogLine(line: string, locale: Locale): string {
  if (!isChinese(locale)) return line;

  const match = line.match(
    /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+([A-Z]+)\s+\[[^\]]+\]\s+(.*)$/,
  );

  if (!match) return translateMessageToChinese(line);

  const [, timestamp, level, message] = match;
  const translatedLevel = LEVEL_LABELS[level] ?? level;
  return `${timestamp} ${translatedLevel}：${translateMessageToChinese(message)}`;
}

export function translateScanLog(lines: string[], locale: Locale): string {
  return lines.map(line => translateScanLogLine(line, locale)).join('\n');
}
