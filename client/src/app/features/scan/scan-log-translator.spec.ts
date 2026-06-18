import { describe, expect, it } from 'vitest';
import { scanPhaseLabel, translateScanLogLine } from './scan-log-translator';

describe('scan log translator', () => {
  it('translates common scan startup messages to Chinese', () => {
    expect(
      translateScanLogLine(
        '2026-06-19 05:54:28 INFO  [facet.config] Config validation passed: all 34 categories have valid weight totals',
        'zh',
      ),
    ).toBe('2026-06-19 05:54:28 信息：配置校验通过：34 个分类的权重总和有效');

    expect(
      translateScanLogLine(
        '2026-06-19 05:54:33 INFO  [facet.scorer] Using cpu',
        'zh',
      ),
    ).toBe('2026-06-19 05:54:33 信息：正在使用 CPU');
  });

  it('translates no-new-files scan results to Chinese', () => {
    expect(
      translateScanLogLine(
        '2026-06-19 05:54:34 INFO  [facet] Found 957 total, processing 0 new files.',
        'zh-CN',
      ),
    ).toBe('2026-06-19 05:54:34 信息：发现 957 个文件，本次需要处理 0 个新文件。');

    expect(
      translateScanLogLine(
        '2026-06-19 05:54:34 INFO  [facet] No new files to process.',
        'zh-CN',
      ),
    ).toBe('2026-06-19 05:54:34 信息：没有新文件需要处理。');
  });

  it('keeps paths and technical details where useful', () => {
    expect(
      translateScanLogLine(
        '2026-06-19 05:54:34 WARNING [facet.schema] Could not create photos_vec: no such module: vec0',
        'zh',
      ),
    ).toBe(
      '2026-06-19 05:54:34 警告：未能创建语义搜索向量表：缺少 vec0 模块。普通扫描和评分不受影响。',
    );
  });

  it('translates model and post-processing messages to Chinese', () => {
    expect(
      translateScanLogLine(
        'Multi-pass processing: 100%|██████████| 2/2 [00:09<00:00,  4.52s/it]',
        'zh',
      ),
    ).toBe('多阶段处理：100%|██████████| 2/2 [00:09<00:00,  4.52s/it]');

    expect(
      translateScanLogLine('2026-06-19 06:04:40 INFO  [facet] All models unloaded', 'zh'),
    ).toBe('2026-06-19 06:04:40 信息：所有模型已卸载');

    expect(
      translateScanLogLine('2026-06-19 06:04:40 INFO  [facet] RAM cache: 0/2 hits (0%)', 'zh'),
    ).toBe('2026-06-19 06:04:40 信息：内存缓存：0/2 次命中（0%）');

    expect(
      translateScanLogLine(
        '2026-06-19 06:04:40 INFO  [processing.scan_state] Scan run #5 finished: completed',
        'zh',
      ),
    ).toBe('2026-06-19 06:04:40 信息：扫描任务 #5 已结束：已完成');
  });

  it('translates burst grouping and model loading messages to Chinese', () => {
    expect(
      translateScanLogLine(
        '2026-06-19 06:04:40 INFO  [facet] Processing burst groups (rapid<=0.4s, similarity>=70%, window=0.8min)...',
        'zh',
      ),
    ).toBe(
      '2026-06-19 06:04:40 信息：正在处理连拍分组（快速连拍 <= 0.4 秒，相似度 >= 70%，时间窗口 0.8 分钟）...',
    );

    expect(
      translateScanLogLine('2026-06-19 06:04:40 INFO  [facet] Assigned 195 burst groups', 'zh'),
    ).toBe('2026-06-19 06:04:40 信息：已分配 195 个连拍分组');

    expect(
      translateScanLogLine(
        '2026-06-19 06:04:40 INFO  [open_clip] Parsing model identifier. Schema: None, Identifier: ViT-L-14',
        'zh',
      ),
    ).toBe('2026-06-19 06:04:40 信息：正在解析模型标识：Schema=None，Identifier=ViT-L-14');

    expect(
      translateScanLogLine(
        '2026-06-19 06:04:45 INFO  [open_clip] Loading full pretrained weights from: /Users/example/model.bin',
        'zh',
      ),
    ).toBe('2026-06-19 06:04:45 信息：正在加载完整预训练权重：/Users/example/model.bin');
  });

  it('translates HTTP and sqlite traceback summary messages to Chinese', () => {
    expect(
      translateScanLogLine(
        '2026-06-19 06:04:42 INFO  [httpx] HTTP Request: HEAD https://huggingface.co/model "HTTP/1.1 404 Not Found"',
        'zh',
      ),
    ).toBe(
      '2026-06-19 06:04:42 信息：HTTP 请求：HEAD https://huggingface.co/model（HTTP/1.1 404 Not Found）',
    );

    expect(
      translateScanLogLine('Traceback (most recent call last):', 'zh'),
    ).toBe('异常追踪（最近一次调用）：');

    expect(
      translateScanLogLine('sqlite3.OperationalError: no such module: vec0', 'zh'),
    ).toBe('SQLite 操作错误：缺少 vec0 模块');
  });

  it('does not translate English locale output', () => {
    const line = '2026-06-19 05:54:34 INFO  [facet] No new files to process.';

    expect(translateScanLogLine(line, 'en')).toBe(line);
  });

  it('translates progress phase names', () => {
    expect(scanPhaseLabel('scoring', 'zh')).toBe('照片评分');
    expect(scanPhaseLabel('tagging', 'zh')).toBe('自动标签');
    expect(scanPhaseLabel('scoring', 'en')).toBe('scoring');
  });
});
