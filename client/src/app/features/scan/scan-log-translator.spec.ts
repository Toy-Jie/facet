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
