import { openDatabase } from './sqlite-driver.js';

export interface NativeSqliteSmokeResult {
  ok: boolean;
  nodeVersion: string;
  sqliteEngine: string;
  fts5: boolean;
  error?: string;
}

export function nativeSqliteSmokeTest(): NativeSqliteSmokeResult {
  const opened = openDatabase(':memory:');
  const db = opened.db;
  try {
    const row = db.prepare('select 1 as ok').get() as { ok?: number } | undefined;
    if (row?.ok !== 1) throw new Error('sqlite select smoke test failed');
    db.exec('create virtual table x using fts5(content)');
    db.prepare('insert into x(content) values (?)').run('openclawbrain native sqlite smoke');
    const match = db.prepare("select content from x where x match 'openclawbrain'").get() as { content?: string } | undefined;
    if (match?.content !== 'openclawbrain native sqlite smoke') throw new Error('sqlite FTS5 smoke test failed');
    return {
      ok: true,
      nodeVersion: process.version,
      sqliteEngine: opened.engine,
      fts5: true,
    };
  } catch (error: any) {
    return {
      ok: false,
      nodeVersion: process.version,
      sqliteEngine: opened.engine,
      fts5: false,
      error: safeError(error),
    };
  } finally {
    try { db.close(); } catch { /* ignore close failure */ }
  }
}

function safeError(error: any) {
  const message = String(error?.message || error || 'unknown native sqlite failure');
  return message.replace(/\/Users\/[^\s)]+/g, '<local-path>').slice(0, 500);
}
