import { createRequire } from 'node:module';

export interface StatementLike {
  run(...params: any[]): any;
  get(...params: any[]): any;
  all(...params: any[]): any[];
}

export interface DatabaseLike {
  exec(sql: string): void;
  prepare(sql: string): StatementLike;
  pragma(sql: string, options?: { simple?: boolean }): any;
  transaction<T>(fn: () => T): () => T;
  close(): void;
}

export interface OpenDatabaseResult {
  db: DatabaseLike;
  engine: 'better-sqlite3' | 'node:sqlite';
}

const require = createRequire(import.meta.url);

export function openDatabase(filename: string): OpenDatabaseResult {
  try {
    const mod = require('better-sqlite3');
    const BetterSqlite3 = mod.default || mod;
    return { db: new BetterSqlite3(filename), engine: 'better-sqlite3' };
  } catch (error: any) {
    if (!isNativeBindingFailure(error)) throw error;
    const sqlite = require('node:sqlite');
    return { db: new NodeSqliteAdapter(new sqlite.DatabaseSync(filename)), engine: 'node:sqlite' };
  }
}

export function isNativeBindingFailure(error: any) {
  const message = String(error?.message || error || '');
  return message.includes('Could not locate the bindings file') ||
    message.includes('No native build was found') ||
    message.includes('invalid ELF header') ||
    message.includes('was compiled against a different Node.js version') ||
    message.includes('Cannot find module') && message.includes('better-sqlite3');
}

class NodeSqliteAdapter implements DatabaseLike {
  private db: any;

  constructor(db: any) {
    this.db = db;
  }

  exec(sql: string) {
    this.db.exec(sql);
  }

  prepare(sql: string): StatementLike {
    return this.db.prepare(sql);
  }

  pragma(sql: string, options: { simple?: boolean } = {}) {
    const trimmed = sql.trim();
    if (/^user_version\s*=\s*\d+$/i.test(trimmed)) {
      this.db.exec(`PRAGMA ${trimmed}`);
      return undefined;
    }
    if (/^(journal_mode|synchronous|foreign_keys)\b/i.test(trimmed)) {
      const row = this.db.prepare(`PRAGMA ${trimmed}`).get();
      return options.simple ? firstValue(row) : row;
    }
    const row = this.db.prepare(`PRAGMA ${trimmed}`).get();
    return options.simple ? firstValue(row) : row;
  }

  transaction<T>(fn: () => T): () => T {
    return () => {
      this.db.exec('BEGIN');
      try {
        const result = fn();
        this.db.exec('COMMIT');
        return result;
      } catch (error) {
        try { this.db.exec('ROLLBACK'); } catch { /* ignore rollback failure */ }
        throw error;
      }
    };
  }

  close() {
    this.db.close();
  }
}

function firstValue(row: any) {
  if (!row || typeof row !== 'object') return row;
  const values = Object.values(row);
  return values.length ? values[0] : undefined;
}
