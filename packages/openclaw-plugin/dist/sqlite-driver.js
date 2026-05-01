import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
export function openDatabase(filename) {
    try {
        const mod = require('better-sqlite3');
        const BetterSqlite3 = mod.default || mod;
        return { db: new BetterSqlite3(filename), engine: 'better-sqlite3' };
    }
    catch (error) {
        if (!isNativeBindingFailure(error))
            throw error;
        const sqlite = require('node:sqlite');
        return { db: new NodeSqliteAdapter(new sqlite.DatabaseSync(filename)), engine: 'node:sqlite' };
    }
}
export function isNativeBindingFailure(error) {
    const message = String(error?.message || error || '');
    return message.includes('Could not locate the bindings file') ||
        message.includes('No native build was found') ||
        message.includes('invalid ELF header') ||
        message.includes('was compiled against a different Node.js version') ||
        message.includes('Cannot find module') && message.includes('better-sqlite3');
}
class NodeSqliteAdapter {
    db;
    constructor(db) {
        this.db = db;
    }
    exec(sql) {
        this.db.exec(sql);
    }
    prepare(sql) {
        return this.db.prepare(sql);
    }
    pragma(sql, options = {}) {
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
    transaction(fn) {
        return () => {
            this.db.exec('BEGIN');
            try {
                const result = fn();
                this.db.exec('COMMIT');
                return result;
            }
            catch (error) {
                try {
                    this.db.exec('ROLLBACK');
                }
                catch { /* ignore rollback failure */ }
                throw error;
            }
        };
    }
    close() {
        this.db.close();
    }
}
function firstValue(row) {
    if (!row || typeof row !== 'object')
        return row;
    const values = Object.values(row);
    return values.length ? values[0] : undefined;
}
