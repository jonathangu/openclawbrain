import BetterSqlite3 from 'better-sqlite3';
export function nativeSqliteSmokeTest() {
    let db = null;
    try {
        db = new BetterSqlite3(':memory:');
        const row = db.prepare('select 1 as ok').get();
        if (row?.ok !== 1)
            throw new Error('sqlite select smoke test failed');
        db.exec('create virtual table x using fts5(content)');
        db.prepare('insert into x(content) values (?)').run('openclawbrain native sqlite smoke');
        const match = db.prepare("select content from x where x match 'openclawbrain'").get();
        if (match?.content !== 'openclawbrain native sqlite smoke')
            throw new Error('sqlite FTS5 smoke test failed');
        return {
            ok: true,
            nodeVersion: process.version,
            betterSqlite3: 'imported',
            fts5: true,
        };
    }
    catch (error) {
        return {
            ok: false,
            nodeVersion: process.version,
            betterSqlite3: 'failed',
            fts5: false,
            error: safeError(error),
        };
    }
    finally {
        try {
            db?.close();
        }
        catch { /* ignore close failure */ }
    }
}
function safeError(error) {
    const message = String(error?.message || error || 'unknown native sqlite failure');
    return message.replace(/\/Users\/[^\s)]+/g, '<local-path>').slice(0, 500);
}
