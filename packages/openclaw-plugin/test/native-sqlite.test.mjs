import assert from 'node:assert/strict';
import test from 'node:test';

import { nativeSqliteSmokeTest } from '../dist/native-sqlite.js';

test('sqlite driver can open memory db and create FTS5 table', () => {
  const result = nativeSqliteSmokeTest();
  assert.equal(result.ok, true, result.error || 'sqlite smoke failed');
  assert.match(result.sqliteEngine, /^(better-sqlite3|node:sqlite)$/);
  assert.equal(result.fts5, true);
  assert.match(result.nodeVersion, /^v/);
});
