import assert from 'node:assert/strict';
import test from 'node:test';

import { nativeSqliteSmokeTest } from '../dist/native-sqlite.js';

test('native sqlite binding can open memory db and create FTS5 table', () => {
  const result = nativeSqliteSmokeTest();
  assert.equal(result.ok, true, result.error || 'native sqlite smoke failed');
  assert.equal(result.betterSqlite3, 'imported');
  assert.equal(result.fts5, true);
  assert.match(result.nodeVersion, /^v/);
});
