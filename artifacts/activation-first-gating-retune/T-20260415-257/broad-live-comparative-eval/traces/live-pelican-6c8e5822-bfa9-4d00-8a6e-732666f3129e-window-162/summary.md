# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b02e3d7c43b0542a9708c97a4decb5ab50a7fecdb19a413e8ba04a6c6f24587b`
- fixture hash: `sha256-fc0fa875ed0ba10ef61e5e8b6c1b783878d38dd1c5525b62b1d2717e4e66617b`
- score hash: `sha256-63739ac67f4b740878e1488c338d057fa3ed850ef960d4b370248322c4bb313d`
- bundle hash: `sha256-0863f371e71859b41ffdfdce1f845a2047dc0ebeb9da4a7eb4968d7389efd1f0`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f8f3d7baac7ea624c59c2785d2ad8b5f8904cda6bfe17f914b150feacd473265 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1643ecc6436c5b074bd106bbbb9e02eb43b60f5c587e5b52fba085a1a78dc90c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-43d349ec913ab94035108133b75eabbf8c82c0ffe8ad12373568d4c00b2b4baf |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8fb85b2c4a9505bde765cd6807320fa2881c00e437c5ed2df7675f2055725ec0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-147a7095 | sha256-70094574a21f56ca3bbff952d04bec305b31e8f0ee6730310db056730e77c9e6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-147a7095 | sha256-f2411cfb4ea4dc0412c08fb077a6ac6fb311e4afe6b5124d34200cbfdd5ec25e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-6ce3b642 | sha256-9e67c065649cb6f63a296e63df3e9d50ffe5fda7d237a4a16d177f34fe16ea40 |
