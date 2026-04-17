# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68581f69a97780aac278954522193e99993d4befdc39acceb8ff881974cc0178`
- fixture hash: `sha256-d2931cc864933b7e6af27eb1382872e22dbe9358020b6cefacd8fc78d2489792`
- score hash: `sha256-e67048705ebbe35b43ec47375a6c3896b52837d0c1c109882686f2b2dee18945`
- bundle hash: `sha256-d8175b0b90eeaccabbd76da50b64f3877ebd6f9f4287a9fb0235aafc066188f8`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-181208f7b843fa2c39286593bf1b96c7f44d97e1cb317cd9b55efb3be3bcccb4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0a3f427e84d0c7902d3d580bf798751eb4a4412fe6f8c6b369f345ab0f356a65 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1ee687671acdcb6fada288b2f9eb9f46f7dea68c9b4e7297e88699ec7fb735fe |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2f1c8ef2246040f999a72e8f81dd22bb578a25e8e8c48f8223c367c221d15c49 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c2a4f5c0 | sha256-da73188a9f79bc8df12e47de26977bf8af278e713d020846393e392008548ccc |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c2a4f5c0 | sha256-17b70505492a17d83b785ceef8476a0564cf791e55bd8d51851793d88d40c812 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-073b8f21 | sha256-126137f76e65406745bb021e8d8bfd4d4614a3ada30e781f864374dbce5621e8 |
