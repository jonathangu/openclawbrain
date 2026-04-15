# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb85648e6a75ae5a3ba6cc73943d9146864f7df475704ace40a5204fa142526`
- fixture hash: `sha256-7f13329cf1857fada1958a6fb5e614617a7842e6a6185ecb6a9d160264a3397f`
- score hash: `sha256-d5e8669d082912ba01e4d1581199b78a7f1f441ec0731c0e57ae16e8b499ca06`
- bundle hash: `sha256-d87cf4dd79bdf740b399674bfa8ca685e217447bbe1bf06cbd438fde6b4b7a56`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8dfb1b133ecf1f6249a4f0b0aded2fd5af80368793a13e3eb702ebb8c1e8fca |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f41c8be01deea76ae69e923e5881d025d5faeabca7c17b99ef7a2525a22012b5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a430d4f3df5cece5cc8f0b847c1179397d674c322ea69584c68759a18aa2d213 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-69aa3deec505e0108769a02e5cb5224bcfecb699024d91011daa5e86c7e14582 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f46d0c79 | sha256-af27bd61a537e038ca93ea217958d0ce77c6be766bfd06490da11298d6298fbd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f46d0c79 | sha256-e29539c72965856c122b66ad1e00a9e21ee76949e541c24b0bbf5217702e4d30 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f46d0c79 | sha256-af27bd61a537e038ca93ea217958d0ce77c6be766bfd06490da11298d6298fbd |
