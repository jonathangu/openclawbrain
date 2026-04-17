# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b02e3d7c43b0542a9708c97a4decb5ab50a7fecdb19a413e8ba04a6c6f24587b`
- fixture hash: `sha256-fc0fa875ed0ba10ef61e5e8b6c1b783878d38dd1c5525b62b1d2717e4e66617b`
- score hash: `sha256-56d29063d9421dde0a4335e25dbcbee1ae8b08b78e0e13d4d8cfe3d979aeba57`
- bundle hash: `sha256-cf3ae0cfc2c5c3beddba90fd9dba8186147fc6797e378aa53e99643467e23783`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d7e1d2006f051975eccb3d12d8637d252cfd557f386e012cf8b2230991ea7274 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f8c52636c3fbf845b18beceef7ab3ca1de37c4d370947c819220131c766f4bc |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d5e1f2bb769c3c793c36c8be41d184e4ddd3a527fd95f9125e2965843238d53b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c266514c | sha256-082ec0b1df75a02ec875eaac2720d5930faabe7706ca073e540fb073238757c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c266514c | sha256-283cb53fbca3d85ef5406495253cdb2568a7e42ba603cc4fb17b20f204157832 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1acf96f9 | sha256-0f5b942ad930369a5ea555de47b77c24af2af5f67f2187c2c07df4e73d507f8b |
