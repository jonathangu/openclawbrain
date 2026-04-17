# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7095a4d9ce26969c4dde9c329e749be730ceb1f708c47df4f4c59a5abea7434f`
- fixture hash: `sha256-107b047d2badf45fec45fded8a1234ee55c336b1a2803fdeba6955f2f30cad1f`
- score hash: `sha256-1724ebafaa983e980d84c0ce7578fb77d221a1c339e7d4ccda482e88400ee13d`
- bundle hash: `sha256-1f464faabd4eac33c6481cad1bbc3976b271cd1b8101dc84e538fddbfe4a3c23`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8413d38761902f8b7b6bde87782ba48c8aa416069cad02d85c57f922d6bd4f24 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4ba60317965be8d6717c10b20e2bc5c01f0a7ef02fc4c631b264e587bc1e33e9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2104d0148acf8c3cdaafcf8e526478db93c064cf241259d7282dbcbb92583726 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-b0d42be4671395e8316c0bbeddc8e64638652cbe07bf93074f76b71be2920304 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0caa900d | sha256-f041e6fb1cfda1d5c782c3d7e7f52346045e28e15e53841a0b09541afe04ae00 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0caa900d | sha256-a9e1ca45edbcae217200f4525c0613ad19a74b5c276f093efc42273290296226 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-be17fae8 | sha256-31fdd11f46645dd21c4b7252ad35a1b3e69e683cbe5520af7dbe28cdd3e329fb |
