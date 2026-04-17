# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7095a4d9ce26969c4dde9c329e749be730ceb1f708c47df4f4c59a5abea7434f`
- fixture hash: `sha256-107b047d2badf45fec45fded8a1234ee55c336b1a2803fdeba6955f2f30cad1f`
- score hash: `sha256-c4599a5d08a095adf951e5a0bdeda6b0ce04330bc26d555f96f1e6cde6e180e4`
- bundle hash: `sha256-038b6a5c691b486ce83dd37eef2f2b7c47e717039a5de87e11339248a755a66f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8413d38761902f8b7b6bde87782ba48c8aa416069cad02d85c57f922d6bd4f24 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5a2325ef43bfc2ad73dd6d8060a4039033f96e275b1ecc1fee2614f79fdc85e8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f36884ea774f17fb176be4f03dcf51cb2ad1f4daacdebc24dddbb8a3cde43303 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-d67cdd3db26fa1bc20c666af9396c3e24db327a60be248b88e46376910c76777 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-359f8de4 | sha256-683e2c42141745a35e3b2e419ac0e4560b73e9694aaa16bc825cfbb9e6f8f648 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-359f8de4 | sha256-ea346610e8404fc8d4159a61f551be272193a76a4a31d9c557b6b0337e606cfa |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-e70cf8bf | sha256-ec4592878c494ed674f086ba4e2a418c6f80f6aabb72623b5c432096b22666af |
