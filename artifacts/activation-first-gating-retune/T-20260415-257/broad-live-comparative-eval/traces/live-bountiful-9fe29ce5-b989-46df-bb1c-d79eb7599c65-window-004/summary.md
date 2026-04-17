# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-74d7bfdc59dd3db31bb5515d27016afd37c92da2c7bc4eab4e2eb908b0aa9b0c`
- fixture hash: `sha256-cb31ecdb2e85be5c4d11a69e22103d461970f7e7d752e4a1e0598a4d80c4542a`
- score hash: `sha256-0c1d2070c1d6784acf76ac22aa8a680fd222eae3003301e6edc9adf946fcf8a9`
- bundle hash: `sha256-44f9f173ffeadbeee490a909b774462b96ea38376caff7f9b2fc1ddeb64cb3f9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-629a5787a7df922c0732d245b071087f795fa4094d553ac2f295095a8256f812 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b9e1aeec9a1711fc6573df957ad5ae12e0daa6eab5428dd97312a6f349104029 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a29309eb5e7a02988bee1e62655acdceb869b7ca207f9385927521cffa12a53f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c50f1f0d72c37de71e3e2eef07e640282c17255e8a8ff1eedb628b73a7efdf12 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-09962502 | sha256-38b9b130046fe4e171dea5ca699c35d7d9007303ea139f0f84c2e4f5bf2ac576 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-09962502 | sha256-8dededb0bf1778d9ba42a44b0260d750198eec4a79b582dd03053c90ccfa19b5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5a27d519 | sha256-fba665dce0f7fed70a0b0389894efba3f2e52ec136a0123fafa2c855100d08f8 |
