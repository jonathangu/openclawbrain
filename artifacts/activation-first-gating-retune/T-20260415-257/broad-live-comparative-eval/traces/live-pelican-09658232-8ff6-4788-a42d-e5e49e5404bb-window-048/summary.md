# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048`
- winner mode: `learned_route`
- trace hash: `sha256-9c32a87b231e4d5848a772d9d1cb8d355e8b17c5c883fc0f1ca8776ef042ba2c`
- fixture hash: `sha256-66d4441e9cd89d5df06e129fcf70accf27e8123573950bf81a6f813e2979adc4`
- score hash: `sha256-00ca9dbb0a99e0f3a94474ad1427ad5de345afa63bfee3ef14006a58db2e7448`
- bundle hash: `sha256-6c1517fdf549999d6fd0bde035e6de061c59cba38454606def7a0c4a8e58e114`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b9d9843197c6ea9cf1bbaf94c65647f4ecfa1e2224f8678711a552cc896cd7e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-9824ecbd8103f72e52c694839bce3a246cc10202e2fd5435b38b73ffbea7216e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b757031a40080ad6bad4a7dc24ae2aae15873c180a46816f0b9bc155a7e2f0d7 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-08ab9abd61dcb3b32606cd51cd0617fa973eb7aaddd9abd409a88bc3c6a12461 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-18fc6eb6 | sha256-9c7365bd34189d0709fa183b00ac0d0dddf6fe2cadb8a70774ee9cc36658f807 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-18fc6eb6 | sha256-7a7dc39157949df7cfccb706188a37877e23e83f345756c53deb8a6cb5322c72 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-68175755 | sha256-f5a6e3867a1733b624918e055ca41c805622dfa5dfeca2c634711aa9cc937706 |
